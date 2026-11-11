"""M2 pilot extraction: one unit = one model run (GEFS 35-day or IFS-ENS, 00Z) with all its active NA/EP storms.

Per unit, written to data/pilot/<model>/<YYYYMMDDHH>/ (the unit counts as done only when done.json exists):
- tracks.parquet: per storm, member and lead, the tracked centre, pmin, vmax, gust, RMW, MSLP depth and inner-core
  (≤100 km) / ring (200-500 km) means of the other native fields.
- env.parquet: per storm and lead, SHIPS-style environment from the control member around the ensemble-mean centre.
- patches_<storm>.npz: float16 inner-core patches (members × leads × [msl, u10, v10, gust] × ±4°) and 16°×16°
  environment patches (fields × leads) from the control member, for a possible CNN later.

Run:   scripts/pilot.sh start [gefs|ifs|both]      (detached, capped, resumable; see scripts/pilot.sh)
Test:  .venv/bin/python -m tcic.pilot.extract --model ifs --init 2025-10-26 --dry-run
"""
import argparse
import io
import json
import time

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR
from tcic.io.runner import Guard, Runner, Unit, atomic_write_bytes
from tcic.pilot import cases
from tcic.pilot.grid import Region, crop_global, haversine_km, ring_mean
from tcic.pilot.readers import EcmwfControl, GefsControl, NativeReader
from tcic.pilot.tracker import track_members

PILOT_DIR = DATA_DIR / "pilot"
LEADS = cases.LEADS
MODELS = {
    "ifs": dict(native="ecmwf-ifs-ens-forecast-15-day-0-25-degree", control=EcmwfControl,
                vars={"msl": "pressure_reduced_to_mean_sea_level", "u10": "wind_u_10m", "v10": "wind_v_10m", "gust": "wind_gust_10m",
                      "t2m": "temperature_2m", "d2m": "dew_point_temperature_2m", "z500": "geopotential_height_500hpa",
                      "z850": "geopotential_height_850hpa", "z925": "geopotential_height_925hpa", "t850": "temperature_850hpa",
                      "t925": "temperature_925hpa"}),
    "gefs": dict(native="noaa-gefs-forecast-35-day", control=GefsControl,
                 vars={"msl": "pressure_reduced_to_mean_sea_level", "u10": "wind_u_10m", "v10": "wind_v_10m", "gust": "wind_gust_surface",
                       "t2m": "temperature_2m", "rh2m": "relative_humidity_2m", "pwat": "precipitable_water_atmosphere",
                       "z500": "geopotential_height_500hpa"}),
}
CORE_PATCH_VARS = ["msl", "u10", "v10", "gust"]
CORE_HALF_DEG, ENV_HALF_DEG = 4.0, 8.0
PATCH_SCALE = {"vo850": 1e5, "d200": 1e5, "q850": 1e3}  # keep float16 patches in a well-resolved range (1e-5 /s, g/kg)


def _margin(lead_h):  # read margin around the first guess grows with lead (ensemble spread)
    return 4.0 + lead_h / 48.0


def _dk_mpi(sst_c):  # DeMaria & Kaplan (1994) empirical maximum potential intensity, kt
    return 28.2 + 55.8 * np.exp(0.1813 * (sst_c - 30.0))


def environment(fields: dict, grid, lat, lon) -> dict:
    """SHIPS-style environment around (lat, lon) from global control-member fields at one lead."""
    if not np.isfinite(lat):
        return {}
    half = 10.0
    patch = {k: crop_global(v, grid, lat, lon, half) for k, v in fields.items()}
    n = patch["u200"].shape[0]
    rr = (np.arange(n) - n // 2) * grid.res
    la, lo = np.meshgrid(lat - rr, lon + rr, indexing="ij")
    d = haversine_km(lat, lon, la, lo)
    ring = lambda k, a, b: ring_mean(patch[k], d, a, b)
    e = {}
    for lev in (200, 500):
        du, dv = ring(f"u{lev}", 200, 800) - ring("u850", 200, 800), ring(f"v{lev}", 200, 800) - ring("v850", 200, 800)
        e[f"shear_{lev}_850"] = float(np.hypot(du, dv))
        e[f"shear_{lev}_850_dir"] = float(np.degrees(np.arctan2(du, dv)) % 360)
    e["u_steer"] = np.mean([ring(f"u{l}", 0, 600) for l in (200, 500, 850)])
    e["v_steer"] = np.mean([ring(f"v{l}", 0, 600) for l in (200, 500, 850)])
    e["rh500"], e["rh700"] = ring("r500", 200, 800), ring("r700", 200, 800)
    e["q850"] = ring("q850", 0, 500)
    e["t200"], e["t500"] = ring("t200", 0, 1000), ring("t500", 0, 1000)
    e["vo850"] = ring("vo850", 0, 500) * 1e5
    e["d200"] = ring("d200", 0, 1000) * 1e5 if "d200" in patch else np.nan
    e["tcw"] = ring("tcw" if "tcw" in patch else "tcwv", 0, 300)
    skt = ring("skt", 0, 100)
    e["sst_c"] = skt - 273.15 if skt > 100 else skt
    e["mpi_dk"] = float(_dk_mpi(e["sst_c"])) if np.isfinite(e["sst_c"]) else np.nan
    return e


def _divergence200(u, v, grid):
    """Horizontal divergence (1/s) of global fields on a north->south grid (for GEFS, which has no 'd')."""
    lat = np.radians(grid.lats(np.arange(grid.nlat)))[:, None]
    dx = np.radians(grid.res) * 6.371e6 * np.cos(lat)
    dy = np.radians(grid.res) * 6.371e6
    dudx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * np.maximum(dx, 1.0))
    dvdy = (np.roll(v, 1, axis=-2) - np.roll(v, -1, axis=-2)) / (2 * dy)  # rows go north->south
    return dudx + dvdy


def _read_many(reader, names, init, tiles, region, parallel: int | None = None) -> dict:
    """Read several variables at once (reads are latency-bound; the global throttle still caps the total rate)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    parallel = parallel or int(os.environ.get("TCIC_VAR_PARALLEL", "2"))
    with ThreadPoolExecutor(parallel) as ex:
        return dict(zip(names, ex.map(lambda v: reader.read(v, init, LEADS, tiles, region), names)))


def run_unit(model: str, init: pd.Timestamp, storms: pd.DataFrame, out_dir, dry_run: bool = False) -> dict:
    cfg = MODELS[model]
    reader = NativeReader(cfg["native"], cfg["vars"])
    ctrl = cfg["control"]()
    t0 = time.time()
    fields = ctrl.read(init, LEADS)
    if "d200" not in fields:
        fields["d200"] = np.stack([_divergence200(fields["u200"][k], fields["v200"][k], ctrl.grid) for k in range(len(LEADS))])
    t_ctrl = time.time() - t0
    tracks, envs, patches = [], [], {}
    for storm_id, s in storms.groupby("storm_id"):
        s = s.sort_values("lead")
        fg_lat, fg_lon = s["fg_lat"].to_numpy(), s["fg_lon"].to_numpy()
        tiles = reader.tiles_for(fg_lat, fg_lon, [_margin(h) for h in LEADS])
        region = reader.region_for(tiles)
        data = _read_many(reader, ["msl", "u10", "v10", "gust"], init, tiles, region)
        lat0, lon0 = float(s["lat0"].iloc[0]), float(s["lon0"].iloc[0])
        tr = track_members(region, data["msl"], data["u10"], data["v10"], data["gust"], lat0, lon0, fg_lat, fg_lon, LEADS)
        nm = tr["lat"].shape[0]
        df = pd.DataFrame({"member": np.repeat(np.arange(nm), len(LEADS)), "lead": np.tile(LEADS, nm),
                           **{k: v.ravel() for k, v in tr.items()}})
        # inner-core and ring means of the other native fields at each tracked centre
        core_patch = np.full((nm, len(LEADS), len(CORE_PATCH_VARS), 33, 33), np.nan, dtype=np.float16)
        la2, lo2 = np.meshgrid(region.lat, region.lon, indexing="ij")
        others = _read_many(reader, [v for v in cfg["vars"] if v not in data], init, tiles, region)
        for m in range(nm):
            for k in range(len(LEADS)):
                c_lat, c_lon = tr["lat"][m, k], tr["lon"][m, k]
                if not np.isfinite(c_lat):
                    continue
                win = region.window(c_lat, c_lon, CORE_HALF_DEG)
                if win is not None:
                    for i, v in enumerate(CORE_PATCH_VARS):
                        core_patch[m, k, i] = data[v][m, k][win] / (100.0 if v == "msl" else 1.0)  # msl in hPa (float16 max is 65504)
                dwin = haversine_km(c_lat, c_lon, la2[win] if win else la2, lo2[win] if win else lo2)
                idx = m * len(LEADS) + k
                for v, arr in others.items():
                    fld = arr[m, k][win] if win else arr[m, k]
                    df.loc[idx, f"core_{v}"] = ring_mean(fld, dwin, 0, 100)
                    df.loc[idx, f"ring_{v}"] = ring_mean(fld, dwin, 200, 500)
        df.insert(0, "storm_id", storm_id)
        tracks.append(df)
        # environment around the ensemble-mean centre (control member fields)
        mlat, mlon = np.nanmean(tr["lat"], axis=0), np.nanmean(tr["lon"], axis=0)
        env_patch = np.full((len(fields), len(LEADS), 2 * int(ENV_HALF_DEG / ctrl.grid.res) + 1, 2 * int(ENV_HALF_DEG / ctrl.grid.res) + 1),
                            np.nan, dtype=np.float16)
        names = sorted(fields)
        for k, h in enumerate(LEADS):
            e = environment({n: fields[n][k] for n in names}, ctrl.grid, mlat[k], mlon[k])
            envs.append({"storm_id": storm_id, "lead": h, "lat_mean": mlat[k], "lon_mean": mlon[k], "env_member": ctrl.member,
                         "n_members": int(np.isfinite(tr["lat"][:, k]).sum()), **e})
            if np.isfinite(mlat[k]):
                for i, n in enumerate(names):
                    env_patch[i, k] = crop_global(fields[n][k], ctrl.grid, mlat[k], mlon[k], ENV_HALF_DEG) * PATCH_SCALE.get(n, 1.0)
        patches[storm_id] = dict(core=core_patch, core_vars=np.array(["msl_hpa", "u10", "v10", "gust"]), env=env_patch,
                                 env_vars=np.array([f"{n}*{PATCH_SCALE[n]:g}" if n in PATCH_SCALE else n for n in names]),
                                 leads=np.array(LEADS), region_bounds=np.array([region.r0, region.r1, region.c0, region.c1]))
    tracks = pd.concat(tracks, ignore_index=True)
    tracks.insert(1, "init", init)
    env = pd.DataFrame(envs)
    env.insert(1, "init", init)
    info = {"model": model, "init": str(init), "storms": sorted(patches), "env_member": ctrl.member, "seconds": round(time.time() - t0, 1),
            "control_seconds": round(t_ctrl, 1),
            "tracked_frac_by_lead": {int(h): round(float(np.isfinite(g["lat"]).mean()), 2) for h, g in tracks.groupby("lead") if h % 24 == 0}}
    if dry_run:
        return {"tracks": tracks, "env": env, "patches": patches, "info": info}
    for name, frame in (("tracks", tracks), ("env", env)):
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        atomic_write_bytes(out_dir / f"{name}.parquet", buf.getvalue())
    for storm_id, p in patches.items():
        buf = io.BytesIO()
        np.savez_compressed(buf, **p)
        atomic_write_bytes(out_dir / f"patches_{storm_id}.npz", buf.getvalue())
    atomic_write_bytes(out_dir / "done.json", json.dumps(info, indent=1).encode())  # written last: marks the unit done
    return {k: info[k] for k in ("storms", "seconds")}


def units(model: str, start=None, end=None) -> list[Unit]:
    out = []
    for init, storms in cases.runs(model, start=pd.Timestamp(start) if start else cases.START).items():
        if (start and init < pd.Timestamp(start)) or (end and init > pd.Timestamp(end)):
            continue
        d = PILOT_DIR / model / f"{init:%Y%m%d%H}"
        out.append(Unit(f"{model}/{init:%Y%m%d%H}", d / "done.json", {"model": model, "init": init, "storms": storms, "dir": d}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ifs", "gefs", "both"], default="both")
    ap.add_argument("--init", help="single run (YYYY-MM-DD), for testing")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--dry-run", action="store_true", help="run one --init without writing anything")
    ap.add_argument("--hours", help="only work during these local hours, e.g. 20-8 (pauses outside them)")
    args = ap.parse_args()
    models = ["ifs", "gefs"] if args.model == "both" else [args.model]
    if args.dry_run:
        u = [x for x in units(models[0], args.init, args.init) if x.payload["init"] == pd.Timestamp(args.init)][0]
        r = run_unit(u.payload["model"], u.payload["init"], u.payload["storms"], u.payload["dir"], dry_run=True)
        print(json.dumps(r["info"], indent=1))
        print(r["tracks"].groupby(["storm_id", "lead"])[["vmax", "pmin"]].agg(["mean", "std", "count"]).round(1).iloc[::4].to_string())
        print(r["env"].round(1).iloc[::4].to_string())
        return
    all_units = []
    for m in models:  # IFS first: a partial night still yields one complete model
        all_units += units(m, args.start or (args.init and args.init), args.end or (args.init and args.init))
    hours = tuple(int(h) for h in args.hours.split("-")) if args.hours else None
    runner = Runner("pilot", PILOT_DIR, guard=Guard(max_load_frac=0.50, min_mem_avail_gb=32, min_disk_free_gb=200, active_hours=hours))
    runner.run(all_units, lambda u: run_unit(u.payload["model"], u.payload["init"], u.payload["storms"], u.payload["dir"]))


if __name__ == "__main__":
    main()
