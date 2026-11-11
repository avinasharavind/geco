"""Early-cycle consensus features (M1.5): one row per (storm, synoptic time t0, lead).

Everything here is available by t0 + ~3 h:
- CARQ at t0 (operational initial intensity/position and the -6/-12/-24 h history).
- Interpolated ("I") early aids, with 6-h and 12-h versions merged (`atcf.merge_interpolated`).
- GEFS members AP01-30 from the run 6 h before t0, and ECMWF members EE01-50 from the run 12 h before t0, leads shifted to t0.
- Optionally, operational SHIPS predictors from the t0 run (computed along OFCI).
Real-time view (default output): NHC's public real-time a-decks carry no ECMWF aids (EMXI, EMNI, EE01-50), so every ECMWF column is
removed and the reference / aid-spread features are recomputed without ECMWF (`realtime`). The full table, with ECMWF, is kept as
consensus_features_with_ecmwf.parquet (comparison in docs/m15_realtime.md; script in archive/scripts/m15_realtime.py).
Targets come from the final b-deck best track; cases follow the NHC convention (TC at t0 and at the valid time).
"""
import numpy as np
import pandas as pd

from tcic.config import FEATURES_DIR, INTERIM_DIR, RI_THRESHOLD_KT
from tcic.data import land
from tcic.data.atcf import merge_interpolated
from tcic.evaluate import load_truth

DET_AIDS = ["AVNI", "AEMI", "HWFI", "HFAI", "HFBI", "HMNI", "CTCI", "EMXI", "EMNI", "EGRI", "CMCI", "NVGI", "DSHP", "LGEM"]
ENSEMBLES = {  # prefix -> (member techs, allowed run ages at t0 in hours; the newest allowed run is used)
    "gefs": ([f"AP{i:02d}" for i in range(1, 31)], (6, 12)),
    "ecens": ([f"EE{i:02d}" for i in range(1, 51)], (12, 18)),
}
LEADS = list(range(12, 169, 12))
SHIPS_COLS = ["shear", "shear_adj", "shear_dir", "sst", "pot_int", "t200", "vxt200", "the_dev", "rh_700_500", "model_vtx",
              "vor850", "div200", "tadv_700_850", "land_km", "storm_speed", "ohc"]
SHIPS_SCALARS = ["t12_vmax", "steer_p", "ir_std", "ir_pct", "ohc_climo", "ir_available"]


def _load_adecks() -> pd.DataFrame:
    members = [t for techs, _ in ENSEMBLES.values() for t in techs]
    from tcic.data.atcf import INTERPOLATED

    keep = set(DET_AIDS) | set(INTERPOLATED.values()) | set(members) | {"CARQ"}
    a = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", sorted(keep))])
    a["storm_id"] = a["storm_id"].astype(str)
    a["tech"] = a["tech"].astype(str)
    a["tau"] = a["tau"].astype(int)
    return merge_interpolated(a)


def _carq(a: pd.DataFrame) -> pd.DataFrame:
    c = a[a["tech"] == "CARQ"].pivot_table(index=["storm_id", "init"], columns="tau", values=["vmax", "mslp", "lat", "lon"])
    out = pd.DataFrame(index=c.index)
    out["vmax0"] = c[("vmax", 0)]
    out["pmin0"] = c[("mslp", 0)]
    out["lat0"] = c[("lat", 0)]
    out["lon0"] = c[("lon", 0)]
    for h in (6, 12, 24):
        if ("vmax", -h) in c:
            out[f"dv{h}"] = out["vmax0"] - c[("vmax", -h)]
    # storm motion over the past 12 h (km/h, eastward/northward)
    dlat = out["lat0"] - c[("lat", -12)]
    dlon = (out["lon0"] - c[("lon", -12)]) * np.cos(np.radians(out["lat0"]))
    out["u_motion"] = dlon * 111.2 / 12
    out["v_motion"] = dlat * 111.2 / 12
    out = out.dropna(subset=["vmax0", "lat0", "lon0"]).reset_index().rename(columns={"init": "t0"})
    return out[out["t0"].dt.hour % 6 == 0]


def _det_aids(a: pd.DataFrame) -> pd.DataFrame:
    d = a[a["tech"].isin(DET_AIDS) & a["tau"].isin([0] + LEADS + [l - 12 for l in LEADS])]
    w = d.pivot_table(index=["storm_id", "init", "tau"], columns="tech", values=["vmax", "mslp", "lat", "lon"])
    w.columns = [f"{t}_{v}" for v, t in w.columns]
    w = w.reset_index().rename(columns={"init": "t0", "tau": "lead"})
    prev = w.copy()
    prev["lead"] += 12
    prev = prev.set_index(["storm_id", "t0", "lead"])[[f"{t}_vmax" for t in DET_AIDS if f"{t}_vmax" in w]]
    w = w.set_index(["storm_id", "t0", "lead"])
    for c in prev.columns:
        w[c.replace("_vmax", "_dv12")] = w[c] - prev[c].reindex(w.index)
    return w.reset_index()


def _ensemble(a: pd.DataFrame, name: str, members: list[str], ages: tuple[int, ...]) -> pd.DataFrame:
    """Member statistics from the newest run whose age at t0 is in `ages`."""
    out = None
    for age in ages:
        part = _ensemble_one(a, name, members, age)
        part[f"{name}_age"] = age
        if out is None:
            out = part
        else:
            key = ["storm_id", "t0", "lead"]
            have = out.set_index(key).index
            out = pd.concat([out, part[~part.set_index(key).index.isin(have)]], ignore_index=True)
    return out


def _ensemble_one(a: pd.DataFrame, name: str, members: list[str], age_h: int) -> pd.DataFrame:
    e = a[a["tech"].isin(members)].copy()
    e["t0"] = e["init"] + pd.Timedelta(hours=age_h)
    e["lead"] = e["tau"] - age_h
    e = e[e["lead"] >= -24]
    v = e.pivot_table(index=["storm_id", "t0", "tech"], columns="lead", values="vmax")
    p = e.pivot_table(index=["storm_id", "t0", "tech"], columns="lead", values="mslp")
    rows = {}
    for lead in LEADS:
        if lead not in v:
            continue
        vl = v[lead]
        g = vl.groupby(level=["storm_id", "t0"])
        stats = pd.DataFrame({
            f"{name}_vmax_mean": g.mean(), f"{name}_vmax_std": g.std(), f"{name}_vmax_p10": g.quantile(0.1),
            f"{name}_vmax_p90": g.quantile(0.9), f"{name}_n": g.count(),
        })
        if lead - 24 in v:
            ri = (vl - v[lead - 24]) >= RI_THRESHOLD_KT
            stats[f"{name}_ri_frac"] = ri.where(vl.notna()).groupby(level=["storm_id", "t0"]).mean()
        if lead in p:
            stats[f"{name}_mslp_mean"] = p[lead].groupby(level=["storm_id", "t0"]).mean()
        if 0 in v:
            stats[f"{name}_vmax0_mean"] = v[0].groupby(level=["storm_id", "t0"]).mean()  # member intensity valid at t0
        if 0 in p:
            stats[f"{name}_mslp0_mean"] = p[0].groupby(level=["storm_id", "t0"]).mean()
        rows[lead] = stats
    out = pd.concat(rows, names=["lead"]).reset_index()
    return out


def build(with_ships: bool = True) -> pd.DataFrame:
    a = _load_adecks()
    base = _carq(a)
    # One row per (storm, t0, lead)
    rows = base.merge(pd.DataFrame({"lead": LEADS}), how="cross")
    rows = rows.merge(_det_aids(a), on=["storm_id", "t0", "lead"], how="left")
    for name, (members, age) in ENSEMBLES.items():
        ens = _ensemble(a, name, members, age)
        rows = rows.merge(ens, on=["storm_id", "t0", "lead"], how="left")
        rows[f"{name}_offset0"] = rows[f"{name}_vmax0_mean"] - rows["vmax0"]
    rows = rows.copy()  # defragment after many column inserts

    vcols = [f"{t}_vmax" for t in DET_AIDS if f"{t}_vmax" in rows]
    rows["n_aids"] = rows[vcols].notna().sum(1)
    rows["ref_vmax"] = rows[vcols].mean(1).where(rows["n_aids"] >= 2)  # IVCN-like equal-weight mean of available aids
    # Interpolated aids carry no pressure, so the pmin reference is the CARQ pressure at t0 plus the GEFS mean pressure change.
    rows["ref_pmin"] = rows["pmin0"] + (rows["gefs_mslp_mean"] - rows["gefs_mslp0_mean"])
    rows["aid_vmax_std"] = rows[vcols].std(1)
    rows["aid_vmax_range"] = rows[vcols].max(1) - rows[vcols].min(1)
    lat_c = rows[[c.replace("_vmax", "_lat") for c in vcols if c.replace("_vmax", "_lat") in rows]].mean(1)
    lon_c = rows[[c.replace("_vmax", "_lon") for c in vcols if c.replace("_vmax", "_lon") in rows]].mean(1)
    rows["lat_c"], rows["lon_c"] = lat_c, lon_c
    rows["dist2land_c"] = land.dist2land(lat_c.values, lon_c.values)
    rows["dist2land0"] = land.dist2land(rows["lat0"].values, rows["lon0"].values)
    rows["basin_ep"] = rows["storm_id"].str.startswith("EP").astype(int)
    rows["doy"] = rows["t0"].dt.dayofyear
    rows["season"] = rows["t0"].dt.year
    rows = rows.drop(columns=[c for c in rows if c.endswith(("_lat", "_lon"))])

    if with_ships:
        s = pd.read_parquet(INTERIM_DIR / "ships.parquet").rename(columns={"init": "t0", "tau": "lead"})
        per_lead = s[["storm_id", "t0", "lead"] + SHIPS_COLS].rename(columns={c: f"ships_{c}" for c in SHIPS_COLS})
        scal = s[s["lead"] == 0][["storm_id", "t0"] + SHIPS_COLS + SHIPS_SCALARS]
        scal = scal.rename(columns={c: f"ships0_{c}" for c in SHIPS_COLS + SHIPS_SCALARS})
        rows = rows.merge(per_lead, on=["storm_id", "t0", "lead"], how="left").merge(scal, on=["storm_id", "t0"], how="left")
        rows["ships_mpi_minus_ref"] = rows["ships_pot_int"] - rows["ref_vmax"]
        rows["has_ships"] = rows["ships0_sst"].notna().astype(int)

    # Targets (b-deck best track) and verification flags (NHC convention)
    truth = load_truth("nhc")
    rows["valid"] = rows["t0"] + pd.to_timedelta(rows["lead"], unit="h")
    tv = truth.rename(columns=lambda c: f"obs_{c}")
    rows = rows.join(tv, on=["storm_id", "valid"])
    rows = rows.join(truth[["is_tc"]].rename(columns={"is_tc": "obs0_is_tc"}), on=["storm_id", "t0"])
    rows["verifiable"] = rows["obs_is_tc"].fillna(False).astype(bool) & rows["obs0_is_tc"].fillna(False).astype(bool) & rows["obs_vmax"].notna()
    rows["ri"] = (rows["obs_dv24"] >= RI_THRESHOLD_KT).astype(int)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(FEATURES_DIR / "consensus_features_with_ecmwf.parquet", index=False)
    rows = realtime(rows)
    rows.to_parquet(FEATURES_DIR / "consensus_features.parquet", index=False)
    return rows


ECMWF_PREFIXES = ("EMXI", "EMNI", "ecens")


def realtime(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop all ECMWF inputs (absent from NHC's real-time public decks) and recompute the aid reference and spread without them."""
    rows = rows.drop(columns=[c for c in rows.columns if c.startswith(ECMWF_PREFIXES)]).copy()
    vcols = [f"{t}_vmax" for t in DET_AIDS if f"{t}_vmax" in rows]
    rows["n_aids"] = rows[vcols].notna().sum(1)
    rows["ref_vmax"] = rows[vcols].mean(1).where(rows["n_aids"] >= 2)
    rows["aid_vmax_std"] = rows[vcols].std(1)
    rows["aid_vmax_range"] = rows[vcols].max(1) - rows[vcols].min(1)
    if "ships_pot_int" in rows:
        rows["ships_mpi_minus_ref"] = rows["ships_pot_int"] - rows["ref_vmax"]
    return rows


if __name__ == "__main__":
    import sys

    df = build(with_ships="--no-ships" not in sys.argv)
    v = df[df["verifiable"]]
    print(f"{len(df):,} rows, {len(v):,} verifiable; {v[['storm_id','t0']].drop_duplicates().shape[0]:,} cycles; {df.shape[1]} columns")
    print("coverage among verifiable rows:")
    cov = v[[c for c in ("ref_vmax", "gefs_vmax_mean", "ecens_vmax_mean", "has_ships") if c in v]].notna().mean()
    if "has_ships" in v:
        cov["has_ships"] = v["has_ships"].mean()
    print(cov.round(3).to_string())
    print(v.groupby("season").size().to_string())
