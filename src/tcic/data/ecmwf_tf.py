"""ECMWF ENS per-member tropical-cyclone tracks (BUFR "tf" files from scripts/fetch_ecmwf_tracks.py) -> tidy table.

One BUFR message per storm, one subset per ensemble member (plus the control
and the deterministic IFS run, flagged in `ftype`). Within a message the keys come in time order: the observed centre
(significance 1, one value), the analysis centre (4) and pressure, the max-wind location (3) and max 10-m wind, the wind radii, then
for each `timePeriod` (h) the forecast centre (1) with pressure and the max-wind location (3) with max wind. The keys are walked in
order rather than by fixed rank, so wind-radius blocks of any length are skipped safely.

Storm identifiers: "13L" -> AL13, "09E" -> EP09, "01C" -> CP01; ECMWF's own numbers for systems it is tracking before (or without)
an agency number are >= 70 and are kept as e.g. "70W" (basin "W", number 70).
    python -m tcic.data.ecmwf_tf      (rebuild data/interim/ecmwf_tf.parquet)
"""
import numpy as np
import pandas as pd

from tcic.config import INTERIM_DIR, RAW_DIR

SRC = RAW_DIR / "ecmwf_tf"
MS_TO_KT = 1.943844
BASIN = {"L": "AL", "E": "EP", "C": "CP"}
FTYPE = {4: "pf", 1: "cf", 0: "hres"}  # ensembleForecastType: perturbed members 1-50, control (member 51), deterministic IFS (52)


def _arr(h, key: str, n: int) -> np.ndarray:
    import eccodes

    v = np.asarray(eccodes.codes_get_array(h, key), dtype=float)
    v = np.full(n, v[0]) if v.size == 1 and n > 1 else v
    return np.where(v < -1e90, np.nan, v)


def read_file(path, agency_only: bool = False) -> pd.DataFrame:
    """All storms and members in one tf file: (init, ecmwf_id, member, ftype, lead, lat, lon, pmin, vmax[kt], lat_vmax, lon_vmax)."""
    import eccodes

    rows = []
    with open(path, "rb") as f:
        while (h := eccodes.codes_bufr_new_from_file(f)) is not None:
            try:
                eccodes.codes_set(h, "unpack", 1)
                n = eccodes.codes_get(h, "numberOfSubsets")
                sid = eccodes.codes_get(h, "stormIdentifier").strip()
                init = pd.Timestamp(*(int(eccodes.codes_get(h, k)) for k in ("year", "month", "day", "hour")))
                if agency_only and agency_id(sid, init.year) is None:  # most messages are ECMWF-only systems; skip before the key walk
                    continue
                mem = np.asarray(eccodes.codes_get_array(h, "ensembleMemberNumber"), dtype=int)
                mem = np.full(n, mem[0]) if mem.size == 1 and n > 1 else mem
                typ = np.asarray(eccodes.codes_get_array(h, "ensembleForecastType"), dtype=int)
                typ = np.full(n, typ[0]) if typ.size == 1 and n > 1 else typ
                it = eccodes.codes_bufr_keys_iterator_new(h)
                keys = []
                while eccodes.codes_bufr_keys_iterator_next(it):
                    keys.append(eccodes.codes_bufr_keys_iterator_get_name(it))
                eccodes.codes_bufr_keys_iterator_delete(it)
                lead, sig, step = 0, None, {}
                steps = {0: step}
                for k in keys:
                    base = k.split("#")[-1]
                    if base == "timePeriod":
                        lead = int(eccodes.codes_get_array(h, k)[0])
                        step = steps.setdefault(lead, {})
                    elif base == "meteorologicalAttributeSignificance":
                        sig = int(eccodes.codes_get_array(h, k)[0])
                    elif base in ("latitude", "longitude") and k.startswith("#") and sig in (1, 3, 4):
                        if sig == 1 and lead == 0:  # observed centre (one value), not a member value
                            continue
                        step[base[:3] + ("_vmax" if sig == 3 else "")] = _arr(h, k, n)
                    elif base == "pressureReducedToMeanSeaLevel":
                        step["pmin"] = _arr(h, k, n) / 100.0
                    elif base == "windSpeedAt10M":
                        step["vmax"] = _arr(h, k, n) * MS_TO_KT
                for ld, s in steps.items():
                    if "lat" not in s:
                        continue
                    d = pd.DataFrame({"member": mem, "ftype": pd.Series(typ).map(FTYPE).to_numpy(), "lead": ld, **{c: s.get(c, np.full(n, np.nan)) for c in
                                                                    ("lat", "lon", "pmin", "vmax", "lat_vmax", "lon_vmax")}})
                    rows.append(d.assign(init=init, ecmwf_id=sid))
            finally:
                eccodes.codes_release(h)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).dropna(subset=["lat", "lon"])
    out["lon"] = np.where(out["lon"] > 180, out["lon"] - 360, out["lon"])
    return out


def agency_id(ecmwf_id: str, year: int) -> str | None:
    """'13L' -> 'AL132024'; None for ECMWF-only numbers (>= 70) and other basins."""
    num, b = ecmwf_id[:-1], ecmwf_id[-1:]
    if not num.isdigit() or int(num) >= 70 or b not in BASIN:
        return None
    return f"{BASIN[b]}{int(num):02d}{year}"


def _read_agency(path) -> pd.DataFrame:
    d = read_file(path, agency_only=True)
    if not len(d):
        return d
    d["storm_id"] = [agency_id(s, y) for s, y in zip(d["ecmwf_id"], d["init"].dt.year)]
    return d.dropna(subset=["storm_id"])


def build(procs: int = 4) -> pd.DataFrame:
    """NHC-numbered storms (AL/EP/CP) only, all runs and members -> data/interim/ecmwf_tf.parquet."""
    from multiprocessing import Pool

    with Pool(procs) as pool:
        parts = pool.map(_read_agency, sorted(SRC.glob("*.bufr")), chunksize=16)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.to_parquet(INTERIM_DIR / "ecmwf_tf.parquet", index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(len(d), "rows;", d["init"].nunique(), "runs;", d["storm_id"].nunique(), "agency storms")
