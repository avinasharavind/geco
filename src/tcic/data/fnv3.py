"""Google DeepMind Weather Lab FNV3 cyclone ensemble (50 members), from the public download endpoint.

Historical data (relating to times >= 1 h in the past) is CC BY 4.0. Files: one CSV per initialisation, with every active
storm's member tracks (ATCF ids), intensity, pressure and wind radii. We use plain "FNV3": its 50-member mean matches the
GDMN aid in NHC's 2025 a-decks to within 1 kt (checked for Melissa 2025-10-26 00Z); FNV3P1/P2 are later revisions.
"""
import io

import pandas as pd

from tcic.config import INTERIM_DIR, RAW_DIR
from tcic.io.throttle import throttled_get

URL = "https://deepmind.google.com/science/weatherlab/download/cyclones/{m}/ensemble/paired/csv/{m}_{t:%Y_%m_%dT%H_%M}_paired.csv"
FNV3_DIR = RAW_DIR / "fnv3"


def download(inits, model: str = "FNV3") -> list:
    FNV3_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for t in inits:
        path = FNV3_DIR / f"{model}_{t:%Y_%m_%dT%H_%M}_paired.csv"
        if not path.exists():
            try:
                path.write_bytes(throttled_get(URL.format(m=model, t=t)))
            except Exception as e:  # noqa: BLE001 - no file for this init (no active storms) or transient error
                print(f"skip {t}: {e}")
                continue
        got.append(path)
    return got


def load(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            d = pd.read_csv(io.StringIO(p.read_text()), comment="#")
        except pd.errors.EmptyDataError:  # licence header only: no tracks at this init
            continue
        if d.empty:
            continue
        frames.append(pd.DataFrame({
            "storm_id": d["track_id"].astype(str), "init": pd.to_datetime(d["init_time"]), "member": d["sample"].astype(int),
            "lead": (pd.to_timedelta(d["lead_time"]).dt.total_seconds() / 3600).astype(int),
            "lat": d["lat"], "lon": d["lon"], "vmax": d["maximum_sustained_wind_speed_knots"], "pmin": d["minimum_sea_level_pressure_hpa"],
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[out["storm_id"].str.match(r"^(AL|EP)\d{2}\d{4}$")]


def build(year: int = 2025, hours=(0, 6, 12)) -> pd.DataFrame:
    a = pd.read_parquet(INTERIM_DIR / "adecks.parquet", columns=["storm_id", "init", "tech"], filters=[("tech", "==", "CARQ")])
    a = a[(a["init"].dt.year == year) & a["init"].dt.hour.isin(hours) & a["storm_id"].astype(str).str[2:4].astype(int).lt(50)]
    inits = sorted(a["init"].unique())
    d = load(download(pd.to_datetime(inits)))
    d.to_parquet(INTERIM_DIR / f"fnv3_{year}.parquet", index=False)
    return d


if __name__ == "__main__":
    d = build()
    print(f"{len(d):,} rows, {d['init'].nunique()} inits, {d['storm_id'].nunique()} storms, members {d['member'].nunique()}")


def load_test() -> pd.DataFrame:
    """FNV3 runs for all held-out test seasons (benchmark only), from the per-season parquet files that exist."""
    from tcic.config import TEST_SEASONS

    parts = [pd.read_parquet(p) for y in TEST_SEASONS if (p := INTERIM_DIR / f"fnv3_{y}.parquet").exists()]
    return pd.concat(parts, ignore_index=True)
