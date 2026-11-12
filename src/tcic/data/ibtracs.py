"""IBTrACS v04r01 best tracks for the North Atlantic and East Pacific.

vmax = USA_WIND, falling back to WMO_WIND / 0.88 (10-min -> 1-min); pmin = USA_PRES, falling back to WMO_PRES (as in the paper).
Only 6-hourly synoptic times are kept. Storms are keyed by USA_ATCF_ID (e.g. AL132025) to join with ATCF decks.
"""
import numpy as np
import pandas as pd

from tcic.config import BASINS, INTERIM_DIR, RAW_DIR
from tcic.io.throttle import throttled_get

URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
IBTRACS_DIR = RAW_DIR / "ibtracs"
COLS = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "USA_ATCF_ID", "USA_STATUS",
        "LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES", "DIST2LAND"]


def download(basins=BASINS, refresh: bool = False):
    IBTRACS_DIR.mkdir(parents=True, exist_ok=True)
    for b in basins:
        path = IBTRACS_DIR / f"ibtracs.{b}.list.v04r01.csv"
        if refresh or not path.exists():
            path.write_bytes(throttled_get(URL.format(basin=b)))


def load(first_season: int = 2015, basins=BASINS) -> pd.DataFrame:
    """Tidy best track: one row per (storm_id, time)."""
    frames = []
    for b in basins:
        path = IBTRACS_DIR / f"ibtracs.{b}.list.v04r01.csv"
        # keep_default_na=False: the basin code "NA" must not become NaN
        df = pd.read_csv(path, usecols=COLS, skiprows=[1], keep_default_na=False, na_values=[" ", ""], low_memory=False)
        frames.append(df[df["SEASON"] >= first_season])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["BASIN"].isin(basins)]
    t = pd.to_datetime(df["ISO_TIME"])
    df = df[(t.dt.hour % 6 == 0) & (t.dt.minute == 0)].assign(time=t)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    vmax = num("USA_WIND").fillna(num("WMO_WIND") / 0.88)
    pmin = num("USA_PRES").fillna(num("WMO_PRES"))
    out = pd.DataFrame({
        "storm_id": df["USA_ATCF_ID"].str.upper(),
        "sid": df["SID"],
        "name": df["NAME"],
        "basin": df["BASIN"],
        "time": df["time"],
        "lat": num("LAT"),
        "lon": num("LON"),
        "vmax": vmax.astype("float32"),
        "pmin": pmin.astype("float32"),
        "status": df["USA_STATUS"].fillna(""),
        "nature": df["NATURE"],
        "dist2land_km": num("DIST2LAND"),
    })
    out = out[out["storm_id"].str.len() == 8]  # drop records without an ATCF id
    return out.drop_duplicates(["storm_id", "time"]).sort_values(["storm_id", "time"]).reset_index(drop=True)


def build(first_season: int = 2015) -> pd.DataFrame:
    download()
    df = load(first_season)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_DIR / "ibtracs.parquet", index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby(d["time"].dt.year)["storm_id"].nunique().to_string())
    print("missing vmax:", float(np.isnan(d["vmax"]).mean()))
