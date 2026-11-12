"""NHC ATCF a-deck / b-deck download and parsing.

Archive seasons live at ftp.nhc.noaa.gov/atcf/archive/YYYY/ (a{bb}{nn}{yyyy}.dat.gz, b{bb}{nn}{yyyy}.dat.gz);
the current season is in atcf/aid_public/ (a-decks, gzipped) and atcf/btk/ (b-decks).
"""
import gzip
import io
import re
from pathlib import Path

import pandas as pd

from tcic.config import ATCF_BASINS, RAW_DIR
from tcic.io.throttle import throttled_get

BASE = "https://ftp.nhc.noaa.gov/atcf"
ATCF_DIR = RAW_DIR / "atcf"

# Fixed leading columns of an ATCF a/b-deck line.
_COLS = ["basin", "cy", "init", "technum", "tech", "tau", "lat", "lon", "vmax", "mslp", "ty", "rad"]


def _listing(url: str) -> list[str]:
    html = throttled_get(url).decode()
    return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))


def download_season(year: int, kinds: str = "ab", basins=ATCF_BASINS, archive: bool = True) -> list[Path]:
    """Download a- and/or b-decks for one season into data/raw/atcf/YYYY/ (skips files already present)."""
    out_dir = ATCF_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [f"{BASE}/archive/{year}/"] if archive else [f"{BASE}/aid_public/", f"{BASE}/btk/"]
    pattern = re.compile(rf"^[{kinds}]({'|'.join(basins)})\d{{2}}{year}\.dat(\.gz)?$")
    paths = []
    for src in sources:
        for name in _listing(src):
            if not pattern.match(name):
                continue
            path = out_dir / name
            if not path.exists():
                path.write_bytes(throttled_get(src + name))
            paths.append(path)
    return paths


def _latlon(s: str) -> float:
    s = s.strip()
    if not s:
        return float("nan")
    v = int(s[:-1]) / 10.0
    return -v if s[-1] in "SW" else v


def read_deck(path: Path) -> pd.DataFrame:
    """Parse one a- or b-deck into one row per (init, tech, tau); wind-radii duplicate lines are dropped."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    rows = []
    for line in io.StringIO(raw.decode(errors="replace")):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9 or not parts[2].isdigit():
            continue
        rows.append((parts + [""] * len(_COLS))[: len(_COLS)])
    df = pd.DataFrame(rows, columns=_COLS)
    df["storm_id"] = df["basin"].str.upper() + df["cy"].str.zfill(2) + df["init"].str[:4]
    df["init"] = pd.to_datetime(df["init"], format="%Y%m%d%H")
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce").astype("Int64")
    df["lat"] = df["lat"].map(_latlon)
    df["lon"] = df["lon"].map(_latlon)
    for c in ("vmax", "mslp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df.loc[df[c] <= 0, c] = float("nan")
    df = df.drop_duplicates(["init", "tech", "tau"], keep="first")
    return df[["storm_id", "basin", "init", "tech", "tau", "lat", "lon", "vmax", "mslp", "ty"]].reset_index(drop=True)


# Interpolated (early) aids: the 6-h version "XXXI" and its 12-h fallback "XXX2" (built from the run two cycles back).
INTERPOLATED = {
    "AVNI": "AVN2", "AEMI": "AEM2", "HWFI": "HWF2", "HFAI": "HFA2", "HFBI": "HFB2", "HMNI": "HMN2", "CTCI": "CTC2",
    "EMXI": "EMX2", "EMNI": "EMN2", "EGRI": "EGR2", "UKMI": "UKM2", "UKXI": "UKX2", "CMCI": "CMC2", "NVGI": "NVG2",
}


def merge_interpolated(adecks: pd.DataFrame, pairs: dict = INTERPOLATED) -> pd.DataFrame:
    """Treat 6-h and 12-h interpolated aids as one early aid (as NHC verification does).

    For each (storm, init) where "XXXI" is missing but "XXX2" exists, the "XXX2" rows are relabelled "XXXI".
    The 6-h version is preferred when both exist. Other techs are returned unchanged.
    """
    tech = adecks["tech"].astype(str)
    keep = ~tech.isin(pairs.values())
    out = [adecks[keep]]
    for six, twelve in pairs.items():
        have6 = adecks.loc[tech == six, ["storm_id", "init"]].drop_duplicates()
        r12 = adecks[tech == twelve]
        r12 = r12.merge(have6, on=["storm_id", "init"], how="left", indicator=True)
        r12 = r12[r12["_merge"] == "left_only"].drop(columns="_merge")
        out.append(r12.assign(tech=six))
    merged = pd.concat(out, ignore_index=True)
    merged["tech"] = merged["tech"].astype(str)
    return merged


def build(years=range(2018, 2026)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse downloaded decks into data/interim/{adecks,bdecks}.parquet.

    b-decks keep only 6-hourly synoptic best-track points (non-synoptic landfall/peak points are dropped).
    """
    from tcic.config import INTERIM_DIR

    a, b = [], []
    for y in years:
        for path in sorted((ATCF_DIR / str(y)).glob("*.dat*")):
            (a if path.name.startswith("a") else b).append(read_deck(path))
    adecks = pd.concat(a, ignore_index=True)
    bdecks = pd.concat(b, ignore_index=True)
    bdecks = bdecks[(bdecks["tech"] == "BEST") & (bdecks["tau"] == 0) & (bdecks["init"].dt.hour % 6 == 0)]
    bdecks = bdecks.rename(columns={"init": "time", "mslp": "pmin"}).drop(columns=["tech", "tau"])
    for df in (adecks, bdecks):
        for c in ("storm_id", "basin", "ty") + (("tech",) if "tech" in df else ()):
            df[c] = df[c].astype("category")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    adecks.to_parquet(INTERIM_DIR / "adecks.parquet", index=False)
    bdecks.to_parquet(INTERIM_DIR / "bdecks.parquet", index=False)
    return adecks, bdecks


if __name__ == "__main__":
    a, b = build()
    print(f"a-decks: {len(a):,} rows, {a['tech'].nunique()} techs; b-decks: {len(b):,} best-track points, {b['storm_id'].nunique()} storms")
