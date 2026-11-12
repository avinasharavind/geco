"""Operational (real-time) SHIPS text output, archived by the NCAR RAL TC Guidance Project.

Each file is the SHIPS run for one synoptic time; predictors at 0-120/168 h are computed along the OFCI track (the previous
cycle's official forecast, interpolated), so everything in it is available at t0 + ~2 h: no future information.
We parse the per-lead predictor table and a few scalar header predictors; the RI-probability line is skipped
because its definition changes between years.
"""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from tcic.config import INTERIM_DIR, RAW_DIR
from tcic.io.throttle import throttled_get

BASE = "https://hurricanes.ral.ucar.edu/realtime/plots"
REGIONS = {"al": "northatlantic", "ep": "northeastpacific"}
SHIPS_DIR = RAW_DIR / "ships"

ROWS = {  # SHIPS row label -> column name
    "V (KT) NO LAND": "v_noland", "V (KT) LAND": "v_land", "V (KT) LGEM": "v_lgem",
    "SHEAR (KT)": "shear", "SHEAR ADJ (KT)": "shear_adj", "SHEAR DIR": "shear_dir", "SST (C)": "sst",
    "POT. INT. (KT)": "pot_int", "200 MB T (C)": "t200", "200 MB VXT (C)": "vxt200", "TH_E DEV (C)": "the_dev",
    "700-500 MB RH": "rh_700_500", "MODEL VTX (KT)": "model_vtx", "850 MB ENV VOR": "vor850", "200 MB DIV": "div200",
    "700-850 TADV": "tadv_700_850", "LAND (KM)": "land_km", "LAT (DEG N)": "lat", "LONG(DEG W)": "lon_w",
    "STM SPEED (KT)": "storm_speed", "HEAT CONTENT": "ohc",
}
_NAME = re.compile(r"^(\d{8})(AL|EP)(\d{2})(\d{2})_ships\.txt$")


def _links(url: str) -> list[str]:
    return sorted(set(re.findall(r'href="([^"?#]+)"', throttled_get(url).decode(errors="replace"))))


def download_season(year: int, basins=("al", "ep"), workers: int = 4) -> list:
    """Download all SHIPS text files for numbered NA/EP storms of one season into data/raw/ships/YYYY/."""
    out_dir = SHIPS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in basins:
        season_url = f"{BASE}/{REGIONS[b]}/{year}/"
        storms = [s.rstrip("/").split("/")[-1] for s in _links(season_url)]
        storms = sorted({s for s in storms if re.fullmatch(rf"{b}(\d{{2}}){year}", s) and int(s[2:4]) < 50})
        for s in storms:
            for name in _links(f"{season_url}{s}/stext/"):
                name = name.split("/")[-1]
                m = _NAME.match(name)
                # keep only files for this storm and season (directories sometimes hold stray files from other years)
                if m and m.group(2).lower() == b and m.group(3) == s[2:4] and m.group(4) == str(year)[2:]:
                    path = out_dir / name
                    if not path.exists():
                        jobs.append((f"{season_url}{s}/stext/{name}", path))

    def fetch(job):
        url, path = job
        try:
            path.write_bytes(throttled_get(url))
        except Exception as e:  # noqa: BLE001 - log and continue; a missing cycle is just a NaN later
            print(f"failed {url}: {e}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, jobs))
    return sorted(out_dir.glob("*_ships.txt"))


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return np.nan  # 'xx.x', 'N/A', 'LOST', ...


def parse(path) -> pd.DataFrame:
    """One row per lead time with the per-lead predictors plus header scalars repeated on every row."""
    m = _NAME.match(path.name)
    init = pd.to_datetime(m.group(1), format="%y%m%d%H")
    storm_id = f"{m.group(2)}{m.group(3)}{2000 + int(m.group(4))}"
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    leads = None
    cols = {}
    for line in lines:
        if line.startswith("TIME (HR)"):
            leads = [int(x) for x in line[len("TIME (HR)"):].split()]
            continue
        if leads is None:
            continue
        for label, name in ROWS.items():
            if line.startswith(label) and name not in cols:
                toks = line[len(label):].split()
                if len(toks) == len(leads):
                    cols[name] = [_num(t) for t in toks]
                break
        if line.strip().startswith("INDIVIDUAL CONTRIBUTIONS"):
            break
    if leads is None:
        return pd.DataFrame()
    df = pd.DataFrame(cols, index=pd.Index(leads, name="tau")).reset_index()
    if "lon_w" in df:
        df["lon"] = -df.pop("lon_w")

    def scalar(pattern):
        hit = re.search(pattern, text)
        return _num(hit.group(1)) if hit else np.nan

    df["t12_vmax"] = scalar(r"T-12 MAX WIND:\s*(-?\d+)")
    df["steer_p"] = scalar(r"PRESSURE OF STEERING LEVEL \(MB\):\s*(\d+)")
    df["ir_std"] = scalar(r"BRIGHTNESS TEMP\. STD DEV\.\s+50-200 KM RAD:\s*(-?[\d.]+)")
    df["ir_pct"] = scalar(r"% GOES IR PIXELS WITH T < -20 C\s+50-200 KM RAD:\s*(-?[\d.]+)")
    df["init_heading"] = scalar(r"INITIAL HEADING/SPEED \(DEG/KT\):\s*(\d+)/")
    df["init_speed"] = scalar(r"INITIAL HEADING/SPEED \(DEG/KT\):\s*\d+/\s*(\d+)")
    df["ohc_climo"] = float("OHC CLIMO USED" in text)
    df["ir_available"] = float("IR SAT DATA AVAILABLE" in text)
    df.insert(0, "init", init)
    df.insert(0, "storm_id", storm_id)
    return df


RI_MODELS = {"SHIPS-RII": "rii", "Logistic": "logistic", "Bayesian": "bayes", "Consensus": "cons", "DTOPS": "dtops", "SDCON": "sdcon"}


def parse_ri(path) -> dict:
    """The 'Matrix of RI probabilities' block (operational RI guidance; same layout 2018-2025): one dict with columns
    ri_<model>_<kt>_<h> in [0, 1]; 999% (not run) -> NaN. SDCON from ~2023 only."""
    m = _NAME.match(path.name)
    out = {"storm_id": f"{m.group(2)}{m.group(3)}{2000 + int(m.group(4))}", "t0": pd.to_datetime(m.group(1), format="%y%m%d%H")}
    lines = path.read_text(errors="replace").splitlines()
    thresholds = None
    for line in lines:
        if line.strip().startswith("RI (kt / h)"):
            thresholds = [t.strip().replace("/", "_") for t in line.split("|")[1:]]
            continue
        if thresholds is None:
            continue
        label = line.split(":")[0].strip()
        if label in RI_MODELS and ":" in line:
            vals = [_num(t.rstrip("%")) for t in line.split(":", 1)[1].split()]
            if len(vals) == len(thresholds):
                for th, v in zip(thresholds, vals):
                    out[f"ri_{RI_MODELS[label]}_{th}"] = np.nan if (np.isnan(v) or v > 100) else v / 100
        elif line.strip().startswith("##"):
            break
    return out


def build_ri(years=range(2018, 2026)) -> pd.DataFrame:
    """Operational SHIPS RI probabilities from the already-downloaded text files -> data/interim/ships_ri.parquet."""
    rows = [parse_ri(p) for y in years for p in sorted((SHIPS_DIR / str(y)).glob("*_ships.txt")) if _NAME.match(p.name)]
    df = pd.DataFrame(rows).drop_duplicates(["storm_id", "t0"])
    df.to_parquet(INTERIM_DIR / "ships_ri.parquet", index=False)
    return df


def build(years=range(2018, 2026)) -> pd.DataFrame:
    frames = []
    for y in years:
        for path in download_season(y):
            frames.append(parse(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["storm_id", "init", "tau"])
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_DIR / "ships.parquet", index=False)
    return df


if __name__ == "__main__":
    d = build()
    runs = d.drop_duplicates(["storm_id", "init"])
    print(f"{len(runs):,} SHIPS runs, {runs['storm_id'].nunique()} storms")
    print(runs.groupby(runs["init"].dt.year).size().to_string())
    print("max lead by year:", d.groupby(d["init"].dt.year)["tau"].max().to_dict())
    print("missing fraction (tau<=120):", d[d["tau"] <= 120].drop(columns=["storm_id", "init"]).isna().mean().round(3).to_dict())
