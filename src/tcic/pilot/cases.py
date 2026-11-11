"""Pilot case list: every 00Z run (GEFS 35-day, IFS-ENS) from 2024-04 (GEFS optionally from 2020-10, see --start) with at least one numbered NA/EP storm in CARQ at init.

For each storm we keep the CARQ position/intensity at init and a first-guess track from the same run's own NHC tracker
output (AEMN for GEFS, EEMN for IFS-ENS, falling back to the deterministic model, then to CARQ-motion extrapolation).
The first guess only decides *where to read data*; the tracker finds each member's centre itself.
"""
import numpy as np
import pandas as pd

from tcic.config import INTERIM_DIR, RAW_DIR
from tcic.data.atcf import ATCF_DIR, read_deck

LEADS = list(range(0, 169, 6))
FIRST_GUESS = {"gefs": ["AEMN", "AVNO", "EEMN", "EMX"], "ifs": ["EEMN", "EMX", "AEMN", "AVNO"]}  # own model first, then any same-init track
START = pd.Timestamp("2024-04-02")  # default; the GEFS back-extension passes an earlier start (2020-10-01)


def _adecks_from(start: pd.Timestamp = START) -> pd.DataFrame:
    a = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("init", ">=", start)])
    extra = [read_deck(p) for p in sorted((ATCF_DIR / "2026").glob("a*.dat*"))]  # current season, not in the archive parquet
    a = pd.concat([a.astype({"storm_id": str, "tech": str, "basin": str, "ty": str})] + extra, ignore_index=True)
    a["tau"] = a["tau"].astype(int)
    return a[a["init"].dt.hour == 0]


def build(model: str, end: pd.Timestamp | None = None, start: pd.Timestamp = START) -> pd.DataFrame:
    """One row per (init, storm, lead) with CARQ at init and the first-guess position at that lead."""
    a = _adecks_from(start)
    if end is not None:
        a = a[a["init"] <= end]
    carq = a[(a["tech"] == "CARQ") & (a["tau"] == 0)].dropna(subset=["lat", "lon"])
    carq = carq[carq["storm_id"].str[2:4].astype(int) < 50]  # numbered storms only
    carq = carq[["storm_id", "init", "lat", "lon", "vmax", "mslp", "ty"]].rename(columns={"lat": "lat0", "lon": "lon0", "vmax": "vmax0", "mslp": "pmin0"})
    # CARQ -12 h position for motion-based extrapolation
    prev = a[(a["tech"] == "CARQ") & (a["tau"] == -12)][["storm_id", "init", "lat", "lon"]].rename(columns={"lat": "lat_m12", "lon": "lon_m12"})
    carq = carq.merge(prev, on=["storm_id", "init"], how="left")
    rows = carq.merge(pd.DataFrame({"lead": LEADS}), how="cross")
    rows["fg_lat"], rows["fg_lon"], rows["fg_src"] = np.nan, np.nan, ""
    for tech in FIRST_GUESS[model]:
        fg = a[(a["tech"] == tech) & a["tau"].isin(LEADS)][["storm_id", "init", "tau", "lat", "lon"]]
        fg = fg.rename(columns={"tau": "lead", "lat": f"{tech}_lat", "lon": f"{tech}_lon"})
        rows = rows.merge(fg, on=["storm_id", "init", "lead"], how="left")
        use = rows["fg_lat"].isna() & rows[f"{tech}_lat"].notna()
        rows.loc[use, "fg_lat"] = rows.loc[use, f"{tech}_lat"]
        rows.loc[use, "fg_lon"] = rows.loc[use, f"{tech}_lon"]
        rows.loc[use, "fg_src"] = tech
        rows = rows.drop(columns=[f"{tech}_lat", f"{tech}_lon"])
    # Past the end of an aid track: continue from its last point with its last 12-h motion (at most 48 h of drift)
    rows = rows.sort_values(["storm_id", "init", "lead"]).reset_index(drop=True)
    g = rows.groupby(["storm_id", "init"])
    last_lat, last_lon = g["fg_lat"].ffill(), g["fg_lon"].ffill()
    last_lead = rows["lead"].where(rows["fg_lat"].notna()).groupby([rows["storm_id"], rows["init"]]).ffill()
    lat_12 = rows["fg_lat"].where(rows["fg_lat"].notna()).groupby([rows["storm_id"], rows["init"]]).shift(2)
    lon_12 = rows["fg_lon"].where(rows["fg_lat"].notna()).groupby([rows["storm_id"], rows["init"]]).shift(2)
    v_lat_fg = ((rows["fg_lat"] - lat_12) / 12).groupby([rows["storm_id"], rows["init"]]).ffill()
    v_lon_fg = ((rows["fg_lon"] - lon_12) / 12).groupby([rows["storm_id"], rows["init"]]).ffill()
    tail = rows["fg_lat"].isna() & last_lat.notna()
    dt = np.minimum(rows["lead"] - last_lead, 48)
    rows.loc[tail, "fg_lat"] = (last_lat + v_lat_fg.fillna(0) * dt)[tail]
    rows.loc[tail, "fg_lon"] = (last_lon + v_lon_fg.fillna(0) * dt)[tail]
    rows.loc[tail, "fg_src"] = "tail"
    # No aid track at all: extrapolate CARQ motion (capped at 72 h of drift)
    v_lat = (rows["lat0"] - rows["lat_m12"]) / 12
    v_lon = (rows["lon0"] - rows["lon_m12"]) / 12
    h = np.minimum(rows["lead"], 72)
    miss = rows["fg_lat"].isna()
    rows.loc[miss, "fg_lat"] = rows.loc[miss, "lat0"] + v_lat[miss].fillna(0) * h[miss]
    rows.loc[miss, "fg_lon"] = rows.loc[miss, "lon0"] + v_lon[miss].fillna(0) * h[miss]
    rows.loc[miss, "fg_src"] = "extrap"
    rows.loc[rows["lead"] == 0, ["fg_lat", "fg_lon"]] = rows.loc[rows["lead"] == 0, ["lat0", "lon0"]].to_numpy()
    return rows.sort_values(["init", "storm_id", "lead"]).reset_index(drop=True)


def runs(model: str, end: pd.Timestamp | None = None, start: pd.Timestamp = START) -> dict[pd.Timestamp, pd.DataFrame]:
    c = build(model, end, min(start, START))
    return {t: g for t, g in c.groupby("init")}


_adecks_2024_on = _adecks_from  # old name, used by scripts/report_figures.py
