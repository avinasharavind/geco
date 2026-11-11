"""Project-wide paths, resource limits and case definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # small text inputs only (ATCF decks, IBTrACS); never gridded GRIB
INTERIM_DIR = DATA_DIR / "interim"  # parsed parquet
FEATURES_DIR = DATA_DIR / "features"
PATCHES_DIR = DATA_DIR / "patches"

# Shared machine limits (see plan.md "Resources and constraints").
NET_IFACE = "wlP9s9"  # fallback only; the throttle follows whichever interface has the default route (Wi-Fi or wired)
NET_RATE_MBPS = 17.0  # Wi-Fi: ~85% of the measured ~20 MB/s ceiling
NET_RATE_WIRED_MBPS = 25.0  # wired: ~85% of the measured ~28-33 MB/s ceiling (2026-10-18)
CPU_QUOTA_PERCENT = 800  # 40% of 20 cores
MEMORY_MAX = "48G"  # 40% of 121 GB

BASINS = ("NA", "EP")
ATCF_BASINS = ("al", "ep")
SEASONS_M15 = range(2018, 2027)
TEST_SEASON = 2025
PSEUDO_OPERATIONAL_SEASON = 2026
TEST_SEASONS = (2025, 2026)  # combined held-out test (2026: working best track until NHC finalises it)
TEST_LABEL = "2025–26"
# final corrected ensemble per member source: our IFS-ENS is centred on the gridded consensus (scripts/ens_gridcentre.py)
FINAL_ENSEMBLE = {"pilot_ifs": "IECO", "pilot_gefs": "GECO", "nhc_gefs": "GECO", "nhc_ecmwf": "CCSR80"}
LEADS_H = range(12, 169, 12)  # evaluation leads, as in the paper (12-168 h)
RI_THRESHOLD_KT = 30  # rapid intensification: >= 30 kt in 24 h
