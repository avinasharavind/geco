"""Download ECMWF's own per-member ENS tropical-cyclone tracks (BUFR, "tf" files) from the AWS open-data mirror.

Source: s3://ecmwf-forecasts (eu-central-1, anonymous HTTPS), never ECMWF's portal or Copernicus.
  YYYYMMDD/HHz/ifs/0p25/enfo/YYYYMMDDHH0000-{240h|360h|144h}-enfo-tf.bufr   (from ~2024-02)
  YYYYMMDD/HHz/0p4-beta/enfo/YYYYMMDDHH0000-{240h|144h}-enfo-tf.bufr       (before)
0.2-2 MB per run, all four cycles. Downloads go through the global throttle (tcic.io.throttle).

    scripts/fetch_ecmwf_tracks.py [START [END]]     (YYYY-MM-DD; default 2023-01-01 .. yesterday)
Resumable: data/raw/ecmwf_tf/YYYYMMDDHH.bufr (atomic writes) and manifest.csv (cycle, key or "missing", bytes). Cycles already in
the manifest are skipped; requests are paced and 5xx answers retried with backoff; a request that still fails is logged and retried on the next start.
"""
import csv
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

from tcic.config import RAW_DIR
from tcic.io.runner import atomic_write_bytes
from tcic.io.throttle import throttled_get

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
OUT = RAW_DIR / "ecmwf_tf"
OUT.mkdir(parents=True, exist_ok=True)
MAN, LOG = OUT / "manifest.csv", OUT / "fetch.log"
PACE = 0.5  # s between requests
start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2023, 1, 1)
end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(timezone.utc).date() - timedelta(days=1)


def note(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}\n")


def get(url: str) -> bytes:
    """throttled_get, paced (S3 answers 503 SlowDown to bursts of requests) and retried with backoff on 5xx."""
    for wait in (5, 15, 60, 180, 600):
        time.sleep(PACE)
        try:
            return throttled_get(url)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code < 500:
                raise
            note(f"{url.rsplit('/', 1)[-1]}: {e.response.status_code}, retrying in {wait} s")
            time.sleep(wait)
    return throttled_get(url)


def candidates(cyc: str) -> list[str]:
    d, hh = cyc[:8], cyc[8:]
    steps = ("240h", "360h", "144h") if hh in ("00", "12") else ("144h", "240h")
    return [f"{d}/{hh}z/{root}/enfo/{cyc}0000-{s}-enfo-tf.bufr" for root in ("ifs/0p25", "0p4-beta") for s in steps]


done = {r["cycle"] for r in csv.DictReader(MAN.open())} if MAN.exists() else set()
if not MAN.exists():
    MAN.write_text("cycle,key,bytes\n")
cycles = [f"{start + timedelta(days=i):%Y%m%d}{hh}" for i in range((end - start).days + 1) for hh in ("00", "06", "12", "18")]
todo = [c for c in cycles if c not in done]
note(f"start: {len(cycles)} cycles {start}..{end}, {len(todo)} to do")
n_got = n_miss = n_fail = nbytes = 0
t0 = time.time()
for i, cyc in enumerate(todo):
    row, failed = None, False
    for key in candidates(cyc):
        try:
            raw = get(f"{BASE}/{key}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 404):  # S3 answers 403 for absent keys without list rights
                continue
            failed = True
            note(f"{cyc} {key}: {e}")
            break
        except requests.RequestException as e:
            failed = True
            note(f"{cyc} {key}: {e}")
            break
        atomic_write_bytes(OUT / f"{cyc}.bufr", raw)
        row, nbytes, n_got = (cyc, key, len(raw)), nbytes + len(raw), n_got + 1
        break
    if failed:
        n_fail += 1
        continue
    if row is None:
        row, n_miss = (cyc, "missing", 0), n_miss + 1
    with MAN.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    if (i + 1) % 200 == 0:
        note(f"{i + 1}/{len(todo)}: {n_got} files, {n_miss} missing, {n_fail} failed, {nbytes / 1e6:.0f} MB, {time.time() - t0:.0f} s")
note(f"finished: {n_got} files, {n_miss} missing, {n_fail} failed (retried next start), {nbytes / 1e6:.0f} MB, {time.time() - t0:.0f} s")
