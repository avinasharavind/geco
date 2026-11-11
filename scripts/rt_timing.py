"""Real-time arrival log for NHC's public a-decks: when does each aid for each cycle first appear in atcf/aid_public?

Every POLL seconds: read the aid_public listing, fetch the a-decks (al/ep/cp) whose size changed, and log every (storm, cycle, aid)
seen for the first time, with the UTC time and the delay after the cycle's synoptic time. Small text files only (tens of kB to a few MB).
Resumable: the set of seen keys is rebuilt from the log on start (seen.txt; entries already present at the very first poll are
not logged; after a restart, entries that appeared while it was down are logged with a late time). Key aids for GECO: HCCA, AP01-30 (GEFS members), CARQ, the early aids.

    scripts/rt_timing.py [POLL_SECONDS]        (default 300; run detached, e.g. under systemd-run with the usual caps)
Writes data/rt_timing/arrivals.csv (storm_id, init, tech, first_seen_utc, delay_h) and data/rt_timing/poll.log.
"""
import csv
import gzip
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tcic.config import DATA_DIR
from tcic.data import atcf
from tcic.io.throttle import throttled_get

POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = DATA_DIR / "rt_timing"
OUT.mkdir(parents=True, exist_ok=True)
LOG, ARR = OUT / "poll.log", OUT / "arrivals.csv"
SRC = f"{atcf.BASE}/aid_public/"
PAT = re.compile(r"^a(al|ep|cp)(\d{2})(\d{4})\.dat(\.gz)?$")

SEEN = OUT / "seen.txt"  # every key ever seen (logged or not), one "storm,cycle,aid" per line
seed = not SEEN.exists()  # first ever poll: record what is already there without logging it (its delays would be meaningless)
seen = set(tuple(l.split(",")) for l in SEEN.read_text().splitlines()) if SEEN.exists() else set()
if not ARR.exists():
    ARR.write_text("storm_id,init,tech,first_seen_utc,delay_h\n")
sizes: dict[str, int] = {}


def note(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}\n")


while True:
    try:
        now = datetime.now(timezone.utc)
        names = [n for n in atcf._listing(SRC) if PAT.match(n)]
        new = []
        for name in names:
            raw = throttled_get(SRC + name)
            if sizes.get(name) == len(raw):
                continue
            sizes[name] = len(raw)
            text = (gzip.decompress(raw) if name.endswith(".gz") else raw).decode(errors="ignore")
            b, num, yr = PAT.match(name).groups()[:3]
            sid = f"{b.upper()}{num}{yr}"
            for line in text.splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) < 5:
                    continue
                key = (sid, p[2], p[4])
                if key not in seen:
                    seen.add(key)
                    init = datetime.strptime(p[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
                    new.append((*key, f"{now:%Y-%m-%d %H:%M}", f"{(now - init).total_seconds() / 3600:.2f}"))
        with SEEN.open("a") as f:
            f.writelines(f"{a},{b},{c}\n" for a, b, c, *_ in new)
        if seed:
            new, seed = [], False
        if new:
            with ARR.open("a", newline="") as f:
                csv.writer(f).writerows(new)
        note(f"{len(names)} decks, {len(new)} new (storm, cycle, aid) entries")
    except Exception as e:  # noqa: BLE001 - a failed poll is logged and retried next time
        note(f"poll failed: {e}")
    time.sleep(POLL)
