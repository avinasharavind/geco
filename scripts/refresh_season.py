"""Refresh the current season's public inputs and rebuild the derived tables (small downloads; safe to rerun).

  1. NHC a/b-decks from atcf/aid_public and atcf/btk, re-downloaded (they grow during the season), written atomically
  2. data/interim/{adecks,bdecks}.parquet for 2018..year
  3. operational SHIPS text for the season (new files only) and data/interim/ships.parquet, ships_ri.parquet
  4. FNV3 runs for the season (benchmark only) -> data/interim/fnv3_<year>.parquet
  5. data/features/consensus_features.parquet
Run: systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=16G -p CPUWeight=1 nice -n 19 .venv/bin/python scripts/refresh_season.py 2026
"""
import re
import sys
import time

import pandas as pd

from tcic.config import INTERIM_DIR
from tcic.data import atcf, fnv3, ships
from tcic.io.runner import atomic_write_bytes
from tcic.io.throttle import throttled_get

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
YEARS = range(2018, YEAR + 1)
t0 = time.time()

out = atcf.ATCF_DIR / str(YEAR)
out.mkdir(parents=True, exist_ok=True)
n = 0
for src in (f"{atcf.BASE}/aid_public/", f"{atcf.BASE}/btk/"):
    pat = re.compile(rf"^[ab]({'|'.join(atcf.ATCF_BASINS)})\d{{2}}{YEAR}\.dat(\.gz)?$")
    for name in atcf._listing(src):
        if pat.match(name):
            atomic_write_bytes(out / name, throttled_get(src + name))
            n += 1
print(f"decks: {n} files refreshed ({time.time() - t0:.0f} s)", flush=True)
a, b = atcf.build(YEARS)
print(f"a-decks to {a['init'].max()}, b-decks to {b['time'].max()} ({time.time() - t0:.0f} s)", flush=True)

ships.download_season(YEAR)
frames = [ships.parse(p) for y in YEARS for p in sorted((ships.SHIPS_DIR / str(y)).glob("*_ships.txt")) if ships._NAME.match(p.name)]
s = pd.concat(frames, ignore_index=True).drop_duplicates(["storm_id", "init", "tau"])
s.to_parquet(INTERIM_DIR / "ships.parquet", index=False)
ships.build_ri(YEARS)
print(f"SHIPS: {s.drop_duplicates(['storm_id', 'init']).groupby(s.drop_duplicates(['storm_id', 'init'])['init'].dt.year).size().to_dict()} ({time.time() - t0:.0f} s)", flush=True)

f = fnv3.build(YEAR)
print(f"FNV3 {YEAR}: {f['init'].nunique()} runs, {f['storm_id'].nunique()} storms ({time.time() - t0:.0f} s)", flush=True)

from tcic.consensus import features  # noqa: E402 - after the inputs it reads are rebuilt

F = features.build()
v = F[F["season"] == YEAR]
print(f"consensus features: {len(F):,} rows; {YEAR}: {len(v):,} rows, {int(v['verifiable'].sum()):,} verifiable, SHIPS {v['has_ships'].mean():.2f} "
      f"({time.time() - t0:.0f} s)", flush=True)
