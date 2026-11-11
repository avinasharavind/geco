"""CCSG (formerly CCSG80, then briefly IECO; IECO is now the HCCA-centred IFS-ENS ensemble, scripts/ens_geco.py): IFS-ENS ensemble centred on the gridded consensus (A+G: consensus correction + IFS-ENS gridded features, docs/m2_results.md)
instead of the track-only consensus (TCCA). Everything else as CCSR80: same member shapes, same predicted spread sigma_hat,
k(tau) refit on the 2024 out-of-fold cases with the new centre, full fitted spread (KF = 1; switched from 0.80 on 2026-10-29, CCSG80 kept for comparison).

The A+G predictions are leave-one-season-out over 2024-2026 (each season predicted by a model trained on the other two), because the
IFS-ENS archive starts in 2024-04. They exist only at verifiable leads; elsewhere the last available A+G - TCCA adjustment is carried
forward along lead so that member traces stay complete.
Adds columns CCSG, CCSG80 to data/results/ensemble/test_pilot_ifs.parquet and writes docs/ensemble_gridcentre.md.
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import calib
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
KF = 1.00
t_start = time.time()

ag = pd.read_parquet(DATA_DIR / "results" / "m2" / "cases_ifs.parquet").query("tech == 'A+G'")
ag = ag.rename(columns={"init": "t0", "tau": "lead", "fcst": "ag_mean"})[CASE + ["ag_mean"]]
d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
ones = sig * 0 + 1.0


def gridded_centre(t: pd.DataFrame) -> pd.DataFrame:
    """Replace tcca_mean by the A+G mean; carry the last A+G - TCCA adjustment forward where A+G is missing."""
    c = t.drop_duplicates(CASE)[CASE + ["tcca_mean"]].merge(ag, on=CASE, how="left").sort_values(CASE)
    c["adj"] = (c["ag_mean"] - c["tcca_mean"]).groupby([c["storm_id"], c["t0"]]).ffill()
    c["new"] = c["tcca_mean"] + c["adj"]
    out = t.drop(columns=["tcca_mean"]).merge(c[CASE + ["new"]].rename(columns={"new": "tcca_mean"}), on=CASE, how="left")
    has = out.groupby(KEY)["tcca_mean"].transform(lambda x: x.notna().any())  # forecasts with no A+G at all keep no centre
    return out[has]


oof = pd.read_parquet(R / "oof_pilot_ifs.parquet")
test = pd.read_parquet(R / "test_pilot_ifs.parquet").drop(columns=["CCSG", "CCSG100", "CCSG80"], errors="ignore")
oof_g = gridded_centre(oof[oof["season"].isin(calib.TRAIN)])
crps, c0, c1 = calib.fit_k_q(oof_g, ones, sig, sig, 6.0, 12.0)
print(f"k = {c0:.2f} + {c1:.2f} τ/100 (OOF CRPS {crps:.2f}, {time.time() - t_start:.0f} s)", flush=True)
tg = gridded_centre(test)
dev, cen = smo.prepare(tg, 6.0, 12.0)
z = calib.standardised_smooth(dev)
z = z[z.index.droplevel("member").isin(sig.index)]
m = calib.members_q(z, cen, ones * KF, sig, sig, c0, c1, name="CCSG")
m["CCSG80"] = calib.members_q(z, cen, ones * 0.8, sig, sig, c0, c1, name="x")["x"].to_numpy()  # previous 80% setting, for comparison
test = test.merge(m, on=KEY + ["member", "lead"], how="left")
test.to_parquet(R / "test_pilot_ifs.parquet", index=False)

# score: CCSR80 (track-only centre) vs IECO (gridded centre) vs FNV3 early, identical cases, 12Z cycles
t = test[(test["t0"].dt.hour == 12) & test["lead"].isin(LEADS)].dropna(subset=["CCSR80", "CCSG"])
fn = fnv3_data.load_test()
runs = fn[["storm_id", "init"]].drop_duplicates()
c = t[KEY].drop_duplicates().merge(runs, on="storm_id")
c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))].sort_values("init").groupby(KEY).tail(1)
f = c.merge(fn, on=["storm_id", "init"])
f["lead"] = f["lead"] - ((f["t0"] - f["init"]) / pd.Timedelta(hours=1)).astype(int)
f = f[f["lead"].isin(LEADS)].merge(t.drop_duplicates(CASE)[CASE + ["obs_vmax"]], on=CASE).rename(columns={"vmax": "FNV3"})
S = {"Track-only centre (CCSR80)": sc.ensemble_case_scores(t, "CCSR80"), "Gridded centre (CCSG)": sc.ensemble_case_scores(t, "CCSG"),
     "CCSG80 (80% spread)": sc.ensemble_case_scores(t, "CCSG80"),
     "FNV3 early": sc.ensemble_case_scores(f, "FNV3")}
keys = None
for s in S.values():
    keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
H = {k: s.merge(keys, on=CASE) for k, s in S.items()}
rows = []
for label, years in ((TEST_LABEL, list(TEST_SEASONS)), ("2025", [2025]), ("2026", [2026])):
    for k, h in H.items():
        h = h[h["t0"].dt.year.isin(years)]
        err = h["mean"] - h["obs"]
        row = {"sample": label, "ensemble": k, "CRPS": h["crps"].mean(), "MAE of mean": err.abs().mean(), "bias": err.mean(),
               "coverage": h["inside"].mean(), "spread/error": h["std"].mean() / np.sqrt((err ** 2).mean())}
        for ref in ("Track-only centre (CCSR80)", "FNV3 early"):
            if k != ref:
                dd, lo, hi = sc.storm_bootstrap_diff(h, H[ref][H[ref]["t0"].dt.year.isin(years)], "crps")
                row[f"CRPS − {ref.split(' (')[0]}"] = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        rows.append(row)
txt = ("# IFS-ENS ensemble centred on the gridded consensus (CCSG)\n\n"
       "CCSG = CCSR (full spread) with the centre taken from the consensus correction with IFS-ENS gridded features (A+G, leave-one-season-out "
       f"over 2024–26) instead of the track-only consensus. k = {c0:.2f} + {c1:.2f} τ/100 (refit on 2024). 12Z cycles, leads 12–120 h, "
       f"identical cases ({keys['storm_id'].nunique()} storms, {len(keys)} cases).\n\n"
       + pd.DataFrame(rows).set_index(["sample", "ensemble"]).to_markdown(floatfmt=".2f") + f"\n\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_gridcentre.md").write_text(txt)
print(txt)
