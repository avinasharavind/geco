"""Build the HCCA-centred ensembles (src/tcic/ensemble/geco.py): GECO for the NHC-GEFS and our own GEFS member sets, and IECO for our
IFS-ENS members (12Z cycles from the 00Z run; k fit on the 2024 out-of-fold season, the only one available).

k(tau) is fit on each set's out-of-fold training seasons with the HCCA centre. Adds column GECO (IECO) to
data/results/ensemble/test_{nhc_gefs,pilot_gefs} (test_pilot_ifs).parquet; cases without HCCA take the CCSR members (fallback),
flagged in GECO_src (IECO_src).
Writes docs/ensemble_geco.md.
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, ROOT, TEST_LABEL
from tcic.ensemble import calib, geco
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
t_start = time.time()
d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
h = geco.hcca()
rows = []
for name, label, col in (("nhc_gefs", "NHC-GEFS", "GECO"), ("pilot_gefs", "Ours GEFS", "GECO"), ("pilot_ifs", "Ours IFS-ENS", "IECO")):
    oof = pd.read_parquet(R / f"oof_{name}.parquet")
    test = pd.read_parquet(R / f"test_{name}.parquet").drop(columns=[col, f"{col}_src"], errors="ignore")
    crps, c0, c1 = geco.fit_k(geco.hcca_centre(oof[oof["season"].isin(calib.TRAIN)], h), sig)
    m = geco.build(geco.hcca_centre(test, h), sig, c0, c1, name=col)
    test = test.merge(m, on=KEY + ["member", "lead"], how="left")
    has = test.groupby(KEY)[col].transform(lambda x: x.notna().any())
    test[f"{col}_src"] = np.where(has, "HCCA", "CCSR fallback")
    test[col] = test[col].where(has, test["CCSR"])
    test.to_parquet(R / f"test_{name}.parquet", index=False)
    t = test[test["lead"].isin(LEADS) & test["obs_vmax"].notna()].dropna(subset=[col, "CCSR"])
    S = {v: sc.ensemble_case_scores(t, v) for v in ("RAW", "CCSR", col)}
    fb = test.drop_duplicates(KEY)[f"{col}_src"].eq("CCSR fallback").mean()
    for v, s in S.items():
        err = s["mean"] - s["obs"]
        row = {"members": label, "ensemble": v, "CRPS": s["crps"].mean(), "MAE of mean": s["abs_err_mean"].mean(), "bias": err.mean(),
               "coverage": s["inside"].mean(), "spread/error": s["std"].mean() / np.sqrt((err ** 2).mean())}
        if v == col:
            dd, lo, hi = sc.storm_bootstrap_diff(s, S["CCSR"], "crps")
            row["− CCSR"] = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            row["k(τ)"], row["CCSR fallback share"] = f"{c0:.2f} {c1:+.2f} τ/100", fb
        rows.append(row)
    print(f"{label} {col}: k = {c0:.2f} {c1:+.2f} τ/100, fallback {fb:.1%} of forecasts ({time.time() - t_start:.0f} s)", flush=True)
txt = (f"# GECO / IECO (GEFS / IFS-ENS members around HCCA), {TEST_LABEL}\n\nGECO (IECO) = GEFS (IFS-ENS) member shapes (smoothed, anchored, standardised deviations) "
       "around NHC's HCCA as issued, scaled by the predicted spread σ̂ and k(τ) fit on the out-of-fold training seasons; full spread. "
       "Forecasts without HCCA fall back to CCSR. GEFS: all cycles; IFS-ENS: 12Z cycles. Leads 12–120 h. Multi-season evaluation: docs/ensemble_multiseason.md.\n\n"
       + pd.DataFrame(rows).set_index(["members", "ensemble"]).to_markdown(floatfmt=".2f") + f"\n\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_geco.md").write_text(txt)
print(txt)
