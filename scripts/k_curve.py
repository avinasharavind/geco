"""Out-of-fold CRPS, coverage and spread/error of GECO and IECO as a function of a constant spread factor k (2018-24, leads 12-120 h).
Shows why k = 1.2: the CRPS minimum, and how flat it is. Writes data/results/ensemble/k_curve.csv (used by scripts/report_figures_ens.py).
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from tcic.config import DATA_DIR, FEATURES_DIR
from tcic.ensemble import calib, geco
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo
KEY = smo.KEY
d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
ones = sig * 0 + 1
H = geco.hcca()
TCCA = pd.concat([pd.read_parquet(DATA_DIR / "results" / "m15" / f) for f in ("oof_vmax_A.parquet", "test_vmax_A.parquet")])[["storm_id","t0","lead","mean"]].rename(columns={"mean":"tcca_mean"})
sets = {"GECO": pd.read_parquet(DATA_DIR / "results/ensemble/oof_nhc_gefs.parquet")}
m = pd.read_parquet(FEATURES_DIR / "members_tf_ecmwf.parquet").merge(TCCA, on=["storm_id","t0","lead"])
sets["IECO"] = m
rows = []
for name, t in sets.items():
    t = geco.hcca_centre(t[t["season"].isin(calib.TRAIN)], H)
    dev, cen = smo.prepare(t, 6.0, 0.0)
    keys = dev.index.droplevel("member").unique(); keys = keys[keys.isin(sig.index)]
    dev = dev[dev.index.droplevel("member").isin(keys)]
    z = calib.standardised_smooth(dev)
    obs = t.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"])["obs_vmax"]
    for k in [0.6, 0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.6, 2.0]:
        mm = calib.members_q(z, cen, ones, sig, sig, k, 0.0, name="X").join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax", "X"])
        mm = mm[mm["lead"].isin(range(12, 121, 12))]
        s = sc.ensemble_case_scores(mm, "X")
        err = s["mean"] - s["obs"]
        rows.append({"product": name, "k": k, "CRPS": s["crps"].mean(), "coverage": s["inside"].mean(), "spread/error": s["std"].mean() / np.sqrt((err**2).mean()), "cases": len(s)})
T = pd.DataFrame(rows)
T.to_csv(DATA_DIR / "results" / "ensemble" / "k_curve.csv", index=False)
print(T.round(3).to_string(index=False))
# sigma_hat vs actual: ratio of RMS error to predicted MAE (Gaussian: sqrt(pi/2) = 1.253)
tr = d[d["season"].isin(calib.TRAIN)]
print("mean |err| / mean sigma_hat:", round(tr["abs_err"].mean() / tr["sigma_hat"].mean(), 3), " RMS err / mean |err|:", round(np.sqrt((tr["abs_err"]**2).mean()) / tr["abs_err"].mean(), 3))
