"""CCSR: storm-specific spread (CCSP) with the predicted spread and the per-case normalisation smoothed along lead time
(σ = 18 h), symmetric. Optional sharpness factor. Compared with CCSN / CCSP on 2025–26 (our GEFS and IFS-ENS): skill, RI-run
behaviour, member step statistics and the Erin / Melissa cases. Adds CCSR and CCSR85 columns; writes docs/ensemble_smoothspread.md.
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, ROOT
from tcic.ensemble import calib
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
t_start = time.time()
d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
ones = sig * 0 + 1.0
rows, rough, cases = [], [], []
for name, label, hour in (("pilot_gefs", "Ours GEFS", 6), ("pilot_ifs", "Ours IFS-ENS", 12), ("nhc_gefs", "NHC-GEFS", None), ("nhc_ecmwf", "NHC-ECMWF", None)):
    oof, test = pd.read_parquet(R / f"oof_{name}.parquet"), pd.read_parquet(R / f"test_{name}.parquet")
    test = test.drop(columns=[c for c in ("CCSR", "CCSR85", "CCSR80", "CCSR70") if c in test])
    crps, c0, c1 = calib.fit_k_q(oof[oof["season"].isin(calib.TRAIN)], ones, sig, sig, 6.0, 12.0)
    dev, cen = smo.prepare(test, 6.0, 12.0)
    z = calib.standardised_smooth(dev)
    keep = z.index.droplevel("member").isin(sig.index)
    for kf, col in ((1.0, "CCSR"), (0.85, "CCSR85"), (0.80, "CCSR80"), (0.70, "CCSR70")):  # CCSR80 is the final default ensemble
        test = test.merge(calib.members_q(z[keep], cen, ones * kf, sig, sig, c0, c1, name=col), on=KEY + ["member", "lead"], how="left")
    test.to_parquet(R / f"test_{name}.parquet", index=False)
    tt = test[test["t0"].dt.hour == hour] if hour is not None else test if hour is not None else test
    sub = tt[tt["lead"].isin(LEADS) & tt[["CCSN", "CCSP", "CCSR"]].notna().all(axis=1)]
    ri_runs = sub[sub["lead"].between(12, 120) & (sub["obs_dv24"] >= 30)][KEY].drop_duplicates().assign(ri_run=True)
    S = {v: sc.ensemble_case_scores(sub, v) for v in ("CCSN", "CCSP", "CCSR", "CCSR85", "CCSR80", "CCSR70")}
    for v, s in S.items():
        h = s.merge(ri_runs, on=KEY, how="left").fillna({"ri_run": False})
        err = h["mean"] - h["obs"]
        row = {"ensemble": label, "version": v, "CRPS": h["crps"].mean(), "coverage": h["inside"].mean(), "spread/error": h["std"].mean() / np.sqrt((err ** 2).mean())}
        for flag, g in h.groupby("ri_run"):
            tag = "RI runs" if flag else "other runs"
            e = g["mean"] - g["obs"]
            row[f"{tag}: spread / RMSE (kt)"] = f"{g['std'].mean():.1f} / {np.sqrt((e ** 2).mean()):.1f}"
            row[f"{tag}: coverage"] = g["inside"].mean()
        if v != "CCSP":
            dd, lo, hi = sc.storm_bootstrap_diff(s, S["CCSP"], "crps")
            row["CRPS − CCSP [95% CI]"] = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        rows.append(row)
        base = tt.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0"]]
        tr = pd.concat([tt[KEY + ["member", "lead", v]], base.assign(lead=0).rename(columns={"vmax0": v})[KEY + ["member", "lead", v]]])
        r = smo.roughness(tr, v)
        rough.append({"ensemble": label, "version": v, "mean |Δ12| (kt)": r["mean |Δ|"], "zig-zag rate": r["flip rate"], "share |Δ12| > 30 kt": r["share |Δ| > 30 kt"]})
    for sid, day in ((("AL052025", "2025-08-14"), ("AL132025", "2025-10-25")) if hour is not None else ()):
        t0 = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        g = tt[(tt["storm_id"] == sid) & (tt["t0"] == t0)]
        obs = g.drop_duplicates("lead").set_index("lead")["obs_vmax"]
        for v in ("CCSN", "CCSP", "CCSR", "CCSR85", "CCSR80", "CCSR70"):
            qq = g.groupby("lead")[v].quantile([0.1, 0.9]).unstack()
            cases.append({"case": f"{sid} {t0:%d %b %HZ} ({label})", "version": v,
                          **{f"{L} h 10–90% (BT)": f"{qq.loc[L, 0.1]:.0f}–{qq.loc[L, 0.9]:.0f} ({obs[L]:.0f})" for L in (24, 48, 72) if L in qq.index}})
    print(f"{label}: k = {c0:.2f} + {c1:.2f} τ/100 ({time.time() - t_start:.0f} s)", flush=True)
txt = ("# Lead-smoothed storm-specific spread (CCSR), 2025–26\n\nCCSR = CCSP with σ̂ and the per-case normalisation smoothed along lead (σ = 18 h); "
       "CCSR80 = 80% of its spread (the final default ensemble); CCSR85 / CCSR70 = 85% / 70%. Best track 12-h steps: 8.7 kt mean, 0.043 zig-zag, 1.8% > 30 kt.\n\n"
       + pd.DataFrame(rows).set_index(["ensemble", "version"]).to_markdown(floatfmt=".2f") + "\n\n"
       + pd.DataFrame(rough).set_index(["ensemble", "version"]).to_markdown(floatfmt=".3f") + "\n\n"
       + pd.DataFrame(cases).set_index(["case", "version"]).to_markdown() + "\n")
(ROOT / "docs" / "ensemble_smoothspread.md").write_text(txt)
print(txt)
