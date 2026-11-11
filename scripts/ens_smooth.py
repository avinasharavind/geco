"""Build the smoothed consensus-centred ensembles (CCS), score them on 2025–26 against CCC and FNV3, and measure step-to-step
physical consistency (12-h and 6-h intensity changes, direction flips) for the best track, raw physical members, FNV3, CCC
and CCS. Writes docs/ensemble_smooth.md and adds a CCS column to data/results/ensemble/test_*.parquet.
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, INTERIM_DIR, ROOT
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo
from tcic.pilot.extract import PILOT_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
R = DATA_DIR / "results" / "ensemble"
KEY, CASE = smo.KEY, sc.CASE
LEADS = list(range(12, 121, 12))
SIG_DEV, SIG_CEN = 18.0, 12.0
t_start = time.time()
report = ["# Smoothed consensus-centred ensembles (CCS): skill and physical consistency, 2025–26\n",
          f"CCS = M1.5 corrected mean (CARQ-anchored, Gaussian-smoothed σ={SIG_CEN:.0f} h) + λ(τ)·(member deviation anchored at t0, smoothed σ={SIG_DEV:.0f} h); "
          "λ(τ) = exp(a + bτ/100) fit on out-of-fold CRPS. CCC = previous version (raw deviations, independent λ per lead).\n"]

# ---------------------------------------------------------------- sensitivity to the smoothing width (pilot IFS, out-of-fold)
sens = []
oof_ifs = pd.read_parquet(R / "oof_pilot_ifs.parquet")
for sd in (0.0, 12.0, 18.0, 24.0, 36.0):
    crps, a, b = smo.fit(oof_ifs, sd, SIG_CEN)
    dev, cen = smo.prepare(oof_ifs, sd, SIG_CEN)
    tr = smo.members(dev, cen, a, b)
    r = smo.roughness(pd.concat([tr, cen[[0]].stack().rename("CCS").reset_index().merge(dev.reset_index()[KEY + ["member"]], on=KEY).rename(columns={"level_2": "lead"})]), "CCS")
    sens.append({"σ deviations (h)": sd, "OOF CRPS": crps, "λ(12 h)": np.exp(a + b * 0.12), "λ(120 h)": np.exp(a + b * 1.2), **{k: r[k] for k in ("mean |Δ|", "p99 |Δ|", "flip rate")}})
report.append("\n## Choice of smoothing width (our IFS-ENS, out-of-fold 2024/2026)\n\n" + pd.DataFrame(sens).to_markdown(index=False, floatfmt=".3f") + "\n")
print("sensitivity done", round(time.time() - t_start), "s", flush=True)

# ---------------------------------------------------------------- build CCS for every source and score
scores, rough = {}, {}
for name in ("pilot_ifs", "pilot_gefs", "nhc_gefs", "nhc_ecmwf"):
    oof, test = pd.read_parquet(R / f"oof_{name}.parquet"), pd.read_parquet(R / f"test_{name}.parquet")
    crps, a, b = smo.fit(oof, SIG_DEV, SIG_CEN)
    dev, cen = smo.prepare(test, SIG_DEV, SIG_CEN)
    ccs = smo.members(dev, cen, a, b)
    test = test.drop(columns=[c for c in ("CCS",) if c in test]).merge(ccs, on=KEY + ["member", "lead"], how="left")
    test.to_parquet(R / f"test_{name}.parquet", index=False)
    label = {"pilot_ifs": "Ours IFS-ENS", "pilot_gefs": "Ours GEFS", "nhc_gefs": "NHC-GEFS", "nhc_ecmwf": "NHC-ECMWF"}[name]
    print(f"{label}: a={a:.2f} b={b:.2f}  λ(12h)={np.exp(a + 0.12 * b):.2f} λ(120h)={np.exp(a + 1.2 * b):.2f}  ({time.time() - t_start:.0f} s)", flush=True)
    for col in ("CCC", "CCS"):
        s = sc.ensemble_case_scores(test[test[col].notna() & test["lead"].isin(LEADS)], col)
        scores[f"{label} {col}"] = s
    # roughness at 12 h, with t0 (CARQ) as the first point for corrected members and v_t0 for raw
    base = test.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0", "v_t0"]]
    for col, start in (("RAW", "v_t0"), ("CCC", "vmax0"), ("CCS", "vmax0")):
        tr = pd.concat([test[KEY + ["member", "lead", col]], base.assign(lead=0).rename(columns={start: col})[KEY + ["member", "lead", col]]])
        rough[f"{label} {col}"] = smo.roughness(tr, col)

# ---------------------------------------------------------------- FNV3 and best track roughness (same storms/years)
fn = fnv3_data.load_test().rename(columns={"init": "t0"})
fn = fn[fn["storm_id"].str[2:4].astype(int) < 50]
fn = fn[fn["lead"] <= 168]
rough["FNV3 raw"] = smo.roughness(fn, "vmax")
bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet")
bt = bt[bt["time"].dt.year == 2025].assign(storm_id=lambda x: x["storm_id"].astype(str))
starts = fn[["storm_id", "t0"]].drop_duplicates()
btr = starts.merge(bt[["storm_id", "time", "vmax"]], on="storm_id")
btr = btr[(btr["time"] >= btr["t0"]) & (btr["time"] <= btr["t0"] + pd.Timedelta(hours=168))]
btr["lead"] = ((btr["time"] - btr["t0"]) / pd.Timedelta(hours=1)).astype(int)
btr["member"] = 0
rough["Best track"] = smo.roughness(btr, "vmax")
# 6-hourly: FNV3 vs our raw physical members (pilot tracker, 6-hourly) vs best track
r6 = {"Best track": smo.roughness(btr, "vmax", step_h=6, flip_kt=7), "FNV3 raw": smo.roughness(fn, "vmax", step_h=6, flip_kt=7)}
for mdl in ("ifs", "gefs"):
    files = sorted((PILOT_DIR / mdl).glob("2025*/done.json"))
    t = pd.concat([pd.read_parquet(f.parent / "tracks.parquet", columns=["storm_id", "init", "member", "lead", "vmax"]) for f in files]).rename(columns={"init": "t0"})
    r6[f"Raw {'IFS-ENS' if mdl == 'ifs' else 'GEFS'} members (our tracker)"] = smo.roughness(t, "vmax", step_h=6, flip_kt=7)
report.append("\n## Physical consistency: 12-h intensity changes along each trace (2025–26)\n\n"
              "A flip is two consecutive 12-h changes of opposite sign, each ≥ 10 kt (a zig-zag). Corrected members start at the CARQ intensity.\n\n"
              + pd.DataFrame(rough).T.to_markdown(floatfmt=".3f") + "\n")
report.append("\n## 6-h steps: FNV3 vs raw physical members vs best track (2025–26)\n\nFlip threshold 7 kt per 6 h.\n\n"
              + pd.DataFrame(r6).T.to_markdown(floatfmt=".3f") + "\n")

# ---------------------------------------------------------------- skill: CCS vs CCC (homogeneous within each source)
rows = []
for label in ("Ours IFS-ENS", "Ours GEFS", "NHC-GEFS", "NHC-ECMWF"):
    a_, b_ = scores[f"{label} CCC"], scores[f"{label} CCS"]
    k = a_[CASE].merge(b_[CASE], on=CASE)
    for col, s in (("CCC", a_), ("CCS", b_)):
        h = s.merge(k, on=CASE)
        rows.append({"ensemble": f"{label} {col}", "CRPS": h["crps"].mean(), "quantile score": h["qs"].mean(), "MAE of mean": h["abs_err_mean"].mean(),
                     "10–90% coverage": h["inside"].mean(), "spread/RMSE": h["std"].mean() / np.sqrt(((h["mean"] - h["obs"]) ** 2).mean())})
    d, lo, hi = sc.storm_bootstrap_diff(b_, a_, "crps")
    rows[-1]["CRPS CCS − CCC [95% CI]"] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
report.append("\n## Skill: smoothed (CCS) vs previous (CCC), 2025–26, leads 12–120 h\n\n" + pd.DataFrame(rows).set_index("ensemble").to_markdown(floatfmt=".2f") + "\n")
report.append(f"\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_smooth.md").write_text("".join(report))
print("".join(report))
