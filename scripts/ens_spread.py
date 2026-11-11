"""Spread-normalised consensus-centred ensembles (CCSN): fit, build, and compare with CCS and FNV3 on 2025–26.

Adds columns CCSN (k = 1, CRPS-optimal) and CCSN85 (k = 0.85, sharper) to data/results/ensemble/test_*.parquet and writes
docs/ensemble_spread.md (spread vs error by lead, calibration, spread-error correlation, CRPS, smoothness).
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, INTERIM_DIR, ROOT
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
KEY, CASE = smo.KEY, sc.CASE
LEADS = list(range(12, 121, 12))
SIG_DEV, SIG_CEN = 6.0, 12.0  # 6 h chosen in docs/ensemble_sigma_sweep.md (best-track-like step statistics, same skill)
t_start = time.time()
fn = fnv3_data.load_test()
SOURCES = {"pilot_gefs": ("Ours GEFS", 6), "pilot_ifs": ("Ours IFS-ENS", 12), "nhc_gefs": ("NHC-GEFS", None), "nhc_ecmwf": ("NHC-ECMWF", None)}
report = ["# Spread-normalised consensus-centred ensembles (CCSN), 2025–26\n",
          "CCSN: smoothed, t0-anchored member deviations, rescaled by the typical raw spread of the previous 120 days of cases "
          "(past only), keeping a fitted share γ of case-to-case spread variation, then scaled by a calibrated spread s(τ). "
          "CCSN85 is the same at 85% of the calibrated spread (sharper). CCS = previous smoothed version. FNV3 = raw members, early timing "
          "(00Z run for 06Z/12Z cycles; newest run ≥ 6 h old otherwise).\n"]
fit_rows, summary_rows, lead_tables = [], [], []


def fnv3_early(cases: pd.DataFrame) -> pd.DataFrame:
    runs = fn[["storm_id", "init"]].drop_duplicates()
    c = cases[KEY].drop_duplicates().merge(runs, on="storm_id")
    c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))]
    c = c.sort_values("init").groupby(KEY).tail(1)
    m = c.merge(fn, on=["storm_id", "init"])
    m["lead"] = m["lead"] - ((m["t0"] - m["init"]) / pd.Timedelta(hours=1)).astype(int)
    m = m[m["lead"].isin(LEADS)].merge(cases.drop_duplicates(CASE)[CASE + ["obs_vmax"]], on=CASE)
    return m.rename(columns={"vmax": "FNV3"})


for name, (label, hour) in SOURCES.items():
    oof, test = pd.read_parquet(R / f"oof_{name}.parquet"), pd.read_parquet(R / f"test_{name}.parquet")
    test = test.drop(columns=[c for c in ("CCSN", "CCSN85") if c in test])
    history = pd.concat([oof, test], ignore_index=True)
    history["tcca_mean"] = history["tcca_mean"].fillna(history["ens_mean"])  # centre unused for the spread history
    (crps, gamma, A, pw), (sd_all, ref_all) = smo.fit_n(oof, history, SIG_DEV, SIG_CEN)
    dev, cen = smo.prepare(test, SIG_DEV, SIG_CEN)
    z = smo.normalised(dev, sd_all, ref_all, gamma)
    for k, col in ((1.0, "CCSN"), (0.85, "CCSN85")):
        test = test.merge(smo.members_n(z, cen, A, pw, k, col), on=KEY + ["member", "lead"], how="left")
    test.to_parquet(R / f"test_{name}.parquet", index=False)
    curve = smo.spread_curve(A, pw)
    fit_rows.append({"ensemble": label, "γ (case-specific share)": gamma, "s(12 h) kt": curve[1], "s(72 h) kt": curve[6], "s(120 h) kt": curve[10],
                     "power p": pw, "OOF CRPS": crps})
    print(f"{label}: gamma={gamma} A={A:.1f} p={pw:.2f} s12={curve[1]:.1f} s120={curve[10]:.1f} ({time.time() - t_start:.0f} s)", flush=True)

    sub = test[test["lead"].isin(LEADS) & test["CCS"].notna() & test["CCSN"].notna()]
    if hour is not None:
        sub = sub[sub["t0"].dt.hour == hour]
    S = {c: sc.ensemble_case_scores(sub, c) for c in ("RAW", "CCS", "CCSN", "CCSN85")}
    S["FNV3"] = sc.ensemble_case_scores(fnv3_early(sub), "FNV3")
    keys = None
    for s in S.values():
        keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
    H = {c: s.merge(keys, on=CASE) for c, s in S.items()}
    for c, h in H.items():
        err = h["mean"] - h["obs"]
        summary_rows.append({"ensemble": label, "version": c, "CRPS": h["crps"].mean(), "MAE of mean": err.abs().mean(),
                             "10–90% coverage": h["inside"].mean(), "spread/RMSE": h["std"].mean() / np.sqrt((err ** 2).mean()),
                             "spread–error corr.": h["std"].corr(err.abs()), "cases": len(h), "storms": h["storm_id"].nunique()})
    row_of = {r["version"]: r for r in summary_rows if r["ensemble"] == label}
    for c in ("CCS", "CCSN", "CCSN85"):
        d, lo, hi = sc.storm_bootstrap_diff(H[c], H["FNV3"], "crps")
        row_of[c]["CRPS vs FNV3 [95% CI]"] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    for c in ("CCSN", "CCSN85"):
        d, lo, hi = sc.storm_bootstrap_diff(H[c], H["CCS"], "crps")
        row_of[c]["CRPS vs CCS [95% CI]"] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    tab = {}
    for c, h in H.items():
        g = h.groupby("lead")
        tab[(c, "spread")] = g["std"].mean()
        tab[(c, "RMSE")] = g.apply(lambda x: np.sqrt(((x["mean"] - x["obs"]) ** 2).mean()))
    lead_tables.append(f"\n**{label}**" + (f" (t0 = {hour:02d}Z)" if hour is not None else " (all cycles)") + "\n\n" + pd.DataFrame(tab).T.round(1).to_markdown() + "\n")
    # smoothness of the new traces
    base = test.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0"]]
    for c in ("CCS", "CCSN"):
        tr = pd.concat([test[KEY + ["member", "lead", c]], base.assign(lead=0).rename(columns={"vmax0": c})[KEY + ["member", "lead", c]]])
        r = smo.roughness(tr, c)
        summary_rows[[i for i, rr in enumerate(summary_rows) if rr["ensemble"] == label and rr["version"] == c][0]]["zig-zag rate"] = r["flip rate"]

report.append("\n## Fitted settings (out-of-fold training seasons)\n\n" + pd.DataFrame(fit_rows).set_index("ensemble").to_markdown(floatfmt=".2f") + "\n")
report.append("\n## 2025–26 summary (leads 12–120 h, homogeneous within each ensemble)\n\n"
              + pd.DataFrame(summary_rows).set_index(["ensemble", "version"]).to_markdown(floatfmt=".2f") + "\n")
report.append("\n## Spread vs error of the mean by lead (kt), 2025–26\n" + "".join(lead_tables))
report.append(f"\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_spread.md").write_text("".join(report))
print("".join(report[:3]))
