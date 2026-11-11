"""Storm-specific spread (CCSP) and a calibrated RI probability; compares with CCSN (lead-only spread), member-count RI
probabilities and FNV3 on 2025–26. Adds column CCSP to data/results/ensemble/test_*.parquet; writes docs/ensemble_calibration.md.
Run after scripts/ens_spread.py (which uses the same 6-h smoothing).
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tcic.config import DATA_DIR, INTERIM_DIR, ROOT
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import calib
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
OUT = DATA_DIR / "results" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
SIG_DEV, SIG_CEN = 6.0, 12.0
t_start = time.time()
rep = ["# Storm-specific spread and calibrated RI probability (2025–26)\n"]

# ---------------------------------------------------------------- 1. spread model
d, feats = calib.table()
d = calib.spread_model(d, feats)
d.to_parquet(OUT / "spread_ri_table.parquet", index=False)
te, tr = d[d["season"].isin(calib.TEST)], d[d["season"].isin(calib.TRAIN)]
lead_clim = tr.groupby("lead")["abs_err"].mean()
rows = []
for name, sub in (("out-of-fold 2018–24", tr), ("2025–26", te)):
    base = sub["lead"].map(lead_clim)
    rows.append({"sample": name, "corr(σ̂, |error|)": sub["sigma_hat"].corr(sub["abs_err"]),
                 "corr(lead-only, |error|)": base.corr(sub["abs_err"]),
                 "corr within lead (σ̂)": sub.groupby("lead").apply(lambda g: g["sigma_hat"].corr(g["abs_err"])).mean(),
                 "corr within lead (M1.5 10–90% width)": sub.groupby("lead").apply(lambda g: g["tcca_qwidth"].corr(g["abs_err"])).mean()})
rep.append("\n## 1. Spread model: how well does the predicted error σ̂ track the actual error of the corrected consensus?\n\n"
           + pd.DataFrame(rows).set_index("sample").to_markdown(floatfmt=".2f") + "\n")
# binned: predicted vs actual mean |error| on 2025–26
te_b = te.assign(bin=pd.qcut(te["sigma_hat"], 5, labels=["lowest 20%", "20–40%", "40–60%", "60–80%", "highest 20%"]))
rep.append("\n2025 cases binned by predicted error (all leads):\n\n"
           + te_b.groupby("bin", observed=True).agg(cases=("abs_err", "size"), predicted=("sigma_hat", "mean"), actual=("abs_err", "mean")).to_markdown(floatfmt=".1f") + "\n")
print("spread model done", round(time.time() - t_start), "s", flush=True)

# ---------------------------------------------------------------- 2. ensembles with storm-specific spread
sigma = d.pivot_table(index=KEY, columns="lead", values="sigma_hat")
summ = []
fn = fnv3_data.load_test()


def fnv3_early(cases):
    runs = fn[["storm_id", "init"]].drop_duplicates()
    c = cases[KEY].drop_duplicates().merge(runs, on="storm_id")
    c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))]
    c = c.sort_values("init").groupby(KEY).tail(1)
    m = c.merge(fn, on=["storm_id", "init"])
    m["lead"] = m["lead"] - ((m["t0"] - m["init"]) / pd.Timedelta(hours=1)).astype(int)
    return m.rename(columns={"vmax": "FNV3"})


member_ri = {}
for name, label, hour in (("pilot_gefs", "Ours GEFS", 6), ("pilot_ifs", "Ours IFS-ENS", 12), ("nhc_gefs", "NHC-GEFS", None), ("nhc_ecmwf", "NHC-ECMWF", None)):
    oof, test = pd.read_parquet(R / f"oof_{name}.parquet"), pd.read_parquet(R / f"test_{name}.parquet")
    test = test.drop(columns=[c for c in ("CCSP",) if c in test])
    crps_oof, c0, c1 = calib.fit_k(oof[oof["season"].isin(calib.TRAIN)], sigma, SIG_DEV, SIG_CEN)
    dev, cen = smo.prepare(test, SIG_DEV, SIG_CEN)
    z = calib.standardised(dev)
    keep = z.index.droplevel("member").isin(sigma.index)
    test = test.merge(calib.members_p(z[keep], cen, sigma, c0, c1), on=KEY + ["member", "lead"], how="left")
    test.to_parquet(R / f"test_{name}.parquet", index=False)
    print(f"{label}: k(τ) = {c0:.2f} + {c1:.2f} τ/100 ({time.time() - t_start:.0f} s)", flush=True)
    sub = test[test["lead"].isin(LEADS) & test["CCSN"].notna() & test["CCSP"].notna()]
    if hour is not None:
        sub = sub[sub["t0"].dt.hour == hour]
    S = {"CCSN (lead-only spread)": sc.ensemble_case_scores(sub, "CCSN"), "CCSP (storm-specific spread)": sc.ensemble_case_scores(sub, "CCSP")}
    f = fnv3_early(sub)
    f = f[f["lead"].isin(LEADS)].merge(sub.drop_duplicates(CASE)[CASE + ["obs_vmax"]], on=CASE)
    S["FNV3 members"] = sc.ensemble_case_scores(f, "FNV3")
    keys = None
    for s in S.values():
        keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
    H = {k: s.merge(keys, on=CASE) for k, s in S.items()}
    for k, h in H.items():
        err = h["mean"] - h["obs"]
        summ.append({"ensemble": label, "version": k, "CRPS": h["crps"].mean(), "10–90% coverage": h["inside"].mean(),
                     "spread/error": h["std"].mean() / np.sqrt((err ** 2).mean()), "spread–error corr.": h["std"].corr(err.abs()),
                     "spread–error corr. within lead": h.assign(ae=err.abs()).groupby("lead").apply(lambda g: g["std"].corr(g["ae"])).mean()})
    row = {r_["version"]: r_ for r_ in summ if r_["ensemble"] == label}
    for b in ("CCSN (lead-only spread)", "FNV3 members"):
        dd, lo, hi = sc.storm_bootstrap_diff(H["CCSP (storm-specific spread)"], H[b], "crps")
        row["CCSP (storm-specific spread)"][f"CRPS − {b.split(' ')[0]} [95% CI]"] = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    # coverage by thirds of predicted error (same case split for every version): does the interval widen where it should?
    sig_k = keys.merge(d[CASE + ["sigma_hat"]], on=CASE)
    sig_k["third"] = pd.qcut(sig_k["sigma_hat"].rank(method="first"), 3, labels=["calm", "middle", "volatile"])
    for k, h in H.items():
        cov = h.merge(sig_k[CASE + ["third"]], on=CASE).groupby("third", observed=True)["inside"].mean()
        row[k]["coverage calm / middle / volatile third"] = " / ".join(f"{v:.2f}" for v in cov)
    if hour is not None:  # member-count RI probabilities for the RI comparison below
        member_ri[label] = test[test["t0"].dt.hour == hour]

rep.append("\n## 2. Ensembles with storm-specific spread (2025–26, leads 12–120 h, homogeneous within each ensemble)\n\n"
           "CCSP = corrected mean + k(τ)·σ̂·(standardised smoothed member deviations). Coverage by thirds of predicted error shows whether "
           "the interval holds its coverage in both calm and volatile situations (it should be ≈ 0.8 in every third).\n\n"
           + pd.DataFrame(summ).set_index(["ensemble", "version"]).to_markdown(floatfmt=".2f") + "\n")

# ---------------------------------------------------------------- 3. RI probability
r = calib.ri_model(d, feats)
r.to_parquet(OUT / "ri_probabilities.parquet", index=False)
clim = r[r["season"].isin(calib.TRAIN)]["y"].mean()


def member_p(t: pd.DataFrame, col: str) -> pd.DataFrame:
    x = t[KEY + ["member", "lead", col]].dropna()
    start = t.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0"]].assign(lead=0).rename(columns={"vmax0": col})
    x = pd.concat([x, start])
    prev = x.assign(lead=x["lead"] + 24).rename(columns={col: "prev"})
    x = x.merge(prev, on=KEY + ["member", "lead"])
    return (x[col] - x["prev"] >= 30).groupby([x[k] for k in CASE]).mean().rename("p").reset_index()


def fnv3_p(cases: pd.DataFrame) -> pd.DataFrame:
    m = fnv3_early(cases)
    prev = m.assign(lead=m["lead"] + 24)[KEY + ["member", "lead", "FNV3"]].rename(columns={"FNV3": "prev"})
    m = m.merge(prev, on=KEY + ["member", "lead"])
    return (m["FNV3"] - m["prev"] >= 30).groupby([m[k] for k in CASE]).mean().rename("p").reset_index()


ri_rows, rel = [], {}
r25 = r[r["season"].isin(calib.TEST)]
groups = {"all 2025–26 cycles": r25}
for label, t in member_ri.items():
    hour = t["t0"].dt.hour.iloc[0]
    groups[f"{label} cycles ({hour:02d}Z)"] = r25[r25["t0"].dt.hour == hour]
for gname, rr in groups.items():
    systems = {"Climatology": rr.assign(p=clim)[CASE + ["p"]], "RI model (calibrated)": rr[CASE + ["p_ri"]].rename(columns={"p_ri": "p"}),
               "RI model (uncalibrated)": rr[CASE + ["p_ri_raw"]].rename(columns={"p_ri_raw": "p"})}
    for label, t in member_ri.items():
        if gname.startswith(label):
            systems[f"{label} members (share with +30 kt / 24 h)"] = member_p(t, "CCSP")
    systems["FNV3 members (share with +30 kt / 24 h)"] = fnv3_p(rr)
    keys = None
    for s in systems.values():
        keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
    y = keys.merge(rr[CASE + ["y"]], on=CASE)["y"].to_numpy()
    bs_clim = None
    for sname, s in systems.items():
        p = keys.merge(s, on=CASE)["p"].to_numpy()
        bs = float(np.mean((p - y) ** 2))
        bs_clim = bs if sname == "Climatology" else bs_clim
        ri_rows.append({"sample": gname, "system": sname, "Brier": bs, "skill vs climatology": 1 - bs / bs_clim,
                        "AUC": roc_auc_score(y, p) if sname != "Climatology" else np.nan, "mean P": p.mean(), "observed rate": y.mean(),
                        "cases": len(y), "RI cases": int(y.sum())})
    if gname == "all 2025–26 cycles":
        rel["RI model (calibrated)"] = calib.reliability(keys.merge(systems["RI model (calibrated)"], on=CASE)["p"], pd.Series(y))
        rel["FNV3 members"] = calib.reliability(keys.merge(systems["FNV3 members (share with +30 kt / 24 h)"], on=CASE)["p"], pd.Series(y))
rep.append("\n## 3. Rapid-intensification probability (P of ≥ 30 kt rise in the 24 h ending at t0 + τ, τ = 24–120 h), 2025–26\n\n"
           "Skill vs climatology = 1 − Brier / Brier(climatology), where climatology is the 2018–24 RI frequency. AUC measures ranking only.\n\n"
           + pd.DataFrame(ri_rows).set_index(["sample", "system"]).to_markdown(floatfmt=".3f") + "\n")
for k, v in rel.items():
    rep.append(f"\nReliability, {k} (all 2025–26 cycles):\n\n" + v.to_markdown(floatfmt=".3f") + "\n")
# 24-h lead only (the standard 'RI in the next 24 h' product)
r24 = r25[r25["lead"] == 24]
bs = lambda p: float(np.mean((p - r24["y"]) ** 2))
rep.append(f"\nNext-24-h product only (τ = 24 h, {len(r24)} cases, {int(r24['y'].sum())} RI): Brier calibrated model {bs(r24['p_ri']):.3f}, "
           f"climatology {bs(np.full(len(r24), clim)):.3f}, AUC {roc_auc_score(r24['y'], r24['p_ri']):.2f}.\n")
rep.append(f"\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_calibration.md").write_text("".join(rep))
print("".join(rep))
