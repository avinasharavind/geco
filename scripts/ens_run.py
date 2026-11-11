"""Corrected intensity ensembles: turn raw GEFS / ECMWF members into a calibrated member ensemble, from two sources.

Methods (per source and model):
  RAW   raw member intensities
  BIAS  RAW + per-lead mean bias (from training seasons)                          -- "mean shift"
  BIASC BIAS with spread scaled about the mean, per lead (lambda fit on out-of-fold)
  MEM   member-wise LightGBM correction: v_i + f(member trajectory, rank, ensemble context, storm state[, structure, env])
  MEMC  MEM with spread scaling
  HYBC  member-wise model's mean + raw member deviations, spread-scaled
  CCC   consensus-centred: M1.5 TCCA mean + raw member deviations, spread-scaled
Benchmark: M1.5 TCCA quantiles after conformal calibration (q10/q50/q90). Test seasons 2025 + 2026 (combined); training = earlier seasons
(NHC source: 2018-2024, out-of-fold by season for lambda) or 2024 (pilot IFS-ENS) and 2020-2024 (pilot GEFS).
Run: systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=32G -p CPUWeight=1 nice -n 19 .venv/bin/python scripts/ens_run.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, FEATURES_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.consensus import calibrate
from tcic.ensemble import score as sc
from tcic.ensemble.members import STATE

OUT = DATA_DIR / "results" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)
TEST = list(TEST_SEASONS)  # combined held-out test, 2025 + 2026
LEADS_EVAL = list(range(12, 121, 12))
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # single-member cases give undefined spreads
CASE = sc.CASE
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
ROUNDS = 400
t_start = time.time()
# M1.5 TCCA corrected mean (variant A) at every cycle: out-of-fold for 2018-2024, held-out test for 2025-26
TCCA_MEAN = pd.concat([pd.read_parquet(DATA_DIR / "results" / "m15" / f) for f in ("oof_vmax_A.parquet", "test_vmax_A.parquet")])
TCCA_MEAN = TCCA_MEAN[["storm_id", "t0", "lead", "mean"]].rename(columns={"mean": "tcca_mean"})


def features(d: pd.DataFrame) -> list[str]:
    base = ["v", "v_shift", "off0", "v_t0", "dv12_own", "dv24_own", "z", "ens_rank", "ens_mean", "ens_std", "ens_n", "ens_ri_frac", "lead", "age"] + STATE
    extra = [c for c in d.columns if c in ("p", "p_t0", "rmw_km", "depth", "gust") or c.startswith(("cr_", "e_"))]
    return [c for c in base + extra if c in d.columns]


def fit(train: pd.DataFrame, feats):
    w = np.where(train["ri"] == 1, 2.0, 1.0) / train["ens_n"]  # each case counts equally regardless of member count
    return lgb.train(PARAMS, lgb.Dataset(train[feats], train["obs_vmax"] - train["v"], weight=w), ROUNDS)


def add_methods(d: pd.DataFrame, train: pd.DataFrame, model, feats) -> pd.DataFrame:
    d = d.copy()
    d["RAW"] = d["v"]
    bias = (train.drop_duplicates(CASE).assign(e=lambda x: x["obs_vmax"] - x["ens_mean"]).groupby("lead")["e"].mean())
    d["BIAS"] = d["v"] + d["lead"].map(bias).fillna(bias.iloc[-1])
    d["MEM"] = np.clip(d["v"] + model.predict(d[feats]), 10, 200)
    # hybrid: corrected mean from the member-wise model + each member's raw deviation from the raw mean (scaled later)
    d["HYB"] = d.groupby(CASE)["MEM"].transform("mean") + (d["v"] - d["ens_mean"])
    # consensus-centred: M1.5's corrected mean + each member's raw deviation from the raw mean (scaled later)
    d = d.merge(TCCA_MEAN, on=CASE, how="left")
    d["CC"] = d["tcca_mean"] + (d["v"] - d["ens_mean"])
    return d


def run_source(source: str, model_name: str, train_seasons: list[int]):
    d = pd.read_parquet(FEATURES_DIR / f"members_{source}_{model_name}.parquet")
    d = d[d["verifiable"]]
    feats = features(d)
    # out-of-fold predictions over the training seasons (for the spread factors)
    oof = []
    for s in train_seasons:
        tr = d[d["season"].isin([x for x in train_seasons if x != s])]
        te = d[d["season"] == s]
        if tr.empty or te.empty:
            continue
        oof.append(add_methods(te, tr, fit(tr, feats), feats))
    if not oof:  # a single training season (our IFS-ENS: 2024): in-sample for the member-wise methods; the consensus-centred
        tr = d[d["season"].isin(train_seasons)]  # ensembles only take the centre and raw member shapes from these rows
        oof.append(add_methods(tr, tr, fit(tr, feats), feats))
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(OUT / f"oof_{source}_{model_name}.parquet", index=False)  # used by scripts/ens_smooth.py
    lam = {m: sc.spread_scale(oof, m) for m in ("BIAS", "MEM", "HYB")}
    lam["CC"] = sc.spread_scale(oof[oof["CC"].notna()], "CC")
    tr = d[d["season"].isin(train_seasons)]
    mdl = fit(tr, feats)
    test = add_methods(d[d["season"].isin(TEST)], tr, mdl, feats)
    test = sc.apply_scale(test, "BIAS", lam["BIAS"], "BIASC")
    test = sc.apply_scale(test, "MEM", lam["MEM"], "MEMC")
    test = sc.apply_scale(test, "HYB", lam["HYB"], "HYBC")
    test = sc.apply_scale(test, "CC", lam["CC"], "CCC")
    imp = pd.Series(mdl.feature_importance("gain"), index=mdl.feature_name())
    print(f"{source}/{model_name}: features {len(feats)}, train rows {len(tr):,}, test rows {len(test):,}, "
          f"lambda HYB {lam['HYB'].round(1).to_dict()} CC {lam['CC'].round(1).to_dict()}  ({time.time() - t_start:.0f} s)", flush=True)
    test.to_parquet(OUT / f"test_{source}_{model_name}.parquet", index=False)
    return test, lam, (100 * imp / imp.sum()).sort_values(ascending=False)


def scores_for(test: pd.DataFrame, label: str) -> dict[str, pd.DataFrame]:
    out = {}
    for m in ("RAW", "BIAS", "BIASC", "MEM", "MEMC", "HYBC", "CCC"):
        s = sc.ensemble_case_scores(test[test[m].notna()], m)
        s["system"] = f"{label} {m}"
        out[f"{label} {m}"] = s
    return out


def tcca_scores() -> pd.DataFrame:
    F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet", columns=CASE + ["obs_vmax"])
    oof = pd.read_parquet(DATA_DIR / "results" / "m15" / "oof_vmax_A.parquet").merge(F, on=CASE)
    te = pd.read_parquet(DATA_DIR / "results" / "m15" / "test_vmax_A.parquet").merge(F, on=CASE)
    te = calibrate.apply(te, calibrate.fit(oof, oof["obs_vmax"]))
    s = sc.quantile_case_scores(te)
    s["system"] = "M1.5 TCCA (conformal)"
    return s


def summarize(scores: dict[str, pd.DataFrame], title: str, ref: str, diffs: list[tuple[str, str]]) -> str:
    keys = None
    for s in scores.values():
        k = s[s["lead"].isin(LEADS_EVAL)][CASE]
        keys = k if keys is None else keys.merge(k, on=CASE)
    rows, sub = [], {}
    for name, s in scores.items():
        h = s.merge(keys, on=CASE)
        sub[name] = h
        rows.append({"system": name, "CRPS": h["crps"].mean() if "crps" in h else np.nan, "quantile score": h["qs"].mean(),
                     "MAE of mean": h["abs_err_mean"].mean(), "10–90% coverage": h["inside"].mean(),
                     "spread/RMSE": (h["std"].mean() / np.sqrt(((h["mean"] - h["obs"]) ** 2).mean())) if "std" in h else np.nan})
    t = pd.DataFrame(rows).set_index("system").sort_values("quantile score")
    n_storms = keys["storm_id"].nunique()
    txt = f"\n### {title}\n\n{n_storms} storms, {len(keys)} cases (storm × t0 × lead), leads 12–120 h, homogeneous.\n\n" + t.to_markdown(floatfmt=".2f") + "\n"
    for a, b, col in diffs:
        if a in sub and b in sub:
            d, lo, hi = sc.storm_bootstrap_diff(sub[a], sub[b], col)
            txt += f"- {col}: {a} − {b} = {d:+.2f} [{lo:+.2f}, {hi:+.2f}]{' **significant**' if hi < 0 or lo > 0 else ''}\n"
    pit = {n: np.histogram(s["pit"].dropna(), bins=10, range=(0, 1))[0] / s["pit"].notna().sum() for n, s in sub.items() if "pit" in s}
    txt += "\nPIT histogram (10 bins, ideal 0.10 each):\n\n" + pd.DataFrame(pit, index=[f"{i/10:.1f}" for i in range(10)]).T.to_markdown(floatfmt=".2f") + "\n"
    return txt


def ri_brier(test: pd.DataFrame, label: str) -> dict:
    """Brier score of P(observed +30 kt / 24 h) from the share of members whose own 24-h change is >= 30 kt (leads >= 36 h)."""
    out = {}
    for m in ("RAW", "BIASC", "MEMC", "HYBC", "CCC"):
        d = test[["storm_id", "t0", "lead", "member", m, "obs_dv24"]].dropna(subset=[m]).copy()
        prev = d[["storm_id", "t0", "lead", "member", m]].assign(lead=lambda x: x["lead"] + 24).rename(columns={m: "prev"})
        d = d.merge(prev, on=["storm_id", "t0", "lead", "member"])
        d = d[d["lead"].between(36, 120) & d["obs_dv24"].notna()]
        p = (d[m] - d["prev"] >= 30).groupby([d["storm_id"], d["t0"], d["lead"]]).mean()
        o = (d.drop_duplicates(CASE).set_index(CASE)["obs_dv24"] >= 30).astype(float).loc[p.index]
        out[f"{label} {m}"] = {"Brier": float(((p - o) ** 2).mean()), "mean P": float(p.mean()), "obs freq": float(o.mean()),
                               "Brier (climatology)": float(((o.mean() - o) ** 2).mean()), "cases": int(len(p))}
    return out


report = [f"# Corrected intensity ensembles ({TEST_LABEL} test)\n",
          "Raw = member intensities from a tracker (NHC's or ours) on the model output. BIAS = + per-lead mean bias; MEM = member-wise "
          "LightGBM correction; C = spread scaled about the mean (lambda per lead, fit out-of-fold on training seasons). CRPS = fair "
          "ensemble CRPS; quantile score = mean pinball loss at 10/50/90%; spread/RMSE ≈ 1 when the ensemble spread matches the "
          "error of its mean. All values in kt except coverage and ratios. CIs: 95% storm-block bootstrap.\n"]

# NHC source: train 2018-2024 (ECMWF members are public in the archive only from 2020)
tg, lam_g, imp_g = run_source("nhc", "gefs", list(range(2018, 2025)))
te, lam_e, imp_e = run_source("nhc", "ecmwf", list(range(2020, 2025)))
# pilot source: everything before the test seasons (IFS-ENS store starts 2024-04; GEFS extended back to 2020-10)
pi, lam_pi, imp_pi = run_source("pilot", "ifs", [2024])
pg, lam_pg, imp_pg = run_source("pilot", "gefs", list(range(2020, 2025)))
tcca = tcca_scores()

S = {**scores_for(tg, "NHC-GEFS"), **scores_for(te, "NHC-ECMWF"), **scores_for(pi, "Ours-IFS"), **scores_for(pg, "Ours-GEFS")}
tc = {"M1.5 TCCA (conformal)": tcca}

# Group 1: IFS-ENS cycles (t0 = 12Z): ECMWF members from NHC's tracker vs our IFS-ENS extraction
g1 = {k: v[v["t0"].dt.hour == 12] for k, v in S.items() if k.startswith(("NHC-ECMWF", "Ours-IFS"))}
g1.update({k: v[v["t0"].dt.hour == 12] for k, v in tc.items()})
report.append("\n## ECMWF ensemble on IFS-ENS cycles (t0 = 12Z)\n" + summarize(
    g1, TEST_LABEL, "NHC-ECMWF RAW",
    [("Ours-IFS HYBC", "NHC-ECMWF HYBC", "crps"), ("Ours-IFS HYBC", "Ours-IFS RAW", "crps"), ("NHC-ECMWF HYBC", "NHC-ECMWF BIASC", "crps"),
     ("Ours-IFS HYBC", "Ours-IFS BIASC", "crps"), ("Ours-IFS HYBC", "Ours-IFS MEMC", "crps"), ("Ours-IFS HYBC", "M1.5 TCCA (conformal)", "qs"),
     ("Ours-IFS CCC", "Ours-IFS HYBC", "crps"), ("Ours-IFS CCC", "M1.5 TCCA (conformal)", "qs"), ("Ours-IFS CCC", "NHC-ECMWF CCC", "crps"),
     ("NHC-ECMWF HYBC", "M1.5 TCCA (conformal)", "qs")]))
# Group 2: GEFS cycles (t0 = 06Z)
g2 = {k: v[v["t0"].dt.hour == 6] for k, v in S.items() if k.startswith(("NHC-GEFS", "Ours-GEFS"))}
g2.update({k: v[v["t0"].dt.hour == 6] for k, v in tc.items()})
report.append("\n## GEFS on GEFS-pilot cycles (t0 = 06Z)\n" + summarize(
    g2, TEST_LABEL, "NHC-GEFS RAW",
    [("Ours-GEFS HYBC", "NHC-GEFS HYBC", "crps"), ("Ours-GEFS HYBC", "Ours-GEFS RAW", "crps"), ("NHC-GEFS HYBC", "NHC-GEFS BIASC", "crps"),
     ("Ours-GEFS HYBC", "Ours-GEFS BIASC", "crps"), ("NHC-GEFS HYBC", "NHC-GEFS MEMC", "crps"), ("NHC-GEFS HYBC", "M1.5 TCCA (conformal)", "qs"),
     ("Ours-GEFS CCC", "Ours-GEFS HYBC", "crps"), ("Ours-GEFS CCC", "M1.5 TCCA (conformal)", "qs"), ("Ours-GEFS CCC", "NHC-GEFS CCC", "crps")]))
# Group 3: NHC-tracker ensembles on all test cycles where both exist (bigger sample)
g3 = {k: v for k, v in S.items() if k.startswith(("NHC-GEFS", "NHC-ECMWF")) and k.endswith(("RAW", "BIASC", "MEMC", "HYBC", "CCC"))}
g3.update(tc)
report.append("\n## NHC-tracker ensembles, all test cycles\n" + summarize(
    g3, TEST_LABEL, "NHC-GEFS RAW",
    [("NHC-ECMWF HYBC", "NHC-GEFS HYBC", "crps"), ("NHC-GEFS HYBC", "NHC-GEFS BIASC", "crps"), ("NHC-GEFS HYBC", "M1.5 TCCA (conformal)", "qs"),
     ("NHC-ECMWF HYBC", "M1.5 TCCA (conformal)", "qs")]))

ri = {}
for t, lab in ((tg, "NHC-GEFS"), (te, "NHC-ECMWF"), (pi, "Ours-IFS"), (pg, "Ours-GEFS")):
    ri.update(ri_brier(t, lab))
report.append("\n## Rapid-intensification probability (leads 36–120 h, each source's own test cases)\n\n"
              "P(RI) = share of members whose own 24-h change is ≥ 30 kt. Lower Brier is better; the climatology column uses the observed frequency.\n\n"
              + pd.DataFrame(ri).T.to_markdown(floatfmt=".3f") + "\n")
report.append("\n## Spread factors (lambda by lead) and top features\n\n")
for lab, lam, imp in (("NHC-GEFS", lam_g, imp_g), ("NHC-ECMWF", lam_e, imp_e), ("Ours-IFS", lam_pi, imp_pi), ("Ours-GEFS", lam_pg, imp_pg)):
    report.append(f"- **{lab}** lambda BIAS: {lam['BIAS'].round(1).to_dict()}; MEM: {lam['MEM'].round(1).to_dict()}; HYB: {lam['HYB'].round(1).to_dict()}; CC: {lam['CC'].round(1).to_dict()}\n"
                  f"  top features: {', '.join(f'{k} {v:.0f}%' for k, v in imp.head(8).items())}\n")
report.append(f"\n_Run time: {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_results.md").write_text("".join(report))
print(f"wrote docs/ensemble_results.md in {time.time() - t_start:.0f} s")
