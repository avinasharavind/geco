"""M2: do gridded IFS-ENS / GEFS features add skill beyond the M1.5 early consensus? (max wind)

Experiments per model (rows = that model's early cycles: IFS-ENS t0=12Z, GEFS t0=06Z), leave-one-season-out over each model's seasons
(IFS-ENS 2024-2026; GEFS 2020-2026 after the back-extension):
  G       correct the model's own (CARQ-shifted) ensemble-mean intensity from gridded features + storm state (paper-like)
  A-same  M1.5 variant-A features, retrained on exactly these rows/seasons (fair baseline)
  A+G     M1.5 variant-A features + gridded features (the "does gridded add?" test: compare with A-same)
  G-pool  G trained on both models' rows with a model flag
Baselines: full M1.5 TCCA (trained 2018-24 on all cycles; 2025-26 test), HCCA, IVCN, OFCL, uncorrected aid mean.
Run: scripts/run_capped.sh .venv/bin/python scripts/m2_run.py   (~minutes)
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tcic import evaluate as ev
from tcic.config import DATA_DIR, FEATURES_DIR, INTERIM_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.consensus import model as m
from tcic.data.atcf import merge_interpolated

OUT = DATA_DIR / "results" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
LEADS = list(range(12, 169, 12))
NHC_LEADS = list(ev.NHC_LEADS)
BASE = ["vmax0", "pmin0", "dv6", "dv12", "dv24", "u_motion", "v_motion", "lat0", "lon0", "dist2land0", "lead", "basin_ep", "doy"]

t_start = time.time()
G = {mdl: pd.read_parquet(FEATURES_DIR / f"gridded_{mdl}.parquet") for mdl in ("ifs", "gefs")}
F15 = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet")
M15A = m.feature_columns(F15, "A")
GRID = [c for c in G["ifs"].columns if c.startswith(("g_", "g0_", "e_")) and c in G["gefs"].columns] + ["env_pf1"]
GRID_ALL = {mdl: [c for c in G[mdl].columns if c.startswith(("g_", "g0_", "e_"))] + ["env_pf1"] for mdl in G}


def fit_predict(train, test, feats, ref):
    y = train["obs_vmax"] - train[ref]
    w = np.where(train["ri"] == 1, 2.0, 1.0)
    booster = lgb.train(m.PARAMS, lgb.Dataset(train[feats], y, weight=w), num_boost_round=m.N_ROUNDS)
    return np.clip(test[ref].to_numpy() + booster.predict(test[feats]), 10, 200), booster


def loso(df, feats, ref, tech):
    rows = df[df["verifiable"] & df["obs_vmax"].notna() & df[ref].notna()]
    out = []
    for s in sorted(rows["season"].unique()):
        tr, te = rows[rows["season"] != s], rows[rows["season"] == s]
        if te.empty or tr.empty:
            continue
        p, _ = fit_predict(tr, te, feats, ref)
        out.append(pd.DataFrame({"storm_id": te["storm_id"], "init": te["t0"], "tau": te["lead"], "fcst": p, "obs": te["obs_vmax"],
                                 "ri": te["ri"].astype(bool), "tech": tech}))
    c = pd.concat(out, ignore_index=True)
    c["err"] = c["fcst"] - c["obs"]
    return c


def ref_cases(df, col, tech):
    r = df[df["verifiable"] & df[col].notna()]
    c = pd.DataFrame({"storm_id": r["storm_id"], "init": r["t0"], "tau": r["lead"], "fcst": r[col], "obs": r["obs_vmax"], "ri": r["ri"].astype(bool), "tech": tech})
    c["err"] = c["fcst"] - c["obs"]
    return c


A = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", ["OFCL", "IVCN", "HCCA", "GDMI", "GDMN"])])
A["storm_id"] = A["storm_id"].astype(str)
A = merge_interpolated(A)
truth = ev.load_truth("nhc")
full_tcca = m.as_cases(pd.read_parquet(DATA_DIR / "results" / "m15" / "test_vmax_A.parquet"), F15, "TCCA-full")

report = ["# M2 results: gridded IFS-ENS / GEFS features (max wind)\n",
          "Leave-one-season-out over each model's seasons: IFS-ENS 2024–2026, GEFS 2020–2026 (2026 truth = provisional b-decks). Homogeneous samples, storm-bootstrap 95% CIs. "
          "G = gridded-only correction of the model's CARQ-shifted ensemble mean; A-same = M1.5 variant A retrained on the same rows; "
          "A+G = A-same + gridded features; G-pool = G trained on both models; TCCA-full = the M1.5 model trained on all 2018–24 cycles "
          f"({TEST_LABEL} test); TCRF = uncorrected aid mean; ANCH = the CARQ-shifted ensemble mean that G corrects.\n"]


def md(df):
    return df.to_markdown(floatfmt=".2f") + "\n"


def section(cases, techs, title, diffs, leads=NHC_LEADS):
    techs = [t for t in techs if t in set(cases["tech"])]
    c = ev.homogeneous(cases[cases["tau"].isin(leads)], techs)
    if c.empty:
        report.append(f"\n### {title}\n\nNo homogeneous cases.\n")
        return
    o = ev.overall(c, techs).sort_values("mae")
    report.append(f"\n### {title}\n\n{c['storm_id'].nunique()} storms, {c[['storm_id', 'init']].drop_duplicates().shape[0]} forecasts, "
                  f"{len(c) // len(techs)} points per system.\n\n" + md(o[["mae", "mae_lo", "mae_hi", "bias"]]))
    report.append("\nMAE by lead (kt)\n\n" + md(ev.by_lead(c, techs)["mae"].unstack("tau")))
    for a, b in diffs:
        if a in techs and b in techs:
            d, lo, hi = ev.mae_difference(c, a, b)
            report.append(f"- MAE({a}) − MAE({b}) = {d:+.2f} kt [{lo:+.2f}, {hi:+.2f}]{' **significant**' if hi < 0 or lo > 0 else ''}\n")
    ri = c[c["ri"]]
    if ri["storm_id"].nunique() >= 3:
        report.append(f"\nRI cases ({ri['storm_id'].nunique()} storms, {len(ri) // len(techs)} points)\n\n" + md(ev.overall(ri, techs).sort_values("mae")[["mae", "mae_lo", "mae_hi", "bias"]]))
        for a, b in diffs[:2]:
            if a in techs and b in techs:
                d, lo, hi = ev.mae_difference(ri, a, b)
                report.append(f"- RI: MAE({a}) − MAE({b}) = {d:+.2f} kt [{lo:+.2f}, {hi:+.2f}]{' **significant**' if hi < 0 or lo > 0 else ''}\n")


pool = pd.concat([G["ifs"].assign(is_ifs=1), G["gefs"].assign(is_ifs=0)], ignore_index=True)
pool_cases = loso(pool, BASE + GRID + ["is_ifs"], "g_anchor_vmax", "G-pool")
importance = {}
for mdl in ("ifs", "gefs"):
    df = G[mdl]
    name = "IFS-ENS (t0 = 12Z)" if mdl == "ifs" else "GEFS (t0 = 06Z)"
    cs = [loso(df, BASE + GRID_ALL[mdl], "g_anchor_vmax", "G"),
          loso(df, M15A, "ref_vmax", "A-same"),
          loso(df, M15A + GRID_ALL[mdl], "ref_vmax", "A+G"),
          pool_cases.merge(df[["storm_id", "t0", "lead"]].rename(columns={"t0": "init", "lead": "tau"}), on=["storm_id", "init", "tau"]),
          ref_cases(df, "ref_vmax", "TCRF"), ref_cases(df, "g_anchor_vmax", "ANCH")]
    keys = df[["storm_id", "t0"]].drop_duplicates().rename(columns={"t0": "init"})
    base = ev.pair(A.merge(keys, on=["storm_id", "init"]), truth, ["OFCL", "IVCN", "HCCA", "GDMI"], leads=LEADS)
    cases = pd.concat(cs + [base, full_tcca.merge(keys, on=["storm_id", "init"])], ignore_index=True)
    cases.to_parquet(OUT / f"cases_{mdl}.parquet", index=False)
    report.append(f"\n## {name}\n")
    systems = ["A+G", "A-same", "G", "G-pool", "ANCH", "TCRF", "HCCA", "IVCN", "OFCL"]
    seasons = sorted(df["season"].unique())
    section(cases, systems, f"All seasons {seasons[0]}–{seasons[-1]} (out-of-fold), leads 12–120 h",
            [("A+G", "A-same"), ("G", "ANCH"), ("A+G", "HCCA"), ("G", "A-same"), ("G-pool", "G")])
    c25 = cases[cases["init"].dt.year.isin(TEST_SEASONS)]
    section(c25, systems + ["TCCA-full"], f"{TEST_LABEL} (out-of-fold; comparable to M1.5's test seasons), leads 12–120 h",
            [("A+G", "A-same"), ("A+G", "TCCA-full"), ("A+G", "HCCA"), ("G", "ANCH")])
    section(c25, ["A+G", "A-same", "G", "TCCA-full", "GDMI", "HCCA"], f"{TEST_LABEL} vs FNV3 early (GDMI), leads 12–168 h",
            [("A+G", "GDMI"), ("A+G", "A-same")], leads=LEADS)
    # env_member check (IFS 2026 uses pf1): error of G in 2026 vs other seasons is confounded with season; just report counts
    if mdl == "ifs":
        em = df.drop_duplicates(["storm_id", "t0"])
        em = em["env_member"].fillna("cf").groupby(em["season"]).value_counts().to_dict()  # runs extracted before the fallback have no field: control
        report.append(f"\nIFS environment member by season (storm-cycles): {em}. Note env_pf1 is collinear with season 2026.\n")
    # feature importance for A+G trained on all seasons
    rows = df[df["verifiable"] & df["ref_vmax"].notna()]
    _, b = fit_predict(rows, rows.head(1), M15A + GRID_ALL[mdl], "ref_vmax")
    imp = pd.Series(b.feature_importance("gain"), index=b.feature_name())
    imp = 100 * imp / imp.sum()
    importance[mdl] = imp
    grid_share = imp[[c for c in imp.index if c.startswith(("g_", "g0_", "e_")) or c == "env_pf1"]].sum()
    report.append(f"\nA+G feature importance (gain): gridded features {grid_share:.0f}% of total. Top 15:\n\n" + md(imp.sort_values(ascending=False).head(15).rename("% gain").to_frame()))

report.append(f"\n_Run time: {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "m2_results.md").write_text("".join(report))
print(f"wrote docs/m2_results.md in {time.time() - t_start:.0f} s")
