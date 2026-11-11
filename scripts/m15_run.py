"""M1.5: train and evaluate the early-cycle consensus correction (variants A: no SHIPS, B: + SHIPS).

Outputs (data/results/m15/): out-of-fold and test predictions, plus markdown tables in docs/m15_results.md.
Run under the compute cap: scripts/run_capped.sh .venv/bin/python scripts/m15_run.py
"""
import numpy as np
import pandas as pd

from tcic import evaluate as ev
from tcic.config import DATA_DIR, FEATURES_DIR, INTERIM_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.consensus import model as m
from tcic.data.atcf import merge_interpolated

OUT = DATA_DIR / "results" / "m15"
OUT.mkdir(parents=True, exist_ok=True)
CV_SEASONS = list(range(2018, 2025))
TEST = list(TEST_SEASONS)
BASELINES = ["OFCL", "IVCN", "HCCA", "FSSE", "DSHP", "LGEM", "HFAI", "HWFI", "AVNI", "EMXI"]
NHC_LEADS = list(ev.NHC_LEADS)
ALL_LEADS = list(range(12, 169, 12))

F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet")
A = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", BASELINES + ["GDMN", "GDMI", "AVN2", "EMX2", "HFA2", "HWF2"])])
A["storm_id"] = A["storm_id"].astype(str)
A = merge_interpolated(A)
truth = ev.load_truth("nhc")
report = ["# M1.5 results: early-cycle consensus correction\n",
          "Truth: final b-deck best track. Cases: TC at t0 and at the valid time (NHC convention). Homogeneous samples. "
          "95% CIs from resampling storms. TCCA = variant A (no SHIPS), TCCB = variant B (+ operational SHIPS), "
          "TCRF = our equal-weight early-aid mean (the reference both variants correct).\n"]


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".1f") + "\n"


def ours_cases(pred, tech, var="vmax"):
    return m.as_cases(pred, F, tech, var)


def ref_cases(pred, var="vmax"):
    ref = pred[m.ID_COLS + ["lead"]].copy()
    ref["mean"] = F.set_index(["storm_id", "t0", "lead"]).loc[pd.MultiIndex.from_frame(pred[["storm_id", "t0", "lead"]]), m.TARGETS[var][1]].to_numpy()
    return m.as_cases(ref, F, "TCRF", var)


def compare(cases: pd.DataFrame, techs: list[str], title: str, leads: list[int], unit: str = "kt", diffs=(("TCCA", "IVCN"), ("TCCA", "HCCA"), ("TCCB", "HCCA"), ("TCCB", "TCCA"))):
    techs = [t for t in techs if t in set(cases["tech"])]
    c = ev.homogeneous(cases[cases["tau"].isin(leads)], techs)
    if c.empty:
        report.append(f"\n## {title}\n\nNo homogeneous cases.\n")
        return
    o = ev.overall(c, techs).sort_values("mae")
    t = ev.by_lead(c, techs)["mae"].unstack("tau")
    report.append(f"\n## {title}\n\n{c[['storm_id', 'init']].drop_duplicates().shape[0]} forecasts, {c['storm_id'].nunique()} storms, "
                  f"{len(c) // len(techs)} verifying points per model.\n\n**Overall (all leads pooled)**\n\n")
    report.append(md(o[["mae", "mae_lo", "mae_hi", "bias", "bias_lo", "bias_hi"]]))
    report.append(f"\n**MAE by lead ({unit})**\n\n" + md(t))
    lines = []
    for a, b in diffs:
        if a in techs and b in techs:
            d, lo, hi = ev.mae_difference(c, a, b)
            sig = "significant" if (hi < 0 or lo > 0) else "not significant"
            lines.append(f"- MAE({a}) − MAE({b}) = {d:+.2f} {unit} [{lo:+.2f}, {hi:+.2f}], {sig}")
    if lines:
        report.append("\n" + "\n".join(lines) + "\n")
    ri = c[c["ri"]]
    if ri["storm_id"].nunique() >= 3:
        report.append(f"\n**Rapid intensification cases** ({ri['storm_id'].nunique()} storms, {len(ri) // len(techs)} points)\n\n")
        report.append(md(ev.overall(ri, techs).sort_values("mae")[["mae", "mae_lo", "mae_hi", "bias"]]))


def baseline_cases(seasons, leads):
    a = A[A["init"].dt.year.isin(seasons)]
    return ev.pair(a, truth, [t for t in BASELINES + ["GDMN", "GDMI"]], leads=leads)


for var in ("vmax", "pmin"):
    print(f"=== {var} ===")
    oof = {v: m.leave_one_season_out(F, v, var, CV_SEASONS) for v in ("A", "B")}
    test = {v: m.train_test(F, v, var, CV_SEASONS, TEST)[0] for v in ("A", "B")}
    for v in ("A", "B"):
        oof[v].to_parquet(OUT / f"oof_{var}_{v}.parquet", index=False)
        test[v].to_parquet(OUT / f"test_{var}_{v}.parquet", index=False)
    if var == "pmin":
        report.append("\n# Minimum pressure (mb)\n")
        # Early aids carry no pressure, so compare with our reference and with persistence of the CARQ pressure.
        persist = test["A"][m.ID_COLS + ["lead"]].copy()
        persist["mean"] = F.set_index(["storm_id", "t0", "lead"]).loc[pd.MultiIndex.from_frame(persist[["storm_id", "t0", "lead"]]), "pmin0"].to_numpy()
        cases = pd.concat([ours_cases(test["A"], "TCCA", var), ours_cases(test["B"], "TCCB", var), ref_cases(test["A"], var),
                           ours_cases(persist, "PERS", var)])
        compare(cases, ["TCCA", "TCCB", "TCRF", "PERS"], f"{TEST_LABEL} test, pmin, leads 12–120 h", NHC_LEADS, unit="mb",
                diffs=(("TCCA", "TCRF"), ("TCCB", "TCCA")))
        continue

    # --- vmax: cross-validation (2018-2024, leave one season out) ---
    cv = pd.concat([ours_cases(oof["A"], "TCCA"), ours_cases(oof["B"], "TCCB"), ref_cases(oof["A"]),
                    baseline_cases(CV_SEASONS, ALL_LEADS)])
    report.append("\n# Maximum wind (kt)\n")
    compare(cv, ["TCCA", "TCCB", "TCRF", "OFCL", "IVCN", "HCCA", "FSSE", "DSHP", "LGEM"],
            "Cross-validation 2018–2024 (leave one season out), leads 12–120 h", NHC_LEADS)
    per_season = []
    for s in CV_SEASONS:
        c = ev.homogeneous(cv[(cv["init"].dt.year == s) & cv["tau"].isin(NHC_LEADS)], ["TCCA", "TCCB", "IVCN", "HCCA", "OFCL"])
        if not c.empty:
            per_season.append(ev.overall(c, ["TCCA", "TCCB", "IVCN", "HCCA", "OFCL"], n_boot=1)["mae"].rename(s))
    report.append("\n**Per-season MAE (kt), CV, leads 12–120 h, homogeneous**\n\n" + md(pd.DataFrame(per_season)))

    # --- vmax: held-out test (2025 + 2026 combined) ---
    tc = pd.concat([ours_cases(test["A"], "TCCA"), ours_cases(test["B"], "TCCB"), ref_cases(test["A"]), baseline_cases(TEST, ALL_LEADS)])
    tc.to_parquet(OUT / "test_cases_vmax.parquet", index=False)
    # no EMXI / HWFI / FSSE here: they are absent from the 2026 real-time decks, and a homogeneous sample would silently drop 2026
    compare(tc, ["TCCA", "TCCB", "TCRF", "OFCL", "IVCN", "HCCA", "DSHP", "LGEM", "HFAI", "AVNI"],
            f"{TEST_LABEL} test (trained on 2018–2024), leads 12–120 h", NHC_LEADS)
    for s in TEST:  # each test season on its own (2026 real-time decks carry no ECMWF aids)
        compare(tc[tc["init"].dt.year == s], ["TCCA", "TCCB", "TCRF", "OFCL", "IVCN", "HCCA"], f"{s} only (trained on 2018–2024), leads 12–120 h", NHC_LEADS)
    compare(tc, ["TCCA", "TCCB", "TCRF", "GDMI", "HCCA", "IVCN"], f"{TEST_LABEL} test vs FNV3 early (GDMI), leads 12–168 h", ALL_LEADS,
            diffs=(("TCCA", "GDMI"), ("TCCB", "GDMI"), ("TCCB", "TCCA")))
    # A vs B restricted to cycles with SHIPS available
    with_ships = F[F["has_ships"] == 1][["storm_id", "t0"]].drop_duplicates().rename(columns={"t0": "init"})
    tcs = tc.merge(with_ships, on=["storm_id", "init"])
    compare(tcs, ["TCCA", "TCCB", "TCRF", "HCCA", "IVCN"], f"{TEST_LABEL} test, only cycles with SHIPS, leads 12–120 h", NHC_LEADS)

    # quantile calibration (test, vmax)
    cal = []
    for v in ("A", "B"):
        p = test[v].merge(F[["storm_id", "t0", "lead", "obs_vmax"]], on=["storm_id", "t0", "lead"])
        cal.append({"variant": v, "below q10": (p["obs_vmax"] < p["q10"]).mean(), "below q50": (p["obs_vmax"] < p["q50"]).mean(),
                    "below q90": (p["obs_vmax"] < p["q90"]).mean(), "q10–q90 coverage": ((p["obs_vmax"] >= p["q10"]) & (p["obs_vmax"] <= p["q90"])).mean()})
    report.append("\n**Quantile calibration on the " + TEST_LABEL + " test (ideal: 0.10 / 0.50 / 0.90 / 0.80)**\n\n" + pd.DataFrame(cal).set_index("variant").to_markdown(floatfmt=".2f") + "\n")

    # feature importance (gain), variant B, trained on all CV seasons
    _, mdl = m.train_test(F, "B", var, CV_SEASONS, TEST, quantiles=False)
    imp = pd.Series(mdl.feature_importance("gain"), index=mdl.feature_name()).sort_values(ascending=False)
    report.append("\n**Top 20 features by gain (variant B)**\n\n" + md((100 * imp / imp.sum()).head(20).rename("% gain").to_frame()))

(ROOT / "docs" / "m15_results.md").write_text("".join(report))
print("wrote docs/m15_results.md")
