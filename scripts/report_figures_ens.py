"""Figures, tables and numbers for the ensemble part of the write-up (architecture diagram, spread vs error, case studies,
ensemble summary). Reads saved results only. Run after scripts/ens_spread.py:  nice -n 19 .venv/bin/python scripts/report_figures_ens.py
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tcic.config import DATA_DIR, FINAL_ENSEMBLE, INTERIM_DIR, ROOT, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
OUT = ROOT / "docs" / "report"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
BLUE, AQUA, VIOLET, ORANGE = "#2a78d6", "#1baf7a", "#6250d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": MUTED,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
numbers = {}

# ---------------------------------------------------------------- architecture diagram
fig, ax = plt.subplots(figsize=(7.6, 5.8))
ax.set_xlim(-1.5, 101.5)  # room for the rounded corners of the outermost boxes
ax.set_ylim(0, 82)
ax.axis("off")


def box(x, y, w, h, title, body, edge=MUTED, fill="#f6f5f2", tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge, lw=1.0, clip_on=False))
    ax.text(x + 1.5, y + h - 1.8, title, fontsize=7.4, weight="bold", color=tcol, va="top")
    ax.text(x + 1.5, y + h - 5.6, body, fontsize=6.4, color=INK2, va="top", linespacing=1.4)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0))


ax.text(0, 81.5, "Inputs (all available by t0 + 3 h)", fontsize=7.4, color=INK2, weight="bold", va="top")
box(0, 57, 28, 21, "NHC early aids", "14 interpolated models:\nGFS, GEFS mean, ECMWF,\nECMWF ens. mean, UKMET,\nCMC, NAVGEM, HWRF, HMON,\nHAFS-A/B, COAMPS-TC,\nDSHP, LGEM")
box(0, 38, 28, 16, "Storm state (CARQ)", "intensity, pressure and\nposition at t0; 6/12/24-h\nintensity trend; motion;\ndistance to land")
box(0, 19.5, 28, 15.5, "Ensemble members", "GEFS (31) and ECMWF (51)\nmember tracks and\nintensities (NHC's\ntracker or ours)")
box(0, 1, 28, 15.5, "Optional extras", "operational SHIPS\npredictors; gridded GEFS /\nIFS-ENS fields\n(environment, inner core)")
box(36, 44, 30, 34, "1 · Consensus correction", "reference = mean of the\navailable early aids\n\nLightGBM learns the residual\n(best track − reference) from\n~70 features; RI cases ×2\n\n→ corrected mean intensity\n→ predicted error σ̂ (spread model)\n→ RI probability (RI model)", edge=BLUE, tcol=BLUE)
box(36, 1, 30, 37, "2 · Member deviations", "each member's departure from\nits ensemble mean,\nre-anchored to zero at t0\n\nsmoothed in time\n(Gaussian, σ = 6 h)\n\nnormalised to unit spread\nwithin each forecast (keeps\nthe scenario shape only)", edge=AQUA, tcol="#138a60")
box(73, 44, 27, 34, "3 · Corrected ensemble", "member =\n   corrected mean\n   + 0.80 k(τ) σ̂ × normalised\n      deviation\n\nσ̂: this forecast's predicted\nerror (smoothed over lead);\nk(τ) fitted on past seasons", edge=VIOLET, tcol=VIOLET)
box(73, 1, 27, 37, "Outputs", "• deterministic intensity\n  (the corrected mean)\n• calibrated 10–90%\n  interval\n• calibrated probability of\n  rapid intensification\n• 31 / 51 member traces\n  that start at the observed\n  intensity and fan out\n  storm-specifically", edge=INK2)
arrow(28.6, 67, 35.4, 64)
arrow(28.6, 46, 35.4, 54)
arrow(28.6, 27, 35.4, 24)
arrow(28.6, 27, 35.4, 50)
arrow(28.6, 8, 35.4, 12)
arrow(66.6, 61, 72.4, 61, BLUE)
arrow(66.6, 22, 72.4, 50, AQUA)
arrow(86.5, 43.4, 86.5, 38.6, VIOLET)
fig.savefig(FIG / "fig_architecture.svg")
fig.savefig(OUT / "preview" / "fig_architecture.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- ensemble scores on 2025–26 (our ensembles vs FNV3), homogeneous
fn = fnv3_data.load_test()


def fnv3_early(cases):
    runs = fn[["storm_id", "init"]].drop_duplicates()
    c = cases[KEY].drop_duplicates().merge(runs, on="storm_id")
    c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))]
    c = c.sort_values("init").groupby(KEY).tail(1)
    m = c.merge(fn, on=["storm_id", "init"])
    m["lead"] = m["lead"] - ((m["t0"] - m["init"]) / pd.Timedelta(hours=1)).astype(int)
    m = m[m["lead"].isin(LEADS)].merge(cases.drop_duplicates(CASE)[CASE + ["obs_vmax"]], on=CASE)
    return m.rename(columns={"vmax": "FNV3"})


rows, by_lead = [], {}
for name, label, hour in (("nhc_gefs", "GEFS", None), ("pilot_ifs", "IFS-ENS", 12)):  # the products: GECO (NHC GEFS tracks, all cycles), IECO (12Z)
    t = pd.read_parquet(R / f"test_{name}.parquet")
    fin = FINAL_ENSEMBLE[name]
    t = t[(True if hour is None else t["t0"].dt.hour == hour) & t["lead"].isin(LEADS) & t[fin].notna()]
    S = {"Raw members": sc.ensemble_case_scores(t, "RAW"), "Corrected (ours)": sc.ensemble_case_scores(t, fin),
         "FNV3 members": sc.ensemble_case_scores(fnv3_early(t), "FNV3")}
    keys = S["Raw members"][CASE].merge(S["Corrected (ours)"][CASE], on=CASE).merge(S["FNV3 members"][CASE], on=CASE)
    H = {k: s.merge(keys, on=CASE) for k, s in S.items()}
    for k, h in H.items():
        err = h["mean"] - h["obs"]
        shown = {"Corrected (ours)": fin, "Raw members": "raw members"}.get(k, k)
        rows.append({"Ensemble": f"{label}: {shown}" if k != "FNV3 members" else f"{label} cycles: FNV3 members", "CRPS (kt)": round(h["crps"].mean(), 1),
                     "Bias of mean (kt)": round(err.mean(), 1), "10–90% coverage": f"{100 * h['inside'].mean():.0f}%",
                     "Spread / error": round(h["std"].mean() / np.sqrt((err ** 2).mean()), 2)})
        g = h.groupby("lead")
        by_lead[(label, k)] = pd.DataFrame({"spread": g["std"].mean(), "rmse": g.apply(lambda x: np.sqrt(((x["mean"] - x["obs"]) ** 2).mean()))})
    d, lo, hi = sc.storm_bootstrap_diff(H["Corrected (ours)"], H["FNV3 members"], "crps")
    numbers[f"ens_{label}_vs_fnv3"] = [round(d, 2), round(lo, 2), round(hi, 2)]
    numbers[f"ens_{label}_storms"], numbers[f"ens_{label}_cases"] = int(keys["storm_id"].nunique()), int(len(keys))
    for yr in TEST_SEASONS:  # each test season on its own (same cases)
        hy = {k: h[h["t0"].dt.year == yr] for k, h in H.items()}
        d, lo, hi = sc.storm_bootstrap_diff(hy["Corrected (ours)"], hy["FNV3 members"], "crps")
        numbers[f"ens_{label}_vs_fnv3_{yr}"] = [round(d, 2), round(lo, 2), round(hi, 2)]
        for k, h in hy.items():
            numbers[f"ens_{label}_{k.split()[0].lower()}_{yr}"] = {"crps": round(float(h["crps"].mean()), 2),
                                                                 "bias": round(float((h["mean"] - h["obs"]).mean()), 1),
                                                                 "coverage": round(float(h["inside"].mean()), 2)}
pd.DataFrame(rows).to_csv(TAB / "ensemble_summary.csv", index=False)

# spread vs error figure
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, label in zip(axes, ("GEFS", "IFS-ENS")):
    for k, col, nm in (("Corrected (ours)", BLUE, "ours"), ("FNV3 members", VIOLET, "FNV3"), ("Raw members", MUTED, "raw")):
        d = by_lead[(label, k)]
        ax.plot(d.index, d["spread"], color=col, lw=2, marker="o", ms=3, mec="white", mew=0.5, label=f"{k}: spread")
        if k != "Raw members":
            ax.plot(d.index, d["rmse"], color=col, lw=1.4, ls="--", label=f"{k}: error of mean")
    ax.set_title(f"{label} ({FINAL_ENSEMBLE['nhc_gefs' if label == 'GEFS' else 'pilot_ifs']}), {'all' if label == 'GEFS' else '12Z'} cycles, 2025–26", loc="left", color=INK)
    ax.set_xlabel("Lead time (h)")
    ax.set_xticks(range(12, 121, 24))
axes[0].set_ylabel("kt")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=7)
fig.subplots_adjust(wspace=0.08)
fig.savefig(FIG / "fig_spread_vs_error.svg")
plt.close(fig)

# ---------------------------------------------------------------- physical consistency (12-h zig-zag rate etc.)
rough = {}
for name, label in (("nhc_gefs", "GEFS"), ("pilot_ifs", "IFS-ENS")):
    t = pd.read_parquet(R / f"test_{name}.parquet")
    base = t.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0", "v_t0"]]
    for col, start, nm in (("RAW", "v_t0", f"{label} raw members"), ("CCC", "vmax0", f"{label} first corrected version"), (FINAL_ENSEMBLE[name], "vmax0", f"{FINAL_ENSEMBLE[name]} ({label})")):
        tc = t[t[col].notna()]
        b0 = base.merge(tc[KEY].drop_duplicates(), on=KEY)
        tr = pd.concat([tc[KEY + ["member", "lead", col]], b0.assign(lead=0).rename(columns={start: col})[KEY + ["member", "lead", col]]])
        rough[nm] = smo.roughness(tr, col)
fn2 = fn[(fn["storm_id"].str[2:4].astype(int) < 50) & (fn["lead"] <= 168)].rename(columns={"init": "t0"})
rough["FNV3 members"] = smo.roughness(fn2, "vmax")
bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet").assign(storm_id=lambda x: x["storm_id"].astype(str))
bt = bt[bt["time"].dt.year.isin(TEST_SEASONS)]
st = fn2[["storm_id", "t0"]].drop_duplicates().merge(bt[["storm_id", "time", "vmax"]], on="storm_id")
st = st[(st["time"] >= st["t0"]) & (st["time"] <= st["t0"] + pd.Timedelta(hours=168))]
st["lead"], st["member"] = ((st["time"] - st["t0"]) / pd.Timedelta(hours=1)).astype(int), 0
rough["Best track"] = smo.roughness(st, "vmax")
r6 = {"Best track": smo.roughness(st, "vmax", 6, 7)["flip rate"], "FNV3 members": smo.roughness(fn2, "vmax", 6, 7)["flip rate"]}
numbers["zigzag_6h"] = {k: round(v, 3) for k, v in r6.items()}
order = ["Best track", "FNV3 members", "GEFS raw members", "IFS-ENS raw members", "GEFS first corrected version", "IFS-ENS first corrected version",
         f"{FINAL_ENSEMBLE['nhc_gefs']} (GEFS)", f"{FINAL_ENSEMBLE['pilot_ifs']} (IFS-ENS)"]
pd.DataFrame([{"Traces": k, "Zig-zag rate": f"{rough[k]['flip rate']:.3f}", "Mean |12-h change| (kt)": f"{rough[k]['mean |Δ|']:.1f}",
               "Share of 12-h changes > 30 kt": f"{100 * rough[k]['share |Δ| > 30 kt']:.1f}%"} for k in order]).to_csv(TAB / "roughness.csv", index=False)

# ---------------------------------------------------------------- spread model and RI model summaries
from sklearn.metrics import roc_auc_score

sp = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
te_sp = sp[sp["season"].isin(TEST_SEASONS)]
numbers["spread_corr_2025"] = round(float(te_sp["sigma_hat"].corr(te_sp["abs_err"])), 2)
numbers["spread_corr_within_lead_2025"] = round(float(te_sp.groupby("lead").apply(lambda g: g["sigma_hat"].corr(g["abs_err"])).mean()), 2)
b = te_sp.assign(bin=pd.qcut(te_sp["sigma_hat"], 5, labels=["Calmest fifth", "Second", "Middle", "Fourth", "Most volatile fifth"]))
b.groupby("bin", observed=True).agg(**{"Cases": ("abs_err", "size"), "Predicted error (kt)": ("sigma_hat", "mean"), "Actual error (kt)": ("abs_err", "mean")}).round(1).reset_index().rename(columns={"bin": "Cases binned by predicted error"}).to_csv(TAB / "spread_bins.csv", index=False)
ri = pd.read_parquet(DATA_DIR / "results" / "calibration" / "ri_probabilities.parquet")
clim = float(ri[ri["season"] < 2025]["y"].mean())
r25 = ri[ri["season"].isin(TEST_SEASONS)]
fm = fnv3_early(r25.assign(obs_vmax=np.nan))  # FNV3 member share with +30 kt / 24 h
fm_all = fn.assign(t0=None)
runs = fn[["storm_id", "init"]].drop_duplicates()
c = r25[KEY].drop_duplicates().merge(runs, on="storm_id")
c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))].sort_values("init").groupby(KEY).tail(1)
mm = c.merge(fn, on=["storm_id", "init"])
mm["lead"] = mm["lead"] - ((mm["t0"] - mm["init"]) / pd.Timedelta(hours=1)).astype(int)
prev = mm.assign(lead=mm["lead"] + 24)[KEY + ["member", "lead", "vmax"]].rename(columns={"vmax": "prev"})
mm = mm.merge(prev, on=KEY + ["member", "lead"])
pf = (mm["vmax"] - mm["prev"] >= 30).groupby([mm[k] for k in CASE]).mean().rename("p_fnv3").reset_index()
h = r25.merge(pf, on=CASE)
y = h["y"].to_numpy()
bs = lambda p: float(np.mean((np.asarray(p) - y) ** 2))
rows_ri = [{"Probability of rapid intensification": "Climatology (2018–24 rate)", "Brier score": round(bs(np.full(len(y), clim)), 3), "Skill vs climatology": "0%", "AUC": "–"},
           {"Probability of rapid intensification": "RI model (calibrated)", "Brier score": round(bs(h["p_ri"]), 3), "Skill vs climatology": f"{100 * (1 - bs(h['p_ri']) / bs(np.full(len(y), clim))):.0f}%", "AUC": f"{roc_auc_score(y, h['p_ri']):.2f}"},
           {"Probability of rapid intensification": "FNV3 members (share rising ≥ 30 kt / 24 h)", "Brier score": round(bs(h["p_fnv3"]), 3), "Skill vs climatology": f"{100 * (1 - bs(h['p_fnv3']) / bs(np.full(len(y), clim))):.0f}%", "AUC": f"{roc_auc_score(y, h['p_fnv3']):.2f}"}]
pd.DataFrame(rows_ri).to_csv(TAB / "ri_summary.csv", index=False)
numbers["ri_cases"], numbers["ri_events"], numbers["ri_obs_rate"], numbers["ri_clim_rate"] = int(len(y)), int(y.sum()), round(float(y.mean()), 3), round(clim, 3)
rel = calib_rel = pd.cut(h["p_ri"], [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001], right=False)
pd.DataFrame({"Forecast probability": [str(i) for i in h.groupby(rel, observed=True).size().index],
              "Cases": h.groupby(rel, observed=True).size().values,
              "Mean forecast": h.groupby(rel, observed=True)["p_ri"].mean().round(3).values,
              "Observed frequency": h.groupby(rel, observed=True)["y"].mean().round(3).values}).to_csv(TAB / "ri_reliability.csv", index=False)

# ---------------------------------------------------------------- nine seasons of NHC-GEFS (scripts/ens_multiseason.py)
ms = pd.read_csv(R / "multiseason_nhc_gefs_table.csv", index_col=0)
ms_cols = {"storms": "Storms", "raw, offset to t0": "Raw GEFS", "CCSR": "Consensus centre", "GECO (HCCA centre, unsmoothed)": "GECO",
           "HCCA dressed": "HCCA dressed", "OFCL dressed": "OFCL dressed", "GECO − HCCA dressed": "GECO − HCCA dressed", "GECO coverage": "GECO coverage"}
mt = ms[list(ms_cols)].rename(columns=ms_cols)
for c in ("Raw GEFS", "Consensus centre", "GECO", "HCCA dressed", "OFCL dressed"):
    mt[c] = mt[c].map(lambda x: f"{x:.2f}")
mt["GECO coverage"] = mt["GECO coverage"].map(lambda x: f"{100 * x:.0f}%")
mt["GECO − HCCA dressed"] = mt["GECO − HCCA dressed"].str.replace(" *", "", regex=False)
mt.index.name = "Season"
mt.to_csv(TAB / "multiseason.csv")
for k in ("all", "2018–24", "2025–26"):
    tag = {"all": "all", "2018–24": "1824", "2025–26": "2526"}[k]
    for col, key in (("GECO − HCCA dressed", "hccad"), ("GECO − CCSH", "ccsh"), ("GECO − CCSR", "ccsr")):
        v = ms.loc[k, col].replace(" *", "").replace("[", "").replace("]", "").replace(",", "").split()
        numbers[f"ms_{tag}_{key}"] = [float(x) for x in v]
    numbers[f"ms_{tag}_geco_crps"], numbers[f"ms_{tag}_cov"] = round(float(ms.loc[k, "GECO (HCCA centre, unsmoothed)"]), 2), round(float(ms.loc[k, "GECO coverage"]), 2)
numbers["ms_storms"], numbers["ms_cases"] = int(ms.loc["all", "storms"]), int(ms.loc["all", "cases"])

(OUT / "numbers_ens.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
