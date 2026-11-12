"""Figures, tables and numbers for the GECO / IECO write-up (docs/report/report.typ): architecture diagram, scores by season, 2025–26
against FNV3 and dressed HCCA, CRPS and spread by lead, physical consistency, spread-model ranking. Reads saved results only
(scripts/ens_multiseason.py, scripts/ens_ieco.py, scripts/ens_calib.py):  nice -n 19 .venv/bin/python scripts/report_figures_ens.py
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tcic.config import DATA_DIR, INTERIM_DIR, PRODUCTS, ROOT, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import calib, geco
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
OUT = ROOT / "docs" / "report"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
(OUT / "preview").mkdir(parents=True, exist_ok=True)
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
BLUE, AQUA, VIOLET, ORANGE = "#2a78d6", "#1baf7a", "#6250d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOR = {"GECO": BLUE, "IECO": AQUA, "FNV3 early": VIOLET, "HCCA dressed": ORANGE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": MUTED,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
numbers = {}
ci3 = lambda d: [round(x, 2) for x in d]

# ---------------------------------------------------------------- architecture diagram
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.set_xlim(-1.5, 101.5)  # room for the rounded corners of the outermost boxes
ax.set_ylim(0, 74)
ax.axis("off")


def box(x, y, w, h, title, body, edge=MUTED, fill="#f6f5f2", tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge, lw=1.0, clip_on=False))
    ax.text(x + 1.5, y + h - 1.8, title, fontsize=7.4, weight="bold", color=tcol, va="top")
    ax.text(x + 1.5, y + h - 5.6, body, fontsize=6.4, color=INK2, va="top", linespacing=1.4)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0))


ax.text(0, 73.5, "Public real-time inputs (newest available by t0 + 3 h)", fontsize=7.4, color=INK2, weight="bold", va="top")
box(0, 52, 28, 18, "Member tracks", "GEFS: 30 members, NHC's\ntracker (NHC a-decks)\nIFS-ENS: 50 members,\nECMWF's tracker (AWS)\nnewest run ≥ 6 h old")
box(0, 30, 28, 18, "HCCA", "NHC's corrected consensus,\nin the public a-decks at\nevery cycle (draws on\nECMWF guidance NHC has\ninternally)")
box(0, 1, 28, 25, "Storm state and early aids", "CARQ at t0: intensity,\ntrend, position, motion,\ndistance to land\n\n12 interpolated aids\n(global, regional and\nstatistical models)")
box(36, 52, 30, 18, "1 · Shape  zᵢ(τ)", "member's departure from the\nensemble mean, re-anchored\nto 0 at t0, smoothed (6 h),\nnormalised to unit spread", edge=AQUA, tcol="#138a60")
box(36, 30, 30, 18, "2 · Centre", "HCCA as issued; between its\nleads interpolated, past its\nlast lead the HCCA − TCCA\noffset carried forward", edge=ORANGE, tcol=ORANGE)
box(36, 1, 30, 25, "3 · Size  σ̂(τ)", "LightGBM spread model:\nexpected error of a consensus\nforecast for this storm, cycle\nand lead (trained on 2018–24\nout-of-fold errors of our\nconsensus correction, TCCA)\nsmoothed along lead (18 h)", edge=BLUE, tcol=BLUE)
box(73, 30, 27, 40, "Members", "memberᵢ(τ) =\n   HCCA(τ)\n   + k · σ̂(τ) · zᵢ(τ)\n\nk = 1.2, fitted on\nout-of-fold CRPS\n\nGECO: 30 GEFS members\nIECO: 50 IFS-ENS members\nevery cycle, mean = HCCA", edge=VIOLET, tcol=VIOLET)
box(73, 1, 27, 25, "Outputs", "• calibrated 10–90% range\n  and quantiles\n• member traces that start\n  at the observed intensity\n  and fan out storm-\n  specifically\n• ensemble mean = HCCA", edge=INK2)
arrow(28.6, 61, 35.4, 61)
arrow(28.6, 39, 35.4, 39)
arrow(28.6, 13, 35.4, 13)
arrow(28.6, 56, 35.4, 22)
arrow(66.6, 61, 72.4, 58, AQUA)
arrow(66.6, 39, 72.4, 46, ORANGE)
arrow(66.6, 13, 72.4, 36, BLUE)
arrow(86.5, 29.4, 86.5, 26.6, VIOLET)
fig.savefig(FIG / "fig_architecture.svg")
fig.savefig(OUT / "preview" / "fig_architecture.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- scores by season (each product on its own nine-season sample)
ms = pd.read_csv(R / "multiseason_nhc_gefs_table.csv", index_col=0)
ie = pd.read_csv(R / "ieco_multiseason_table.csv", index_col=0)
strip = lambda s: s.replace(" *", "")
rows = []
for lab in [str(s) for s in range(2018, 2027)] + ["2018–24", "2025–26"]:
    g, i = ms.loc[lab if lab in ms.index else lab], ie.loc[lab if lab != "all" else "all 2018–26"]
    rows.append({"Season": lab, "GECO storms": int(g["storms"]), "GECO": f"{g['GECO (HCCA centre, unsmoothed)']:.2f}",
                 "HCCA dr. (G)": f"{g['HCCA dressed']:.2f}", "GECO − HCCA dr.": strip(g["GECO − HCCA dressed"]),
                 "IECO storms": int(i["storms"]), "IECO": f"{i['CRPS IECO']:.2f}", "HCCA dr. (I)": f"{i['CRPS HCCA dressed']:.2f}",
                 "IECO − HCCA dr.": strip(i["IECO − HCCA dressed"]) if i["storms"] > 1 else f"{i['CRPS IECO'] - i['CRPS HCCA dressed']:+.2f} (1 storm)"})
g, i = ms.loc["all"], ie.loc["all 2018–26"]
rows.append({"Season": "All", "GECO storms": int(g["storms"]), "GECO": f"{g['GECO (HCCA centre, unsmoothed)']:.2f}", "HCCA dr. (G)": f"{g['HCCA dressed']:.2f}",
             "GECO − HCCA dr.": strip(g["GECO − HCCA dressed"]), "IECO storms": int(i["storms"]), "IECO": f"{i['CRPS IECO']:.2f}",
             "HCCA dr. (I)": f"{i['CRPS HCCA dressed']:.2f}", "IECO − HCCA dr.": strip(i["IECO − HCCA dressed"])})
pd.DataFrame(rows).to_csv(TAB / "seasons.csv", index=False)
parse = lambda v: [float(x) for x in v.replace(" *", "").replace("[", "").replace("]", "").replace(",", "").split()]
for lab, tag in (("all", "all"), ("2018–24", "1824"), ("2025–26", "2526")):
    numbers[f"geco_{tag}_hccad"] = parse(ms.loc[lab, "GECO − HCCA dressed"])
    numbers[f"geco_{tag}_cov"] = round(float(ms.loc[lab, "GECO coverage"]), 2)
    il = "all 2018–26" if lab == "all" else lab
    numbers[f"ieco_{tag}_hccad"] = parse(ie.loc[il, "IECO − HCCA dressed"])
    numbers[f"ieco_{tag}_vs_geco"] = parse(ie.loc[il, "IECO − GECO (reference)"])
    numbers[f"ieco_{tag}_cov"] = round(float(ie.loc[il, "IECO coverage"]), 2)
    numbers[f"ieco_{tag}_spread_err"] = round(float(ie.loc[il, "IECO spread/error"]), 2)
numbers["geco_storms"], numbers["geco_cases"] = int(ms.loc["all", "storms"]), int(ms.loc["all", "cases"])
numbers["ieco_storms"], numbers["ieco_cases"] = int(ie.loc["all 2018–26", "storms"]), int(ie.loc["all 2018–26", "cases"])
numbers["ieco_6h_share"] = round(float(ie.loc["all 2018–26", "6-h-old share"]), 2)

# ---------------------------------------------------------------- per-case scores: GECO, IECO, raw members, FNV3 early, dressed HCCA
P = geco.products()
S = {n: sc.ensemble_case_scores(P[n], n) for n in PRODUCTS}
S["Raw GEFS"], S["Raw IFS-ENS"] = sc.ensemble_case_scores(P["GECO"], "RAW"), sc.ensemble_case_scores(P["IECO"], "RAW")
obs = pd.concat([P[n].drop_duplicates(CASE)[CASE + ["obs_vmax"]] for n in PRODUCTS]).drop_duplicates(CASE)
hp = obs.merge(geco.hcca(), on=CASE)
S["HCCA dressed"] = geco.dressed(hp.assign(obs=hp["obs_vmax"], mean=hp["hcca"]))
fn = fnv3_data.load_test()
c = obs[obs["t0"].dt.year.isin(TEST_SEASONS)][KEY].drop_duplicates().merge(fn[["storm_id", "init"]].drop_duplicates(), on="storm_id")
c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))].sort_values("init").groupby(KEY).tail(1)
f = c.merge(fn, on=["storm_id", "init"])
f["lead"] = f["lead"] - ((f["t0"] - f["init"]) / pd.Timedelta(hours=1)).astype(int)
f = f[f["lead"].isin(LEADS)].merge(obs, on=CASE)
S["FNV3 early"] = sc.ensemble_case_scores(f, "vmax")


def common(names, years=None):
    keys = None
    for k in names:
        s = S[k] if years is None else S[k][S[k]["t0"].dt.year.isin(years)]
        keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
    return {k: S[k].merge(keys, on=CASE) for k in names}


# 2025–26, all cycles, identical cases for every system
names = ["GECO", "IECO", "HCCA dressed", "FNV3 early", "Raw GEFS", "Raw IFS-ENS"]
H = common(names, TEST_SEASONS)
label = {"GECO": "GECO (GEFS)", "IECO": "IECO (IFS-ENS)", "HCCA dressed": "HCCA dressed", "FNV3 early": "FNV3 members (early)",
         "Raw GEFS": "Raw GEFS members", "Raw IFS-ENS": "Raw IFS-ENS members"}
rows = []
for k in names:
    h = H[k]
    err = h["mean"] - h["obs"]
    row = {"System": label[k], "CRPS (kt)": f"{h['crps'].mean():.2f}", "Bias of mean (kt)": f"{err.mean():.1f}",
           "10–90% coverage": f"{100 * h['inside'].mean():.0f}%", "Spread / error": f"{h['std'].mean() / np.sqrt((err ** 2).mean()):.2f}"}
    for ref, col in (("HCCA dressed", "− HCCA dressed"), ("FNV3 early", "− FNV3")):
        if k in PRODUCTS:
            d, lo, hi = sc.storm_bootstrap_diff(h, H[ref], "crps")
            row[col] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            numbers[f"{k.lower()}_2526_vs_{'hccad' if ref == 'HCCA dressed' else 'fnv3'}"] = ci3((d, lo, hi))
            for yr in TEST_SEASONS:
                hy = lambda x: x[x["t0"].dt.year == yr]
                numbers[f"{k.lower()}_{yr}_vs_{'hccad' if ref == 'HCCA dressed' else 'fnv3'}"] = ci3(sc.storm_bootstrap_diff(hy(h), hy(H[ref]), "crps"))
        else:
            row[col] = ""
    rows.append(row)
    numbers[f"crps_2526_{k.replace(' ', '_').lower()}"] = round(float(h["crps"].mean()), 2)
    numbers[f"bias_2526_{k.replace(' ', '_').lower()}"] = round(float(err.mean()), 1)
pd.DataFrame(rows).to_csv(TAB / "ensemble_summary.csv", index=False)
numbers["fnv3_storms"], numbers["fnv3_cases"] = int(H["GECO"]["storm_id"].nunique()), int(len(H["GECO"]))
d = sc.storm_bootstrap_diff(H["IECO"], H["GECO"], "crps")
numbers["ieco_vs_geco_2526_common"] = ci3(d)

# CRPS by lead: nine seasons (GECO, IECO, dressed HCCA on common cases) and 2025–26 with FNV3
A = common(["GECO", "IECO", "HCCA dressed"])
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, (HH, title) in zip(axes, ((A, "2018–26, all cycles"), (H, "2025–26, all cycles"))):
    for k in ("HCCA dressed", "FNV3 early", "GECO", "IECO"):
        if k in HH:
            v = HH[k].groupby("lead")["crps"].mean()
            ax.plot(v.index, v.values, color=COLOR[k], lw=2 if k in PRODUCTS else 1.5, ls="-" if k in PRODUCTS else "--",
                    marker="o", ms=3, mec="white", mew=0.5, label=label[k])
    ax.set_title(f"{title} ({HH['GECO']['storm_id'].nunique()} storms)", loc="left", color=INK)
    ax.set_xlabel("Lead time (h)")
    ax.set_xticks(range(12, 121, 24))
axes[0].set_ylabel("Fair CRPS (kt)")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=7)
fig.subplots_adjust(wspace=0.08)
fig.savefig(FIG / "fig_crps_by_lead.svg")
fig.savefig(OUT / "preview" / "fig_crps_by_lead.png", dpi=110)
plt.close(fig)
numbers["all_common_storms"], numbers["all_common_cases"] = int(A["GECO"]["storm_id"].nunique()), int(len(A["GECO"]))

# spread vs error by lead, 2025–26 (same cases as the table)
SHORT = {"GECO": "GECO", "IECO": "IECO", "FNV3 early": "FNV3", "Raw GEFS": "Raw", "Raw IFS-ENS": "Raw"}
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, (prod, raw) in zip(axes, (("GECO", "Raw GEFS"), ("IECO", "Raw IFS-ENS"))):
    for k, col in ((prod, COLOR[prod]), ("FNV3 early", VIOLET), (raw, MUTED)):
        g = H[k].groupby("lead")
        ax.plot(g["std"].mean().index, g["std"].mean().values, color=col, lw=2, marker="o", ms=3, mec="white", mew=0.5, label=f"{SHORT[k]}: spread")
        if not k.startswith("Raw"):
            rm = g.apply(lambda x: np.sqrt(((x["mean"] - x["obs"]) ** 2).mean()))
            ax.plot(rm.index, rm.values, color=col, lw=1.4, ls="--", label=f"{SHORT[k]}: error of mean")
    ax.set_title(f"{label[prod]}, 2025–26", loc="left", color=INK)
    ax.set_xlabel("Lead time (h)")
    ax.set_xticks(range(12, 121, 24))
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
axes[0].set_ylabel("kt")
fig.subplots_adjust(wspace=0.15)
fig.savefig(FIG / "fig_spread_vs_error.svg")
plt.close(fig)

# ---------------------------------------------------------------- physical consistency, 2025–26, leads 0–120 h
rough = {}
for n in PRODUCTS:
    t = P[n][P[n]["season"].isin(TEST_SEASONS)]
    base = t.drop_duplicates(KEY + ["member"])[KEY + ["member", "vmax0", "v_t0"]]
    for col, start, nm in (("RAW", "v_t0", f"Raw {PRODUCTS[n][0]} members"), (n, "vmax0", f"{n} ({PRODUCTS[n][0]})")):
        tr = pd.concat([t[KEY + ["member", "lead", col]], base.assign(lead=0).rename(columns={start: col})[KEY + ["member", "lead", col]]])
        rough[nm] = smo.roughness(tr, col)
fn2 = fn[(fn["storm_id"].str[2:4].astype(int) < 50) & (fn["lead"] <= 120)].rename(columns={"init": "t0"})
rough["FNV3 members"] = smo.roughness(fn2, "vmax")
bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet").assign(storm_id=lambda x: x["storm_id"].astype(str))
bt = bt[bt["time"].dt.year.isin(TEST_SEASONS)]
st = fn2[["storm_id", "t0"]].drop_duplicates().merge(bt[["storm_id", "time", "vmax"]], on="storm_id")
st = st[(st["time"] >= st["t0"]) & (st["time"] <= st["t0"] + pd.Timedelta(hours=120))]
st["lead"], st["member"] = ((st["time"] - st["t0"]) / pd.Timedelta(hours=1)).astype(int), 0
rough["Best track"] = smo.roughness(st, "vmax")
numbers["zigzag_6h"] = {"Best track": round(smo.roughness(st, "vmax", 6, 7)["flip rate"], 3), "FNV3 members": round(smo.roughness(fn2, "vmax", 6, 7)["flip rate"], 3)}
order = ["Best track", "FNV3 members", "Raw GEFS members", "Raw IFS-ENS members", "GECO (GEFS)", "IECO (IFS-ENS)"]
pd.DataFrame([{"Traces": k, "Zig-zag rate": f"{rough[k]['flip rate']:.3f}", "Mean |12-h change| (kt)": f"{rough[k]['mean |Δ|']:.1f}",
               "Share of 12-h changes > 30 kt": f"{100 * rough[k]['share |Δ| > 30 kt']:.1f}%"} for k in order]).to_csv(TAB / "roughness.csv", index=False)
numbers["rough"] = {k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in rough.items()}

# ---------------------------------------------------------------- spread model ranking
sp = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
te_sp = sp[sp["season"].isin(TEST_SEASONS)]
numbers["spread_corr"] = round(float(te_sp["sigma_hat"].corr(te_sp["abs_err"])), 2)
numbers["spread_corr_within_lead"] = round(float(te_sp.groupby("lead").apply(lambda g: g["sigma_hat"].corr(g["abs_err"])).mean()), 2)
b = te_sp.assign(bin=pd.qcut(te_sp["sigma_hat"], 5, labels=["Calmest fifth", "Second", "Middle", "Fourth", "Most volatile fifth"]))
b.groupby("bin", observed=True).agg(**{"Cases": ("abs_err", "size"), "Predicted error (kt)": ("sigma_hat", "mean"), "Actual error (kt)": ("abs_err", "mean")}).round(1).reset_index().rename(columns={"bin": "Cases binned by predicted error"}).to_csv(TAB / "spread_bins.csv", index=False)

# ---------------------------------------------------------------- spread-model inputs (every feature, from the model's own list)
AIDS = {"AVNI": "GFS", "AEMI": "GEFS mean", "EGRI": "UKMET", "CMCI": "Canadian (CMC)", "NVGI": "Navy (NAVGEM)", "HWFI": "HWRF",
        "HMNI": "HMON", "HFAI": "HAFS-A", "HFBI": "HAFS-B", "CTCI": "COAMPS-TC", "DSHP": "SHIPS with decay over land", "LGEM": "LGEM"}
DESC = {
    "vmax0": ("Storm state (CARQ, t0)", "maximum wind at t0 (kt)"), "pmin0": ("Storm state (CARQ, t0)", "minimum pressure at t0 (hPa)"),
    "lat0": ("Storm state (CARQ, t0)", "latitude at t0"), "lon0": ("Storm state (CARQ, t0)", "longitude at t0"),
    "dv6": ("Storm state (CARQ, t0)", "intensity change over the previous 6 h"), "dv12": ("Storm state (CARQ, t0)", "intensity change over the previous 12 h"),
    "dv24": ("Storm state (CARQ, t0)", "intensity change over the previous 24 h"),
    "u_motion": ("Storm state (CARQ, t0)", "eastward motion over the previous 12 h (km/h)"), "v_motion": ("Storm state (CARQ, t0)", "northward motion over the previous 12 h (km/h)"),
    "dist2land0": ("Storm state (CARQ, t0)", "distance to land at t0 (km)"), "basin_ep": ("Storm state (CARQ, t0)", "East Pacific (1) or Atlantic (0)"),
    "doy": ("Storm state (CARQ, t0)", "day of the year of t0"), "lead": ("Lead time", "lead time τ (h)"),
    "n_aids": ("Early-aid consensus", "number of early aids available at this lead"),
    "ref_vmax": ("Early-aid consensus", "equal-weight mean intensity of the available early aids (the reference)"),
    "ref_pmin": ("Early-aid consensus", "reference pressure: CARQ pressure plus the GEFS members' mean pressure change"),
    "aid_vmax_std": ("Early-aid consensus", "standard deviation of the early aids' intensities"), "aid_vmax_range": ("Early-aid consensus", "range (max − min) of the early aids' intensities"),
    "lat_c": ("Early-aid consensus", "mean forecast latitude of the early aids"), "lon_c": ("Early-aid consensus", "mean forecast longitude of the early aids"),
    "dist2land_c": ("Early-aid consensus", "distance to land at the mean forecast position (km)"),
    "gefs_vmax_mean": ("GEFS members (NHC tracks)", "mean member intensity at this lead"), "gefs_vmax_std": ("GEFS members (NHC tracks)", "standard deviation of member intensities"),
    "gefs_vmax_p10": ("GEFS members (NHC tracks)", "10th percentile of member intensities"), "gefs_vmax_p90": ("GEFS members (NHC tracks)", "90th percentile of member intensities"),
    "gefs_n": ("GEFS members (NHC tracks)", "number of members tracked at this lead"), "gefs_mslp_mean": ("GEFS members (NHC tracks)", "mean member minimum pressure at this lead"),
    "gefs_vmax0_mean": ("GEFS members (NHC tracks)", "mean member intensity valid at t0"), "gefs_mslp0_mean": ("GEFS members (NHC tracks)", "mean member pressure valid at t0"),
    "gefs_ri_frac": ("GEFS members (NHC tracks)", "share of members rising ≥ 30 kt in the 24 h ending at this lead"),
    "gefs_age": ("GEFS members (NHC tracks)", "age of the GEFS run at t0 (6 or 12 h)"), "gefs_offset0": ("GEFS members (NHC tracks)", "mean member intensity at t0 minus CARQ intensity"),
    "tcca_mean": ("Consensus correction (TCCA)", "corrected mean intensity at this lead (out of fold)"),
    "tcca_qwidth": ("Consensus correction (TCCA)", "width of its 10–90% quantile range"),
    "tcca_dv24": ("Consensus correction (TCCA)", "its predicted intensity change over the 24 h ending at this lead"),
    "tcca_minus_v0": ("Consensus correction (TCCA)", "its predicted intensity minus the CARQ intensity at t0"),
}
for a, nm in AIDS.items():
    DESC[f"{a}_vmax"] = ("Early aids (interpolated)", f"{a} ({nm}): intensity at this lead")
    DESC[f"{a}_dv12"] = ("Early aids (interpolated)", f"{a} ({nm}): intensity change over the preceding 12 h")
_, feats = calib.table()
missing = [f for f in feats if f not in DESC]
assert not missing, f"spread-model features without a description: {missing}"
order = list(dict.fromkeys(g for g, _ in DESC.values()))
ft = pd.DataFrame([{"Group": DESC[f][0], "Variable": f, "Description": DESC[f][1]} for f in feats if not f.startswith(tuple(AIDS))])
ft = pd.concat([ft, pd.DataFrame([{"Group": "Early aids (interpolated)", "Variable": f"{a}_vmax, {a}_dv12",
                                   "Description": f"{nm}: intensity at this lead; its change over the preceding 12 h"}
                                  for a, nm in AIDS.items() if f"{a}_vmax" in feats and f"{a}_dv12" in feats])], ignore_index=True)
assert len(ft) + sum(1 for r in ft["Variable"] if "," in r) == len(feats)  # every feature appears exactly once
ft = ft.assign(o=ft["Group"].map(order.index)).sort_values("o", kind="stable").drop(columns="o")
ft.to_csv(TAB / "spread_features.csv", index=False)
numbers["spread_n_features"] = len(feats)

# ---------------------------------------------------------------- choice of k (scripts/k_curve.py)
kc = pd.read_csv(R / "k_curve.csv")
kt = kc.pivot(index="k", columns="product", values=["CRPS", "coverage", "spread/error"])
rows_k = []
for k, r in kt.iterrows():
    row = {"k": f"{k:.2f}"}
    for prod in PRODUCTS:
        row[f"{prod} CRPS"] = f"{r[('CRPS', prod)]:.3f}"
        row[f"{prod} coverage"] = f"{100 * r[('coverage', prod)]:.0f}%"
        row[f"{prod} spread/error"] = f"{r[('spread/error', prod)]:.2f}"
    rows_k.append(row)
pd.DataFrame(rows_k).to_csv(TAB / "k_curve.csv", index=False)
tr = sp[sp["season"].isin(calib.TRAIN)]
numbers["rms_over_mae"] = round(float(np.sqrt((tr["abs_err"] ** 2).mean()) / tr["abs_err"].mean()), 2)
numbers["k_cases"] = {prod: int(kc[kc["product"] == prod]["cases"].iloc[0]) for prod in PRODUCTS}

(OUT / "numbers_ens.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
