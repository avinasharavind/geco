"""Figures and key numbers for the project write-up (docs/report/). Recomputes everything from saved results.

Run: nice -n 19 .venv/bin/python scripts/report_figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tcic import evaluate as ev
from tcic.config import FEATURES_DIR, INTERIM_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.consensus import calibrate
from tcic.consensus import model as m
from tcic.data.atcf import merge_interpolated
from tcic.pilot import cases as pcases

OUT = ROOT / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
(OUT.parent / "preview").mkdir(exist_ok=True)
RES = ROOT / "data" / "results" / "m15"

# Reference palette (dataviz skill, light mode), fixed slot order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = {"Ours (TCCA)": BLUE, "HCCA": ORANGE, "OFCL": AQUA, "Uncorrected mean": MUTED}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
numbers = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT.parent / "preview" / f"{name}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- shared inputs
F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet")
A = pd.read_parquet(INTERIM_DIR / "adecks.parquet")
A["storm_id"] = A["storm_id"].astype(str)
A = merge_interpolated(A)
truth = ev.load_truth("nhc")
NHC_LEADS = list(ev.NHC_LEADS)

# ---------------------------------------------------------------- Fig 1: M1 reproduction of NHC Table 5a
NHC_T5A = {
    "OFCL": [5.4, 8.0, 9.9, 11.4, 12.0, 13.3, 13.8, 12.8], "OCD5": [8.3, 13.4, 18.4, 22.7, 23.8, 25.4, 28.1, 19.2],
    "HWFI": [7.5, 9.7, 11.4, 12.3, 12.8, 13.2, 15.8, 17.7], "HMNI": [7.2, 9.1, 11.1, 12.1, 12.5, 13.0, 14.9, 15.6],
    "HFAI": [7.4, 10.2, 12.4, 13.9, 14.8, 16.3, 18.6, 17.5], "HFBI": [7.3, 10.1, 11.4, 14.3, 15.8, 16.8, 16.2, 18.8],
    "DSHP": [7.2, 10.3, 13.0, 14.0, 14.8, 16.3, 15.5, 15.3], "LGEM": [7.5, 10.5, 13.1, 14.0, 13.6, 14.3, 14.6, 14.1],
    "IVCN": [6.3, 8.4, 10.1, 11.5, 12.0, 12.6, 13.3, 12.2], "HCCA": [6.0, 7.8, 9.6, 11.3, 12.1, 12.9, 14.4, 14.8],
    "AVNI": [7.6, 11.0, 13.7, 15.6, 16.8, 17.8, 18.7, 17.6], "EMXI": [8.6, 12.8, 16.3, 18.8, 20.3, 21.3, 21.5, 21.6]}
al24 = A[A["storm_id"].str.startswith("AL") & (A["init"].dt.year == 2024)]
techs = list(NHC_T5A)
c = ev.homogeneous(ev.pair(al24, truth, techs), techs)
t = ev.by_lead(c, techs)["mae"].unstack("tau")[NHC_LEADS]
ours = t.loc[techs].to_numpy().ravel()
theirs = np.array([NHC_T5A[k] for k in techs]).ravel()
numbers["m1_t5a_max_abs_diff"] = float(np.abs(ours - theirs).max())
numbers["m1_t5a_n_within_0p2"] = int((np.abs(ours - theirs) <= 0.2 + 1e-9).sum())
numbers["m1_t5a_n"] = int(ours.size)
fig, ax = plt.subplots(figsize=(3.2, 3.0))
lim = (4, 30)
ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=1)
ax.scatter(theirs, ours, s=22, color=BLUE, edgecolor="white", linewidth=0.8, zorder=2)
ax.set(xlim=lim, ylim=lim, xlabel="NHC published MAE (kt)", ylabel="Our recomputed MAE (kt)")
ax.set_title("12 early aids × 8 leads, Atlantic 2024", loc="left", color=INK)
ax.text(5, 27.5, f"{numbers['m1_t5a_n_within_0p2']}/{ours.size} within ±0.2 kt\nmax |diff| {numbers['m1_t5a_max_abs_diff']:.1f} kt", color=INK2, fontsize=8, va="top")
save(fig, "fig_m1_reproduction")

# ---------------------------------------------------------------- M1.5 cases (CV and test)
def ours_cases(pred, tech):
    return m.as_cases(pred, F, tech)


def ref_cases(pred):
    ref = pred[m.ID_COLS + ["lead"]].copy()
    ref["mean"] = F.set_index(["storm_id", "t0", "lead"]).loc[pd.MultiIndex.from_frame(pred[["storm_id", "t0", "lead"]]), "ref_vmax"].to_numpy()
    return m.as_cases(ref, F, "TCRF")


oofA, oofB = (pd.read_parquet(RES / f"oof_vmax_{v}.parquet") for v in "AB")
testA, testB = (pd.read_parquet(RES / f"test_vmax_{v}.parquet") for v in "AB")
base_cv = ev.pair(A[A["init"].dt.year.between(2018, 2024)], truth, ["OFCL", "IVCN", "HCCA"], leads=list(range(12, 169, 12)))
base_te = ev.pair(A[A["init"].dt.year.isin(TEST_SEASONS)], truth, ["OFCL", "IVCN", "HCCA", "GDMI"], leads=list(range(12, 169, 12)))
cv = pd.concat([ours_cases(oofA, "TCCA"), ours_cases(oofB, "TCCB"), ref_cases(oofA), base_cv])
te = pd.concat([ours_cases(testA, "TCCA"), ours_cases(testB, "TCCB"), ref_cases(testA), base_te])
cmp = ["TCCA", "TCCB", "TCRF", "OFCL", "IVCN", "HCCA"]
cvh = ev.homogeneous(cv[cv["tau"].isin(NHC_LEADS)], cmp)
teh = ev.homogeneous(te[te["tau"].isin(NHC_LEADS)], cmp)
for name, cc in (("cv", cvh), ("test", teh)):
    o = ev.overall(cc, cmp)
    numbers[f"m15_{name}_overall"] = o[["mae", "mae_lo", "mae_hi", "bias"]].round(2).to_dict(orient="index")
    numbers[f"m15_{name}_storms"] = int(cc["storm_id"].nunique())
    numbers[f"m15_{name}_forecasts"] = int(cc[["storm_id", "init"]].drop_duplicates().shape[0])
    for a, b in (("TCCA", "HCCA"), ("TCCA", "IVCN"), ("TCCB", "TCCA")):
        numbers[f"m15_{name}_diff_{a}_{b}"] = [round(x, 2) for x in ev.mae_difference(cc, a, b)]
    ri = cc[cc["ri"]]
    numbers[f"m15_{name}_ri"] = ev.overall(ri, cmp)[["mae", "mae_lo", "mae_hi", "bias"]].round(2).to_dict(orient="index")
    numbers[f"m15_{name}_ri_storms"] = int(ri["storm_id"].nunique())
    for b in ("HCCA", "OFCL"):
        numbers[f"m15_{name}_ri_diff_TCCA_{b}"] = [round(x, 2) for x in ev.mae_difference(ri, "TCCA", b)]
for s in TEST_SEASONS:  # each test season on its own
    cs = teh[teh["init"].dt.year == s]
    numbers[f"m15_test{s}_overall"] = ev.overall(cs, cmp)[["mae", "mae_lo", "mae_hi", "bias"]].round(2).to_dict(orient="index")
    for a, b in (("TCCA", "HCCA"), ("TCCA", "IVCN")):
        numbers[f"m15_test{s}_diff_{a}_{b}"] = [round(x, 2) for x in ev.mae_difference(cs, a, b)]

# Fig 2: MAE by lead, CV and 2025 test (small multiples, shared y)
label = {"TCCA": "Ours (TCCA)", "HCCA": "HCCA", "OFCL": "OFCL", "TCRF": "Uncorrected mean"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for ax, (title, cc) in zip(axes, (("Cross-validation 2018–2024", cvh), (f"Held-out {TEST_LABEL}", teh))):
    tt = ev.by_lead(cc, list(label))["mae"].unstack("tau")[NHC_LEADS]
    for k in ["TCRF", "OFCL", "HCCA", "TCCA"]:
        col = SERIES[label[k]]
        ax.plot(NHC_LEADS, tt.loc[k], color=col, lw=2.4 if k == "TCCA" else 1.6, marker="o", ms=3.5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3 if k == "TCCA" else 2, label=label[k])
    ax.set_title(title, loc="left", color=INK)
    ax.set_xticks(NHC_LEADS)
    ax.set_xlabel("Lead time (h)")
    ax.set_xlim(6, 126)
axes[0].set_ylabel("Max-wind MAE (kt)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.subplots_adjust(wspace=0.15)
save(fig, "fig_m15_mae_by_lead")

# Fig 3: RI cases, MAE with storm-bootstrap CIs
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharex=True)
order = ["TCCA", "OFCL", "HCCA", "IVCN", "TCRF"]
names = {"TCCA": "Ours (TCCA)", "OFCL": "OFCL", "HCCA": "HCCA", "IVCN": "IVCN", "TCRF": "Uncorrected mean"}
for ax, key, title in ((axes[0], "cv", "Cross-validation 2018–24"), (axes[1], "test", f"Held-out {TEST_LABEL}")):
    d = numbers[f"m15_{key}_ri"]
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        col = BLUE if k == "TCCA" else MUTED
        ax.errorbar(d[k]["mae"], yi, xerr=[[d[k]["mae"] - d[k]["mae_lo"]], [d[k]["mae_hi"] - d[k]["mae"]]], fmt="o", color=col,
                    ms=6, mec="white", mew=0.8, elinewidth=1.6, capsize=0)
        ax.text(d[k]["mae_hi"] + 0.6, yi, f"{d[k]['mae']:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [names[k] for k in order])
    ax.set_title(f"{title} · {numbers[f'm15_{key}_ri_storms']} RI storms", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Max-wind MAE (kt), 95% CI")
axes[0].set_xlim(10, 42)
fig.subplots_adjust(wspace=0.55)
save(fig, "fig_m15_ri")

# Fig 4: interval coverage by lead, raw vs conformal
Fo = F[["storm_id", "t0", "lead", "obs_vmax"]]
oof = oofA.merge(Fo, on=["storm_id", "t0", "lead"])
tst = testA.merge(Fo, on=["storm_id", "t0", "lead"])
widen = calibrate.fit(oof, oof["obs_vmax"])
cal = calibrate.apply(tst, widen)
cov = lambda p: ((p.obs_vmax >= p.q10) & (p.obs_vmax <= p.q90)).groupby(p["lead"]).mean()
numbers["cal_widen_kt"] = widen.round(1).to_dict()
for nm, p in (("raw", tst), ("cal", cal)):
    q = p[p.lead <= 120]
    numbers[f"cal_{nm}_coverage_2025"] = round(float(((q.obs_vmax >= q.q10) & (q.obs_vmax <= q.q90)).mean()), 3)
    numbers[f"cal_{nm}_below_2025"] = round(float((q.obs_vmax < q.q10).mean()), 3)
    numbers[f"cal_{nm}_above_2025"] = round(float((q.obs_vmax > q.q90).mean()), 3)
    numbers[f"cal_{nm}_width_2025"] = round(float((q.q90 - q.q10).median()), 1)
qo = oof[oof.lead <= 120]
numbers["cal_raw_coverage_cv"] = round(float(((qo.obs_vmax >= qo.q10) & (qo.obs_vmax <= qo.q90)).mean()), 3)
fig, ax = plt.subplots(figsize=(4.2, 2.5))
ax.axhline(0.8, color=MUTED, lw=1, ls="--")
ax.text(12, 0.81, "nominal 80%", va="bottom", fontsize=7.5, color=INK2)
for p, col, nm in ((tst, MUTED, "raw quantiles"), (cal, BLUE, "conformal")):
    cc = cov(p)
    ax.plot(cc.index, cc.values, color=col, marker="o", ms=3.5, mec="white", mew=0.6)
    ax.annotate(nm, (cc.index[-1], cc.values[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
ax.set(xlabel="Lead time (h)", ylabel="Share inside 10–90% interval", ylim=(0.5, 1.0), xlim=(5, 200))
ax.set_xticks(range(12, 169, 24))
ax.set_title(f"Interval coverage on held-out {TEST_LABEL}", loc="left", color=INK)
save(fig, "fig_calibration")

# ---------------------------------------------------------------- Fig 5: Melissa, IFS-ENS pilot run 2025-10-26 00Z
tr = pd.read_parquet(ROOT / "data" / "pilot" / "ifs" / "2025102600" / "tracks.parquet")
tr = tr[tr["storm_id"] == "AL132025"]
bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet")
bt = bt[(bt["storm_id"].astype(str) == "AL132025")]
init = pd.Timestamp("2025-10-26")
bt = bt[(bt["time"] >= init) & (bt["time"] <= init + pd.Timedelta(hours=168))]
land = xr.open_dataarray(INTERIM_DIR / "dist2land_0p5.nc").sel(latitude=slice(42, 8), longitude=slice(-90, -40))
a_ = pcases._adecks_2024_on()
nhc = a_[(a_["storm_id"] == "AL132025") & (a_["init"] == init) & a_["tech"].str.match(r"^E[EC]\d\d$")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax1.contourf(land.longitude, land.latitude, (land.values < 0).astype(float), levels=[0.5, 1.5], colors=["#e9e8e3"])
ax1.contour(land.longitude, land.latitude, land.values, levels=[0], colors=[MUTED], linewidths=0.5)
for mem, g in tr.groupby("member"):
    ax1.plot(g["lon"], g["lat"], color=BLUE, lw=0.6, alpha=0.35)
ax1.plot(bt["lon"], bt["lat"], color=INK, lw=2, marker="o", ms=2.5, label="Best track")
ax1.set(xlim=(-86, -45), ylim=(12, 42), xlabel="Longitude", ylabel="Latitude")
ax1.set_aspect(1 / np.cos(np.radians(25)))
ax1.set_title("Tracks: 51 members (blue) vs best track", loc="left", color=INK)
ax1.grid(False)
leads_bt = ((bt["time"] - init) / pd.Timedelta(hours=1)).to_numpy()
for mem, g in tr.groupby("member"):
    ax2.plot(g["lead"], g["vmax"], color=BLUE, lw=0.6, alpha=0.3)
ax2.plot(tr.groupby("lead")["vmax"].mean(), color=BLUE, lw=2.2, label="Member mean (ours)")
nm_ = nhc.groupby("tau")["vmax"].mean()
nm_ = nm_[(nm_.index >= 0) & (nm_.index <= 168)]
ax2.plot(nm_.index, nm_.values, color=ORANGE, lw=1.6, ls="--", label="NHC tracker, same members")
ax2.plot(leads_bt, bt["vmax"], color=INK, lw=2, marker="o", ms=2.5, label="Best track")
ax2.set(xlabel="Lead time (h)", ylabel="Max wind (kt)", xlim=(0, 168), ylim=(0, 170))
ax2.set_xticks(range(0, 169, 24))
ax2.set_title("Intensity", loc="left", color=INK)
ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.subplots_adjust(wspace=0.3)
save(fig, "fig_melissa")

# ---------------------------------------------------------------- Fig A1: early-aid coverage by season (appendix)
aids = ["IVCN", "HCCA", "DSHP", "LGEM", "AVNI", "AEMI", "HWFI", "HFAI", "HFBI", "CTCI", "EMXI", "EGRI", "CMCI", "GDMI"]
cov_rows = {}
for y in range(2018, 2026):
    ay = A[A["init"].dt.year == y]
    k = ay.groupby("tech")["init"].nunique()
    carq = k.get("CARQ", 1)
    cov_rows[y] = [k.get(t, 0) / carq for t in aids]
cm = pd.DataFrame(cov_rows, index=aids)
from matplotlib.colors import LinearSegmentedColormap

seq = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(5.0, 3.6))
im = ax.imshow(cm.values.clip(0, 1), cmap=seq, vmin=0, vmax=1, aspect="auto")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = min(cm.values[i, j], 1)
        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.8, color="white" if v > 0.6 else INK2)
ax.set_xticks(range(cm.shape[1]), cm.columns)
ax.set_yticks(range(cm.shape[0]), cm.index)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Share of forecast cycles with each early aid (%)", loc="left", color=INK)
save(fig, "fig_aid_coverage")

# ---------------------------------------------------------------- Fig A2: pilot tracker vs NHC tracker (all finished units)
from tcic.pilot.extract import PILOT_DIR
from tcic.pilot.grid import haversine_km

val = {}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, (model, col, techs_) in zip(axes, (("ifs", BLUE, {0: "EC00", **{i: f"EE{i:02d}" for i in range(1, 51)}}),
                                          ("gefs", ORANGE, {0: "AC00", **{i: f"AP{i:02d}" for i in range(1, 31)}}))):
    files = sorted((PILOT_DIR / model).glob("*/done.json"))
    if not files:
        continue
    ours_ = pd.concat([pd.read_parquet(f.parent / "tracks.parquet", columns=["storm_id", "init", "member", "lead", "lat", "lon", "vmax", "pmin"]) for f in files])
    ours_["tech"] = ours_["member"].map(techs_)
    aa = a_[a_["tech"].isin(techs_.values()) & a_["init"].isin(ours_["init"].unique())][["storm_id", "init", "tech", "tau", "lat", "lon", "vmax", "mslp"]]
    mm = ours_.merge(aa.rename(columns={"tau": "lead"}), on=["storm_id", "init", "tech", "lead"], suffixes=("", "_nhc")).dropna(subset=["lat", "lat_nhc", "vmax", "vmax_nhc"])
    mm["pos"] = haversine_km(mm["lat"], mm["lon"], mm["lat_nhc"], mm["lon_nhc"])
    val[model] = {"units": len(files), "pairs": int(len(mm)), "pos_med_km": round(float(mm["pos"].median()), 1),
                  "pos_p90_km": round(float(mm["pos"].quantile(0.9)), 1), "vmax_corr": round(float(mm["vmax"].corr(mm["vmax_nhc"])), 3),
                  "vmax_bias": round(float((mm["vmax"] - mm["vmax_nhc"]).mean()), 2),
                  "pmin_bias": round(float((mm["pmin"] - mm["mslp"]).mean()), 2), "pmin_corr": round(float(mm["pmin"].corr(mm["mslp"])), 3)}
    h = ax.hexbin(mm["vmax_nhc"], mm["vmax"], gridsize=40, extent=(10, 150, 10, 150), mincnt=1, bins="log", cmap=seq)
    ax.plot((10, 150), (10, 150), color=MUTED, lw=1, ls="--")
    ax.set(xlabel="NHC tracker max wind (kt)", ylabel="Our tracker max wind (kt)", xlim=(10, 150), ylim=(10, 150))
    ax.set_title(f"{'IFS-ENS' if model == 'ifs' else 'GEFS'}: r = {val[model]['vmax_corr']:.2f}, median position diff {val[model]['pos_med_km']:.0f} km", loc="left", color=INK, fontsize=8)
    ax.grid(False)
numbers["pilot_validation"] = val
fig.subplots_adjust(wspace=0.35)
save(fig, "fig_tracker_validation")

(OUT.parent / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
print(json.dumps(numbers, indent=1, default=str)[:6000])

# ---------------------------------------------------------------- CSV tables for the Typst document
TAB = OUT.parent / "tables"
TAB.mkdir(exist_ok=True)
names_all = {"TCCA": "Ours, no SHIPS (TCCA)", "TCCB": "Ours, + SHIPS (TCCB)", "HCCA": "HCCA", "IVCN": "IVCN", "OFCL": "OFCL",
             "TCRF": "Uncorrected aid mean"}
for key, cc in (("cv", cvh), ("test", teh)):
    o = ev.overall(cc, cmp).loc[list(names_all)]
    pd.DataFrame({"System": [names_all[k] for k in o.index], "MAE": o["mae"].round(1), "95% CI": [f"{a:.1f}–{b:.1f}" for a, b in zip(o["mae_lo"], o["mae_hi"])],
                  "Bias": o["bias"].round(1)}).to_csv(TAB / f"overall_{key}.csv", index=False)
    bl = ev.by_lead(cc, cmp)["mae"].unstack("tau")[NHC_LEADS].loc[list(names_all)].round(1)
    bl.index = [names_all[k] for k in bl.index]
    bl.reset_index(names="System").to_csv(TAB / f"bylead_{key}.csv", index=False)
    ri = numbers[f"m15_{key}_ri"]
    pd.DataFrame({"System": [names_all[k] for k in names_all], "MAE": [round(ri[k]["mae"], 1) for k in names_all],
                  "95% CI": [f"{ri[k]['mae_lo']:.1f}–{ri[k]['mae_hi']:.1f}" for k in names_all],
                  "Bias": [round(ri[k]["bias"], 1) for k in names_all]}).to_csv(TAB / f"ri_{key}.csv", index=False)
# 2025 vs FNV3 early to 168 h
fl = list(range(12, 169, 12))
f4 = ["TCCA", "TCCB", "GDMI", "HCCA"]
tf = ev.homogeneous(te[te["tau"].isin(fl)], f4)
o = ev.overall(tf, f4)
numbers["fnv3_overall"] = o[["mae", "mae_lo", "mae_hi", "bias"]].round(2).to_dict(orient="index")
numbers["fnv3_storms"] = int(tf["storm_id"].nunique())
numbers["fnv3_diff_TCCA_GDMI"] = [round(x, 2) for x in ev.mae_difference(tf, "TCCA", "GDMI")]
bl = ev.by_lead(tf, f4)["mae"].unstack("tau")[fl].loc[f4].round(1)
bl.index = ["Ours, no SHIPS (TCCA)", "Ours, + SHIPS (TCCB)", "FNV3 early (GDMI)", "HCCA"]
bl.reset_index(names="System").to_csv(TAB / "bylead_fnv3.csv", index=False)
# feature groups
feats = m.feature_columns(F, "B")
groups = {"CARQ / storm state": lambda c: c in ("vmax0", "pmin0", "lat0", "lon0", "dv6", "dv12", "dv24", "u_motion", "v_motion", "dist2land0"),
          "Interpolated aids (value, 12-h change)": lambda c: c.split("_")[0] in ("AVNI", "AEMI", "HWFI", "HFAI", "HFBI", "HMNI", "CTCI", "EMXI", "EMNI", "EGRI", "CMCI", "NVGI", "DSHP", "LGEM"),
          "GEFS members (AP01–30)": lambda c: c.startswith("gefs_"), "ECMWF members (EE01–50)": lambda c: c.startswith("ecens_"),
          "Consensus summary": lambda c: c in ("ref_vmax", "ref_pmin", "n_aids", "aid_vmax_std", "aid_vmax_range", "lat_c", "lon_c", "dist2land_c"),
          "Calendar / basin / lead": lambda c: c in ("doy", "basin_ep", "lead"), "SHIPS (variant B only)": lambda c: c.startswith("ships")}
rows = []
for gname, fn in groups.items():
    cols = [c for c in feats if fn(c)]
    rows.append({"Group": gname, "Count": len(cols), "Examples": ", ".join(cols[:6]) + (" …" if len(cols) > 6 else "")})
used = {c for g in groups.values() for c in feats if g(c)}
other = [c for c in feats if c not in used]
if other:
    rows.append({"Group": "Other", "Count": len(other), "Examples": ", ".join(other[:6])})
pd.DataFrame(rows).to_csv(TAB / "features.csv", index=False)
numbers["n_features_A"], numbers["n_features_B"] = len(m.feature_columns(F, "A")), len(feats)
v = F[F["verifiable"]]
numbers["m15_rows_verifiable"], numbers["m15_cycles"] = int(len(v)), int(v[["storm_id", "t0"]].drop_duplicates().shape[0])
(OUT.parent / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
