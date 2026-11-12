"""Intensity traces for selected cases: GECO (NHC's GEFS member tracks) and IECO (ECMWF's own ENS member tracks) around HCCA, against
raw FNV3 members (2025-26 only), the best track and OFCL. For eyeballing physical consistency and spread.
Output: docs/case_plots.pdf (+ PNG previews; the first ten 2025 cases also as SVG for the write-up).

Every panel is for the same cycle (valid-time axis; the advisory is issued 3 h after the cycle), as it would run in real time: GECO and
IECO from the newest member run at least 6 h old (the 06Z runs for a 12Z cycle), FNV3 from its newest run at least 6 h old. GECO / IECO
members start at the operational (CARQ) intensity at the cycle and are plotted at the verified leads (12-120 h); raw (grey) members start
at their own intensity. 2018-24 cases (no FNV3) are out of sample (members from scripts/ens_multiseason.py and scripts/ens_ieco.py);
IECO needs NHC's archived or ECMWF's BUFR tracks, so it is missing for some older cases.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tcic.config import INTERIM_DIR, PRODUCTS, ROOT
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import geco

CASES = [  # (storm, cycle, why)
    ("AL052025", "2025-08-14 12:00", "Atlantic rapid intensification (to 140 kt)"),
    ("AL132025", "2025-10-25 12:00", "rapid intensification to 165 kt, then landfall in Jamaica"),
    ("EP112025", "2025-09-01 12:00", "East Pacific steady intensification to Cat 4"),
    ("AL052025", "2025-08-19 12:00", "recurvature and extratropical transition (weakening)"),
    ("EP082025", "2025-08-06 12:00", "weak, long-lived tropical storm"),
    ("AL052025", "2025-08-15 12:00", "about 18 h before extreme RI (65 kt -> 140 kt by 16 Aug 18Z)"),
    ("AL082025", "2025-09-25 12:00", "rapid intensification to 140 kt, then rapid weakening"),
    ("AL072025", "2025-09-18 12:00", "steady Atlantic intensification to 120 kt"),
    ("AL132025", "2025-10-22 12:00", "slow early phase; rapid intensification 3-4 days later"),
    ("EP162025", "2025-10-05 12:00", "East Pacific intensification, then decay over cooler water"),
]
EXTRA = [  # case_plots.pdf only (not in the report)
    ("AL052025", "2025-08-12 12:00", "four days before extreme RI"),
    ("AL052025", "2025-08-13 12:00", "three days before extreme RI"),
    ("AL052025", "2025-08-16 12:00", "mid-RI (95 kt at 06Z, 130 kt at 12Z), peak 140 kt, then eyewall cycles and weakening"),
    ("AL052025", "2025-08-17 12:00", "after the peak: reintensification to 120 kt, then weakening"),
    ("AL072025", "2025-09-20 12:00", "two days before rapid intensification to 120 kt"),
    ("AL072025", "2025-09-22 12:00", "during rapid intensification, then fast weakening"),
    ("AL082025", "2025-09-26 12:00", "midway through rapid intensification (80 -> 140 kt)"),
    ("AL092025", "2025-09-28 12:00", "depression to 80 kt hurricane"),
    ("AL132025", "2025-10-23 12:00", "40 kt, five days before a 165 kt peak"),
    ("AL132025", "2025-10-24 12:00", "four days before the peak"),
    ("AL132025", "2025-10-27 12:00", "145 kt and still strengthening to 165 kt; landfall in Jamaica, then Cuba"),
    ("EP112025", "2025-09-02 12:00", "intensifying hurricane, peak 125 kt two days later"),
    ("EP112025", "2025-09-04 12:00", "near peak, then steady decay over cooler water"),
    ("EP142025", "2025-09-22 12:00", "intensification to 90 kt, then decay"),
    ("EP152025", "2025-10-01 12:00", "long-lived moderate storm"),
    ("EP082025", "2025-08-07 12:00", "weak storm with a late, brief strengthening"),
    ("EP052025", "2025-06-17 12:00", "East Pacific rapid intensification to 120 kt (35 kt start)"),
    ("EP062025", "2025-06-30 12:00", "rapid intensification to 100 kt, then quick decay"),
]
OLDER = [  # case_plots.pdf only, 2018-24 (no FNV3): out-of-sample members
    ("AL142018", "2018-10-08 06:00", "Michael: rapid intensification to 135 kt at the Florida landfall"),
    ("AL062018", "2018-09-08 06:00", "Florence: rapid intensification to 130 kt"),
    ("AL052019", "2019-08-29 06:00", "Dorian: intensification to 160 kt over the Bahamas"),
    ("AL132020", "2020-08-25 06:00", "Laura: rapid intensification to 130 kt before the Louisiana landfall"),
    ("AL312020", "2020-11-14 06:00", "Iota: rapid intensification to 135 kt (late season)"),
    ("AL092021", "2021-08-27 06:00", "Ida: rapid intensification to 130 kt before the Louisiana landfall"),
    ("AL092022", "2022-09-24 06:00", "Ian: intensification to 140 kt before the Florida landfall"),
    ("AL132023", "2023-09-06 06:00", "Lee: extreme rapid intensification (+70 kt in 24 h) to 145 kt"),
    ("EP182023", "2023-10-23 06:00", "Otis: extreme rapid intensification (+90 kt in 24 h) to 145 kt at Acapulco"),
    ("AL022024", "2024-06-29 06:00", "Beryl: earliest Category 5 on record"),
    ("AL142024", "2024-10-06 06:00", "Milton: extreme rapid intensification to 150 kt"),
]
BLUE, AQUA, VIOLET, ORANGE = "#2a78d6", "#1baf7a", "#6250d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": MUTED,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})

P = geco.products()
COLOR = {"GECO": BLUE, "IECO": AQUA}
fn = fnv3_data.load_test()
a = pd.read_parquet(INTERIM_DIR / "adecks.parquet", columns=["storm_id", "init", "tech", "tau", "vmax"], filters=[("tech", "==", "OFCL")])
a["storm_id"] = a["storm_id"].astype(str)
bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet")
bt["storm_id"] = bt["storm_id"].astype(str)
names = pd.read_parquet(INTERIM_DIR / "ibtracs.parquet").drop_duplicates("storm_id").set_index("storm_id")["name"]


def member_traces(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """valid time x member table. Corrected members start at the CARQ intensity at t0; raw members at their own t0 intensity."""
    t0 = d["t0"].iloc[0]
    w = d.pivot_table(index="lead", columns="member", values=col)
    w.loc[0] = d.groupby("member")["v_t0"].first().reindex(w.columns) if col == "RAW" else d["vmax0"].iloc[0]
    w = w.sort_index()
    w.index = t0 + pd.to_timedelta(w.index, unit="h")
    return w


def panel(ax, title, color, members, raw=None, t0=None, ofcl=None, best=None, xlim=None):
    if raw is not None:
        ax.plot(raw.index, raw.values, color=MUTED, lw=0.5, alpha=0.6, zorder=1)
    ax.plot(members.index, members.values, color=color, lw=0.7, alpha=0.45, zorder=2)
    ax.plot(members.index, members.mean(axis=1), color=color, lw=2.4, zorder=4)
    q = members.quantile([0.1, 0.9], axis=1)
    ax.plot(members.index, q.loc[0.1], color=color, lw=1, ls=":", zorder=3)
    ax.plot(members.index, q.loc[0.9], color=color, lw=1, ls=":", zorder=3)
    if ofcl is not None:
        ax.plot(ofcl.index, ofcl.values, color=ORANGE, lw=1.8, ls="--", marker="o", ms=3, zorder=5)
    ax.plot(best.index, best.values, color=INK, lw=2, marker="o", ms=2.5, zorder=6)
    if t0 is not None:
        ax.axvline(t0, color=INK2, lw=0.8, ls="-.")
        ax.text(t0, 0.98, " start", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))


def plot_case(storm, t0, why):
    t0 = pd.Timestamp(t0)
    day = t0.normalize()
    runs = fn[(fn["storm_id"] == storm) & (fn["init"] <= t0 - pd.Timedelta(hours=6)) & (fn["init"] >= t0 - pd.Timedelta(hours=24))]["init"]
    fn_init = runs.max() if len(runs) else None
    o = a[(a["storm_id"] == storm) & (a["init"] == t0)].set_index("tau")["vmax"].sort_index()
    o.index = t0 + pd.to_timedelta(o.index, unit="h")
    end = t0 + pd.Timedelta(hours=132)
    b = bt[(bt["storm_id"] == storm) & (bt["time"] >= day - pd.Timedelta(hours=12)) & (bt["time"] <= end)].set_index("time")["vmax"].sort_index()
    xlim = (day - pd.Timedelta(hours=6), end)
    panels = []
    for name in PRODUCTS:
        g = P[name][(P[name]["storm_id"] == storm) & (P[name]["t0"] == t0)]
        if len(g):
            run = f"{t0 - pd.Timedelta(hours=int(g['age'].iloc[0])):%H}Z run"
            panels.append((f"{name} · {PRODUCTS[name][0]} ({g['member'].nunique()}) · {run}", COLOR[name], member_traces(g, name), member_traces(g, "RAW")))
    if fn_init is not None:
        f = fn[(fn["storm_id"] == storm) & (fn["init"] == fn_init) & (fn["lead"] <= 168)]
        fw = f.pivot_table(index="lead", columns="member", values="vmax")
        fw.index = fn_init + pd.to_timedelta(fw.index, unit="h")
        panels.append((f"FNV3 (50), raw · {fn_init:%HZ} run", VIOLET, fw[fw.index >= t0], None))
    if not panels:
        raise ValueError("no members")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, color, members, raw) in zip(axes, panels):
        panel(ax, title, color, members, raw, t0, o, b, xlim)
    for ax in axes:
        ax.set_ylim(0, max(170, b.max() + 15))
        ax.set_xlabel(f"Valid day ({day:%b %Y})")
    axes[0].set_ylabel("Max wind (kt)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, lw=2, marker="o", ms=3, label="Best track"),
               Line2D([], [], color=ORANGE, lw=1.8, ls="--", marker="o", ms=3, label=f"OFCL ({t0:%HZ} cycle)"),
               Line2D([], [], color=INK2, lw=2.4, label="Ensemble mean (GECO / IECO: = HCCA)"), Line2D([], [], color=INK2, lw=1, ls=":", label="10th / 90th percentile"),
               Line2D([], [], color=INK2, lw=0.7, alpha=0.6, label="Members"), Line2D([], [], color=MUTED, lw=0.8, label="Raw (uncorrected) members")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if len(panels) < 3 else 6, bbox_to_anchor=(0.5, -0.12 if len(panels) < 3 else -0.06), fontsize=7.5)
    when = f"{t0:%d %b %Y}, {t0:%H}Z cycle / {(t0 + pd.Timedelta(hours=3)):%H}Z advisory"
    fig.suptitle(f"{names.get(storm, storm).title()} ({storm}), {when}: {why.split(': ', 1)[-1]}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(wspace=0.08, top=0.82)
    return fig


out = ROOT / "docs" / "case_plots.pdf"
prev = ROOT / "docs" / "report" / "preview"
prev.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".tmp.pdf")  # written in full, then renamed: an interruption never leaves a broken PDF
with PdfPages(tmp) as pdf:
    for k, (storm, t0, why) in enumerate(CASES + EXTRA + OLDER, 1):
        try:
            fig = plot_case(storm, t0, why)
        except Exception as e:  # missing data for one case should not stop the rest
            print(f"skipped case {k} {storm} {t0}: {e}")
            continue
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(prev / f"case_{k}.png", dpi=100, bbox_inches="tight")
        if k <= len(CASES):  # vector copies for the write-up
            fig.savefig(ROOT / "docs" / "report" / "figures" / f"fig_case_{k}.svg", bbox_inches="tight")
        plt.close(fig)
os.replace(tmp, out)
print("wrote", out)
