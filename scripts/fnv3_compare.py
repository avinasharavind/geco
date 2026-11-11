"""Consensus-centred corrected ensembles vs FNV3 members (2025–26), on identical cases.

FNV3 "early": newest run that is >= 6 h old at t0 (00/06/12Z runs available), leads shifted by the age -- the same timing
              rule as our ensembles. FNV3 "late": the run initialised at t0 itself (not available to forecasters in time).
FNV3 members are used raw (they are already at realistic intensities). Writes docs/fnv3_compare.md.
"""
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, FEATURES_DIR, FINAL_ENSEMBLE, INTERIM_DIR, ROOT, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import score as sc

warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys

COL = sys.argv[1] if len(sys.argv) > 1 else "final"  # which of our ensemble versions to score; "final" = each source's FINAL_ENSEMBLE (GECO / IECO)
CASE = sc.CASE
LEADS = list(range(12, 121, 12))
RES = DATA_DIR / "results" / "ensemble"
F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet", columns=CASE + ["obs_vmax", "obs_dv24", "verifiable", "vmax0"])
F = F[F["verifiable"] & F["t0"].dt.year.isin(TEST_SEASONS)]
fn = fnv3_data.load_test()
runs = fn[["storm_id", "init"]].drop_duplicates()


def fnv3_members(cases: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Member rows (storm, t0, lead, member, FNV3) for the given cases."""
    c = cases[["storm_id", "t0"]].drop_duplicates()
    if mode == "late":
        c = c.assign(init=c["t0"])
    else:  # newest run at least 6 h old
        cand = c.merge(runs, on="storm_id")
        cand = cand[(cand["init"] <= cand["t0"] - pd.Timedelta(hours=6)) & (cand["init"] >= cand["t0"] - pd.Timedelta(hours=24))]
        c = cand.sort_values("init").groupby(["storm_id", "t0"]).tail(1)
    c = c.merge(runs, on=["storm_id", "init"])
    c["age"] = (c["t0"] - c["init"]) / pd.Timedelta(hours=1)
    m = c.merge(fn, on=["storm_id", "init"])
    m["lead"] = m["lead"] - m["age"]
    m = m[m["lead"].isin(range(-24, 169, 12))]
    out = m.rename(columns={"vmax": "FNV3"})[["storm_id", "t0", "lead", "member", "FNV3", "age"]]
    return out.merge(F[CASE + ["obs_vmax", "obs_dv24"]], on=CASE, how="left")


def ri_brier(d: pd.DataFrame, col: str) -> dict:
    x = d[["storm_id", "t0", "lead", "member", col]].dropna()
    prev = x.assign(lead=x["lead"] + 24).rename(columns={col: "prev"})
    x = x.merge(prev, on=["storm_id", "t0", "lead", "member"]).merge(F[CASE + ["obs_dv24"]], on=CASE)
    x = x[x["lead"].between(36, 120) & x["obs_dv24"].notna()]
    p = (x[col] - x["prev"] >= 30).groupby([x["storm_id"], x["t0"], x["lead"]]).mean()
    o = (x.drop_duplicates(CASE).set_index(CASE)["obs_dv24"] >= 30).astype(float).loc[p.index]
    return {"Brier": float(((p - o) ** 2).mean()), "mean P": float(p.mean()), "obs freq": float(o.mean()),
            "Brier clim": float(((o.mean() - o) ** 2).mean()), "cases": int(len(p))}


def group(title, ours: dict[str, pd.DataFrame], hour: int | None):
    """ours: label -> member table with a CCC column."""
    base = F if hour is None else F[F["t0"].dt.hour == hour]
    base = base[base["lead"].isin(LEADS)]
    S, ri = {}, {}
    for label, d in ours.items():
        d = d[d[COL].notna()].merge(base[CASE], on=CASE)
        S[label] = sc.ensemble_case_scores(d, COL)
        ri[label] = ri_brier(d, COL)
    for mode in ("early", "late"):
        m = fnv3_members(base, mode)
        mm = m[m["lead"].isin(LEADS)].dropna(subset=["obs_vmax"])
        if mm.empty:
            continue
        S[f"FNV3 {mode} (age {sorted(m['age'].unique().astype(int))} h)"] = sc.ensemble_case_scores(mm, "FNV3")
        ri[f"FNV3 {mode}"] = ri_brier(m, "FNV3")
    keys = None
    for s in S.values():
        k = s[CASE]
        keys = k if keys is None else keys.merge(k, on=CASE)
    rows, sub = [], {}
    for name, s in S.items():
        h = s.merge(keys, on=CASE)
        sub[name] = h
        rows.append({"system": name, "CRPS": h["crps"].mean(), "quantile score": h["qs"].mean(), "MAE of mean": h["abs_err_mean"].mean(),
                     "bias of mean": (h["mean"] - h["obs"]).mean(), "10–90% coverage": h["inside"].mean(),
                     "spread/RMSE": h["std"].mean() / np.sqrt(((h["mean"] - h["obs"]) ** 2).mean())})
    t = pd.DataFrame(rows).set_index("system").sort_values("CRPS")
    by_season = pd.DataFrame({n: h.groupby(h["t0"].dt.year).apply(lambda g: pd.Series({
        "CRPS": g["crps"].mean(), "bias": (g["mean"] - g["obs"]).mean(), "coverage": g["inside"].mean()})).stack() for n, h in sub.items()}).T
    txt = (f"\n## {title}\n\n{keys['storm_id'].nunique()} storms, {len(keys)} cases, leads 12–120 h, homogeneous.\n\n" + t.to_markdown(floatfmt=".2f")
           + "\n\nBy season (same cases)\n\n" + by_season.to_markdown(floatfmt=".2f") + "\n")
    names = list(sub)
    fnv = [n for n in names if n.startswith("FNV3")]
    for a in [n for n in names if not n.startswith("FNV3")]:
        for b in fnv:
            d, lo, hi = sc.storm_bootstrap_diff(sub[a], sub[b], "crps")
            txt += f"- CRPS {a} − {b} = {d:+.2f} [{lo:+.2f}, {hi:+.2f}]{' **significant**' if hi < 0 or lo > 0 else ''}\n"
    by = {n: s.merge(keys, on=CASE).groupby("lead")["crps"].mean() for n, s in sub.items()}
    txt += "\nCRPS by lead\n\n" + pd.DataFrame(by).T.to_markdown(floatfmt=".1f") + "\n"
    txt += "\nRI probability (leads 36–120 h; each system's own cases)\n\n" + pd.DataFrame(ri).T.to_markdown(floatfmt=".3f") + "\n"
    return txt


def load(name):
    d = pd.read_parquet(RES / f"test_{name}.parquet")
    if COL == "final":  # default: each source's final ensemble, scored under one column name
        d[COL] = d[FINAL_ENSEMBLE[name]]
    return d


report = ["# Consensus-centred corrected ensembles vs FNV3 members, 2025–26\n",
          "Ours = consensus-centred, spread-calibrated members (CCC): M1.5's corrected mean + each member's raw deviation, scaled per lead. "
          "FNV3 = raw Google DeepMind Weather Lab members (50), early (newest run ≥ 6 h old, like ours) or late (run at t0; an upper bound). "
          "CRPS/quantile score/MAE in kt. 95% storm-block bootstrap CIs.\n"]
lab = lambda name: FINAL_ENSEMBLE[name] if COL == "final" else COL
report[0] = report[0].replace("Consensus-centred corrected ensembles", f"Consensus-centred corrected ensembles ({'final: GECO / IECO' if COL == 'final' else COL})")
report.append(group("IFS-ENS cycles (t0 = 12Z)", {f"Ours IFS-ENS {lab('pilot_ifs')}": load("pilot_ifs")}, 12))
# NHC's ECMWF member tracks exist only in the archive (2025), so they get their own block rather than shrinking the one above
report.append(group("IFS-ENS cycles, 2025 only, with NHC's ECMWF members", {f"Ours IFS-ENS {lab('pilot_ifs')}": load("pilot_ifs"), f"NHC-ECMWF {lab('nhc_ecmwf')}": load("nhc_ecmwf")}, 12))
report.append(group("GEFS cycles (t0 = 06Z)", {f"Ours GEFS {lab('pilot_gefs')}": load("pilot_gefs"), f"NHC-GEFS {lab('nhc_gefs')}": load("nhc_gefs")}, 6))
report.append(group("All 2025–26 cycles (NHC-tracker GEFS members)", {f"NHC-GEFS {lab('nhc_gefs')}": load("nhc_gefs")}, None))
(ROOT / "docs" / f"fnv3_compare{'' if COL == 'final' else '_' + COL.lower()}.md").write_text("".join(report))
print("".join(report))
