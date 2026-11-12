"""GECO and IECO vs FNV3 members and dressed HCCA (2025–26), on identical cases.

FNV3 "early": newest run that is >= 6 h old at t0 (00/06/12Z runs available), leads shifted by the age -- the same timing
              rule as GECO and IECO. FNV3 "late": the run initialised at t0 itself (not available to forecasters in time).
FNV3 members are used raw (they are already at realistic intensities). Writes docs/fnv3_compare.md.
"""
import warnings

import numpy as np
import pandas as pd

from tcic.config import FEATURES_DIR, PRODUCTS, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import geco
from tcic.ensemble import score as sc

warnings.filterwarnings("ignore", category=RuntimeWarning)
CASE = sc.CASE
LEADS = list(range(12, 121, 12))
F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet", columns=CASE + ["obs_vmax", "obs_dv24", "verifiable"])
F = F[F["verifiable"]]
hp = F[F["lead"].isin(range(12, 121, 12))].merge(geco.hcca(), on=CASE)
HCCA_D = geco.dressed(hp.assign(obs=hp["obs_vmax"], mean=hp["hcca"]))  # spread from the other 2018-24 seasons, so fit on all of them
F = F[F["t0"].dt.year.isin(TEST_SEASONS)]
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


def group(title, ours: dict[str, tuple[pd.DataFrame, str]]):
    """ours: label -> (member table, column). Scores ours, FNV3 early / late and dressed HCCA on identical cases."""
    base = F[F["lead"].isin(LEADS)]
    S, ri = {}, {}
    for label, (d, col) in ours.items():
        d = d[d[col].notna()].merge(base[CASE], on=CASE)
        S[label] = sc.ensemble_case_scores(d, col)
        ri[label] = ri_brier(d, col)
    for mode in ("early", "late"):
        m = fnv3_members(base, mode)
        mm = m[m["lead"].isin(LEADS)].dropna(subset=["obs_vmax"])
        S[f"FNV3 {mode}"] = sc.ensemble_case_scores(mm, "FNV3")
        ri[f"FNV3 {mode}"] = ri_brier(m, "FNV3")
    S["HCCA dressed"] = HCCA_D.merge(base[CASE], on=CASE)
    keys = None
    for s in S.values():
        keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
    rows, sub = [], {}
    for name, s in S.items():
        h = s.merge(keys, on=CASE)
        sub[name] = h
        rows.append({"system": name, "CRPS": h["crps"].mean(), "MAE of mean": (h["mean"] - h["obs"]).abs().mean(),
                     "bias of mean": (h["mean"] - h["obs"]).mean(), "10–90% coverage": h["inside"].mean(),
                     "spread/RMSE": h["std"].mean() / np.sqrt(((h["mean"] - h["obs"]) ** 2).mean())})
    t = pd.DataFrame(rows).set_index("system")
    by_season = pd.DataFrame({n: h.groupby(h["t0"].dt.year).apply(lambda g: pd.Series({
        "CRPS": g["crps"].mean(), "bias": (g["mean"] - g["obs"]).mean(), "coverage": g["inside"].mean()})).stack() for n, h in sub.items()}).T
    txt = (f"\n## {title}\n\n{keys['storm_id'].nunique()} storms, {len(keys)} cases, leads 12–120 h, homogeneous.\n\n" + t.to_markdown(floatfmt=".2f")
           + "\n\nBy season (same cases)\n\n" + by_season.to_markdown(floatfmt=".2f") + "\n\n")
    for a in ours:
        for b in ("HCCA dressed", "FNV3 early", "FNV3 late"):
            for yrs, yl in ((list(TEST_SEASONS), TEST_LABEL), *(([y], str(y)) for y in TEST_SEASONS)):
                sa, sb = (sub[x][sub[x]["t0"].dt.year.isin(yrs)] for x in (a, b))
                d, lo, hi = sc.storm_bootstrap_diff(sa, sb, "crps")
                txt += f"- {yl}: CRPS {a} − {b} = {d:+.2f} [{lo:+.2f}, {hi:+.2f}]{' **significant**' if hi < 0 or lo > 0 else ''}\n"
    by = {n: h.groupby("lead")["crps"].mean() for n, h in sub.items()}
    txt += "\nCRPS by lead\n\n" + pd.DataFrame(by).T.to_markdown(floatfmt=".1f") + "\n"
    txt += "\nRI probability from member counts (leads 36–120 h; each system's own cases)\n\n" + pd.DataFrame(ri).T.to_markdown(floatfmt=".3f") + "\n"
    return txt


P = geco.products(TEST_SEASONS)
report = [f"# GECO and IECO vs FNV3 members and dressed HCCA, {TEST_LABEL}\n\n",
          "GECO / IECO: GEFS / IFS-ENS member shapes around HCCA (newest member run ≥ 6 h old). FNV3 = raw Google DeepMind Weather Lab "
          "members (50), early (newest run ≥ 6 h old, like ours) or late (run at t0; an upper bound, not available in time). Dressed HCCA = "
          "HCCA as a Gaussian with its out-of-season RMS error by lead. Fair CRPS / MAE in kt, all cycles. 95% storm-block bootstrap CIs.\n"]
for name in PRODUCTS:
    report.append(group(f"{name} ({PRODUCTS[name][0]})", {name: (P[name], name)}))
report.append(group("GECO and IECO on their common cases", {n: (P[n], n) for n in PRODUCTS}))
(ROOT / "docs" / "fnv3_compare.md").write_text("".join(report))
print("".join(report))
