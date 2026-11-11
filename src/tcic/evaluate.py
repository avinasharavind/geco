"""Intensity verification: homogeneous MAE/bias by lead, category and RI, with storm-block bootstrap CIs.

Two verification conventions are supported:
- "nhc": truth = final b-deck best track; a case counts only if the system is a tropical/subtropical cyclone
  (TD, TS, HU, SD, SS) at both the initial and the valid time (NHC verification report rules).
- "paper": truth = IBTrACS (USA_WIND, fallback WMO_WIND/0.88), any best-track point (Allen et al. 2026).
"""
import numpy as np
import pandas as pd

from tcic.config import INTERIM_DIR, RI_THRESHOLD_KT

TC_TYPES = {"TD", "TS", "HU", "SD", "SS"}
NHC_LEADS = (12, 24, 36, 48, 60, 72, 96, 120)
CATEGORY_BINS = [0, 34, 64, 83, 96, 113, 137, np.inf]
CATEGORY_LABELS = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]


def load_truth(source: str = "nhc") -> pd.DataFrame:
    """Best track indexed by (storm_id, time) with vmax, pmin, is_tc and the observed 24-h intensity change ending at time."""
    if source == "nhc":
        bt = pd.read_parquet(INTERIM_DIR / "bdecks.parquet")
        bt["is_tc"] = bt["ty"].astype(str).isin(TC_TYPES)
    elif source == "ibtracs":
        bt = pd.read_parquet(INTERIM_DIR / "ibtracs.parquet")
        bt["is_tc"] = bt["status"].isin(TC_TYPES)
    else:
        raise ValueError(source)
    bt["storm_id"] = bt["storm_id"].astype(str)
    bt = bt.set_index(["storm_id", "time"])[["vmax", "pmin", "is_tc"]].sort_index()
    prev = bt["vmax"].copy()
    prev.index = prev.index.set_levels(prev.index.levels[1] + pd.Timedelta(hours=24), level="time")
    bt["dv24"] = bt["vmax"] - prev.reindex(bt.index)
    return bt


def pair(forecasts: pd.DataFrame, truth: pd.DataFrame, techs, leads=NHC_LEADS, require_tc: bool = True,
         var: str = "vmax") -> pd.DataFrame:
    """Join forecasts (storm_id, init, tech, tau, <var>) to truth at the valid time; one row per verifiable case."""
    f = forecasts[forecasts["tech"].isin(list(techs)) & forecasts["tau"].isin(list(leads))]
    f = f[["storm_id", "init", "tech", "tau", "vmax", "mslp"]].copy()
    f["storm_id"] = f["storm_id"].astype(str)
    f["tech"] = f["tech"].astype(str)
    f["tau"] = f["tau"].astype(int)
    f["valid"] = f["init"] + pd.to_timedelta(f["tau"], unit="h")
    fvar = "mslp" if var == "pmin" else "vmax"
    f = f.dropna(subset=[fvar])
    t = truth.rename(columns=lambda c: f"obs_{c}")
    f = f.join(t, on=["storm_id", "valid"], how="inner")
    t0 = truth[["is_tc"]].rename(columns={"is_tc": "obs0_is_tc"})
    f = f.join(t0, on=["storm_id", "init"], how="inner")
    if require_tc:
        f = f[f["obs_is_tc"] & f["obs0_is_tc"]]
    f = f.dropna(subset=[f"obs_{var}"])
    f["fcst"] = f[fvar]
    f["obs"] = f[f"obs_{var}"]
    f["err"] = f["fcst"] - f["obs"]
    f["ri"] = f["obs_dv24"] >= RI_THRESHOLD_KT
    f["category"] = pd.cut(f["obs_vmax"], CATEGORY_BINS, labels=CATEGORY_LABELS, right=False)
    return f[["storm_id", "init", "tau", "valid", "tech", "fcst", "obs", "err", "ri", "category"]].reset_index(drop=True)


def homogeneous(cases: pd.DataFrame, techs) -> pd.DataFrame:
    """Keep only (storm, init, tau) cases verified for every tech."""
    techs = list(techs)
    c = cases[cases["tech"].isin(techs)]
    n = c.groupby(["storm_id", "init", "tau"])["tech"].transform("nunique")
    return c[n == len(techs)].reset_index(drop=True)


def by_lead(cases: pd.DataFrame, techs=None) -> pd.DataFrame:
    """MAE, bias and case counts per tech and lead (wide: one row per tech)."""
    g = cases.groupby(["tech", "tau"])["err"]
    out = pd.concat({"mae": g.apply(lambda e: e.abs().mean()), "bias": g.mean(), "n": g.size()}, axis=1)
    if techs is not None:
        out = out.reindex(list(techs), level="tech")
    return out


def overall(cases: pd.DataFrame, techs=None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean absolute error and bias over all cases, with 95% storm-block bootstrap CIs (as in the paper)."""
    techs = list(techs) if techs is not None else sorted(cases["tech"].unique())
    rng = np.random.default_rng(seed)
    storms = cases["storm_id"].unique()
    # per-storm sums so that resampling storms is cheap
    s = cases.assign(ae=cases["err"].abs()).groupby(["tech", "storm_id"]).agg(ae=("ae", "sum"), e=("err", "sum"), n=("err", "size"))
    rows = []
    for tech in techs:
        st = s.loc[tech].reindex(storms, fill_value=0)
        ae, e, n = st["ae"].to_numpy(), st["e"].to_numpy(), st["n"].to_numpy()
        idx = rng.integers(0, len(storms), size=(n_boot, len(storms)))
        nb = n[idx].sum(1)
        mae_b, bias_b = ae[idx].sum(1) / nb, e[idx].sum(1) / nb
        rows.append({"tech": tech, "mae": ae.sum() / n.sum(), "mae_lo": np.percentile(mae_b, 2.5), "mae_hi": np.percentile(mae_b, 97.5),
                     "bias": e.sum() / n.sum(), "bias_lo": np.percentile(bias_b, 2.5), "bias_hi": np.percentile(bias_b, 97.5),
                     "n": int(n.sum()), "storms": int((n > 0).sum())})
    return pd.DataFrame(rows).set_index("tech")


def mae_difference(cases: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """MAE(a) - MAE(b) on homogeneous cases with a 95% storm-block bootstrap CI; significant if the CI excludes 0."""
    c = homogeneous(cases, [a, b]).assign(ae=lambda d: d["err"].abs())
    w = c.pivot_table(index=["storm_id", "init", "tau"], columns="tech", values="ae").reset_index()
    w["d"] = w[a] - w[b]
    s = w.groupby("storm_id")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s["sum"].to_numpy()[idx].sum(1) / s["size"].to_numpy()[idx].sum(1)
    return s["sum"].sum() / s["size"].sum(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)
