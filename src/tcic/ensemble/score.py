"""Scores for intensity ensembles (one row per member) and quantile forecasts, per case = (storm_id, t0, lead)."""
import numpy as np
import pandas as pd

CASE = ["storm_id", "t0", "lead"]
QS_LEVELS = (0.1, 0.5, 0.9)


def to_matrix(df: pd.DataFrame, col: str, max_members: int = 51):
    """Pad members into a (cases, max_members) array (NaN where absent). Returns (case index frame, array)."""
    d = df[CASE + [col]].sort_values(CASE)
    d["k"] = d.groupby(CASE).cumcount()
    w = d.pivot_table(index=CASE, columns="k", values=col)
    return w.index.to_frame(index=False), w.reindex(columns=range(max_members)).to_numpy(dtype=float)


def crps_fair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair ensemble CRPS per row (Ferro 2014); x is (n, m) with NaN padding, y is (n,).

    Uses the sorted-member identity sum_ij |x_i - x_j| = 2 sum_k (2k - m - 1) x_(k), so memory is O(n m), not O(n m^2).
    """
    m = np.sum(np.isfinite(x), axis=1)
    t1 = np.nanmean(np.abs(x - y[:, None]), axis=1)
    xs = np.sort(x, axis=1)  # NaN sort last
    k = np.arange(1, x.shape[1] + 1)[None, :]
    coef = np.where(k <= m[:, None], 2 * k - m[:, None] - 1, 0)
    t2 = np.nansum(coef * np.nan_to_num(xs), axis=1) / (m * np.maximum(m - 1, 1))
    return t1 - t2


def _crps_fair_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference O(n m^2) implementation, kept for the unit test."""
    m = np.sum(np.isfinite(x), axis=1)
    t1 = np.nanmean(np.abs(x - y[:, None]), axis=1)
    t2 = np.nansum(np.abs(x[:, :, None] - x[:, None, :]), axis=(1, 2)) / (2 * m * np.maximum(m - 1, 1))
    return t1 - t2


def pinball(q: np.ndarray, y: np.ndarray, levels=QS_LEVELS) -> np.ndarray:
    """Mean pinball loss over the given quantile levels; q is (n, len(levels))."""
    out = np.zeros(len(y))
    for j, a in enumerate(levels):
        u = y - q[:, j]
        out += np.maximum(a * u, (a - 1) * u)
    return out / len(levels)


def ensemble_case_scores(df: pd.DataFrame, col: str, obs: str = "obs_vmax") -> pd.DataFrame:
    """Per-case scores for an ensemble stored long (one row per member) in column `col`."""
    idx, x = to_matrix(df, col)
    y = df.drop_duplicates(CASE).set_index(CASE).loc[pd.MultiIndex.from_frame(idx), obs].to_numpy(dtype=float)
    q = np.nanquantile(x, QS_LEVELS, axis=1).T
    out = idx.copy()
    out["obs"] = y
    out["mean"] = np.nanmean(x, axis=1)
    out["std"] = np.nanstd(x, axis=1, ddof=1)
    out["crps"] = crps_fair(x, y)
    out["qs"] = pinball(q, y)
    out["q10"], out["q50"], out["q90"] = q[:, 0], q[:, 1], q[:, 2]
    out["abs_err_mean"] = np.abs(out["mean"] - y)
    out["inside"] = ((y >= q[:, 0]) & (y <= q[:, 2])).astype(float)
    out["pit"] = (np.sum(x < y[:, None], axis=1) + 0.5 * np.sum(x == y[:, None], axis=1)) / np.sum(np.isfinite(x), axis=1)
    return out


def quantile_case_scores(df: pd.DataFrame, obs: str = "obs_vmax") -> pd.DataFrame:
    """Per-case scores for a q10/q50/q90 (+ mean) forecast, one row per case."""
    out = df[CASE].copy()
    y = df[obs].to_numpy(dtype=float)
    q = df[["q10", "q50", "q90"]].to_numpy(dtype=float)
    out["obs"], out["mean"], out["q10"], out["q50"], out["q90"] = y, df["mean"].to_numpy(), q[:, 0], q[:, 1], q[:, 2]
    out["qs"] = pinball(q, y)
    out["abs_err_mean"] = np.abs(out["mean"] - y)
    out["inside"] = ((y >= q[:, 0]) & (y <= q[:, 2])).astype(float)
    return out


def storm_bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0):
    """Mean(a[col]) - mean(b[col]) on shared cases, with a 95% storm-block bootstrap CI."""
    m = a[CASE + [col]].merge(b[CASE + [col]], on=CASE, suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_a"] - m[f"{col}_b"]
    s = m.groupby("storm_id")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s["sum"].to_numpy()[idx].sum(1) / s["size"].to_numpy()[idx].sum(1)
    return float(s["sum"].sum() / s["size"].sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def spread_scale(df: pd.DataFrame, col: str, grid=np.concatenate([np.linspace(0.3, 5.0, 48), np.linspace(5.5, 20.0, 30)])) -> pd.Series:
    """Per-lead factor lambda that minimises mean CRPS when members are scaled about the ensemble mean."""
    out = {}
    for lead, g in df.groupby("lead"):
        idx, x = to_matrix(g, col)
        y = g.drop_duplicates(CASE).set_index(CASE).loc[pd.MultiIndex.from_frame(idx), "obs_vmax"].to_numpy(dtype=float)
        mu = np.nanmean(x, axis=1, keepdims=True)
        best = min(grid, key=lambda lam: np.nanmean(crps_fair(mu + lam * (x - mu), y)))
        out[lead] = best
    return pd.Series(out, name="lambda")


def apply_scale(df: pd.DataFrame, col: str, lam: pd.Series, new: str) -> pd.DataFrame:
    mu = df.groupby(CASE)[col].transform("mean")
    df[new] = mu + df["lead"].map(lam).fillna(lam.iloc[-1]).to_numpy() * (df[col] - mu)
    return df
