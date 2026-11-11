"""Conformalized quantile regression (CQR) for the M1.5 10-90% intervals.

The raw LightGBM quantile models are too narrow out of sample (67% coverage in CV, 68% on 2025 vs 80% nominal):
trees fit the training storms' spread, and new storms vary more. CQR fixes this without retraining: for each lead,
take the out-of-fold conformity scores s = max(q10 - y, y - q90) and widen (or narrow) both quantiles by the
(1 - alpha) empirical quantile of s. Calibrated on 2018-2024 out-of-fold predictions; applied to the 2025 test.
"""
import numpy as np
import pandas as pd


def fit(oof: pd.DataFrame, obs: pd.Series, alpha: float = 0.2) -> pd.Series:
    """Per-lead widening (kt) so that the [q10, q90] interval covers 1 - alpha of out-of-fold cases."""
    s = np.maximum(oof["q10"] - obs, obs - oof["q90"])
    def q(x):
        n = len(x)
        return float(np.quantile(x, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
    return s.groupby(oof["lead"]).apply(q).rename("widen_kt")


def apply(pred: pd.DataFrame, widen: pd.Series) -> pd.DataFrame:
    w = pred["lead"].map(widen).fillna(widen.max())
    out = pred.copy()
    out["q10"], out["q90"] = pred["q10"] - w, pred["q90"] + w
    return out
