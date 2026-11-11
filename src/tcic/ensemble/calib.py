"""Storm-specific spread and a calibrated rapid-intensification (RI) probability.

Spread model. Predicts, for every case (storm, t0, lead), the expected absolute error of the M1.5 corrected consensus mean from
the M1.5 (variant A, early-cycle) features plus the width of M1.5's own 10-90% quantile range. Trained on the out-of-fold
residuals of M1.5 for 2018-2024 (leave-one-season-out again, so every training-season case also gets an out-of-fold spread);
the final model (all of 2018-2024) predicts 2025. The ensemble then uses it directly:
    member_i(tau) = centre(tau) + k(tau) * sigma_hat(case, tau) * z_i(tau),
where z_i are the smoothed, t0-anchored member deviations standardised to unit spread within each case (the ensemble keeps
its scenario *shape*, the spread model sets its *size*), and k(tau) = c0 + c1 * tau/100 is fit on out-of-fold CRPS.

RI model. A LightGBM classifier for P(best track rises >= 30 kt in the 24 h ending at t0 + tau), tau = 24..120 h, from the same
features plus the corrected consensus's own predicted 24-h change; leave-one-season-out over 2018-2024, isotonic recalibration
fit on the out-of-fold probabilities, then trained on all of 2018-2024 for 2025.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from tcic.config import DATA_DIR, FEATURES_DIR, TEST_SEASONS
from tcic.consensus import model as m15
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

CASE, KEY = sc.CASE, smo.KEY
TRAIN = list(range(2018, 2025))
TEST = list(TEST_SEASONS)  # combined held-out test, 2025 + 2026
PARAMS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)


def table() -> tuple[pd.DataFrame, list[str]]:
    """M1.5 rows with the corrected consensus (out-of-fold 2018-24, test 2025) and derived features."""
    F = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet")
    feats = m15.feature_columns(F, "A")
    preds = pd.concat([pd.read_parquet(DATA_DIR / "results" / "m15" / f) for f in ("oof_vmax_A.parquet", "test_vmax_A.parquet")])
    preds = preds[CASE[:2] + ["lead", "mean", "q10", "q90"]].rename(columns={"mean": "tcca_mean"})
    d = F[F["verifiable"] & F["season"].between(2018, max(TEST))].merge(preds, on=CASE)
    d["tcca_qwidth"] = d["q90"] - d["q10"]
    prev = d[CASE + ["tcca_mean"]].assign(lead=d["lead"] + 24).rename(columns={"tcca_mean": "tcca_prev24"})
    d = d.merge(prev, on=CASE, how="left")
    d.loc[d["lead"] == 24, "tcca_prev24"] = d.loc[d["lead"] == 24, "vmax0"]
    d["tcca_dv24"] = d["tcca_mean"] - d["tcca_prev24"]
    d["tcca_minus_v0"] = d["tcca_mean"] - d["vmax0"]
    d["abs_err"] = (d["obs_vmax"] - d["tcca_mean"]).abs()
    extra = ["tcca_mean", "tcca_qwidth", "tcca_dv24", "tcca_minus_v0"]
    return d, feats + extra


def _loso(d: pd.DataFrame, feats, target: str, params: dict, rounds: int) -> tuple[pd.Series, lgb.Booster]:
    """Out-of-fold predictions for TRAIN seasons, and a final model on all TRAIN seasons."""
    oof = pd.Series(np.nan, index=d.index)
    tr_all = d[d["season"].isin(TRAIN)]
    for s in TRAIN:
        tr, te = tr_all[tr_all["season"] != s], tr_all[tr_all["season"] == s]
        b = lgb.train(params, lgb.Dataset(tr[feats], tr[target]), rounds)
        oof.loc[te.index] = b.predict(te[feats])
    final = lgb.train(params, lgb.Dataset(tr_all[feats], tr_all[target]), rounds)
    return oof, final


def spread_model(d: pd.DataFrame, feats) -> pd.DataFrame:
    """Adds 'sigma_hat' (expected |error| of the corrected mean): out-of-fold for 2018-24, final-model for 2025."""
    oof, final = _loso(d, feats, "abs_err", {**PARAMS, "objective": "regression"}, 500)
    d = d.copy()
    d["sigma_hat"] = oof
    te = d["season"].isin(TEST)
    d.loc[te, "sigma_hat"] = final.predict(d.loc[te, feats])
    d["sigma_hat"] = d["sigma_hat"].clip(lower=2.0)
    return d


def ri_model(d: pd.DataFrame, feats) -> pd.DataFrame:
    """Adds 'p_ri' (isotonic-calibrated) and 'p_ri_raw' for leads 24-120 h."""
    r = d[d["lead"].between(24, 120) & d["obs_dv24"].notna()].copy()
    r["y"] = (r["obs_dv24"] >= 30).astype(int)
    oof, final = _loso(r, feats, "y", {**PARAMS, "objective": "binary", "is_unbalance": False}, 400)
    r["p_ri_raw"] = oof
    te = r["season"].isin(TEST)
    r.loc[te, "p_ri_raw"] = final.predict(r.loc[te, feats])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(r.loc[~te, "p_ri_raw"], r.loc[~te, "y"])
    r["p_ri"] = iso.predict(r["p_ri_raw"])
    return r


def standardised(dev: pd.DataFrame) -> pd.DataFrame:
    """Member deviations divided by their per-case, per-lead spread (unit spread; zero at t0)."""
    sd = dev.groupby(level=KEY).transform("std")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (dev / sd).where(sd > 1e-6, 0.0)
    z[0] = 0.0
    return z


def members_p(z: pd.DataFrame, centre: pd.DataFrame, sigma: pd.DataFrame, c0: float, c1: float, name: str = "CCSP") -> pd.DataFrame:
    """centre + k(tau) * sigma_hat * z, with sigma (index KEY, columns = leads) giving sigma_hat per case and lead."""
    leads = np.asarray(smo.LEADS0, dtype=float)
    k = np.maximum(c0 + c1 * leads / 100.0, 0.05)
    keys = z.index.droplevel("member")
    cen = centre.reindex(keys).to_numpy()
    sig = sigma.reindex(index=keys, columns=smo.LEADS0).to_numpy()
    vals = cen + k[None, :] * sig * z.to_numpy()
    out = pd.DataFrame(vals, index=z.index, columns=smo.LEADS0).drop(columns=0)
    out = out.stack().rename(name).reset_index().rename(columns={"level_3": "lead"})
    out[name] = out[name].clip(10, 200)
    return out


def fit_k(oof_members: pd.DataFrame, sigma: pd.DataFrame, sigma_dev: float, sigma_centre: float, max_cases: int = 8000, seed: int = 0):
    """Grid-search k(tau) = c0 + c1 tau/100 minimising out-of-fold CRPS."""
    dev, centre = smo.prepare(oof_members, sigma_dev, sigma_centre)
    keys = dev.index.droplevel("member").unique()
    keys = keys[keys.isin(sigma.index)]
    if len(keys) > max_cases:
        keys = keys[np.random.default_rng(seed).choice(len(keys), max_cases, replace=False)]
    dev = dev[dev.index.droplevel("member").isin(keys)]
    z = standardised(dev)
    obs = oof_members.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"])["obs_vmax"]
    best = None
    for c0 in np.linspace(0.6, 2.0, 15):
        for c1 in np.linspace(-0.4, 0.4, 9):
            mm = members_p(z, centre, sigma, c0, c1).join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax", "CCSP"])
            idx, x = sc.to_matrix(mm, "CCSP")
            y = mm.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"]).loc[pd.MultiIndex.from_frame(idx), "obs_vmax"].to_numpy()
            crps = float(np.nanmean(sc.crps_fair(x, y)))
            if best is None or crps < best[0]:
                best = (crps, c0, c1)
    return best


def reliability(p: pd.Series, y: pd.Series, bins=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001)) -> pd.DataFrame:
    b = pd.cut(p, bins, right=False)
    g = pd.DataFrame({"p": p, "y": y}).groupby(b, observed=True)
    return pd.DataFrame({"cases": g.size(), "mean forecast": g["p"].mean(), "observed frequency": g["y"].mean()})


# ---------------------------------------------------------------- smoothed, asymmetric storm-specific spread (CCSQ)
# CCSP rescales members by a spread that is predicted, and normalised, independently at each lead -> lead-to-lead wobble in
# every trace; and it is symmetric, although errors in rapid-intensification situations are one-sided (the storm ends up
# stronger than forecast). CCSQ smooths the predicted spread and the per-case normalisation along lead time, and scales
# members above / below the centre separately, using the skew of M1.5's own quantile forecasts (q90 - q50 vs q50 - q10).

def smooth_leads(w: pd.DataFrame, sigma_h: float) -> pd.DataFrame:
    """Gaussian smoothing across lead columns (NaN-aware), for tables indexed by case with lead columns."""
    return smo._gauss_smooth(w, sigma_h)


def asymmetry(d: pd.DataFrame, sigma_h: float = 18.0, clip=(0.5, 1.5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upper / lower scale ratios per case and lead from M1.5 quantiles: r_up = 2 * up / (up + down), r_down = 2 - r_up."""
    up = (d["q90"] - d["tcca_mean"]).clip(lower=0.5)
    down = (d["tcca_mean"] - d["q10"]).clip(lower=0.5)
    r = d.assign(r_up=2 * up / (up + down)).pivot_table(index=KEY, columns="lead", values="r_up")
    r = smooth_leads(r, sigma_h).clip(*clip)
    return r, 2.0 - r


def standardised_smooth(dev: pd.DataFrame, sigma_h: float = 18.0) -> pd.DataFrame:
    """Member deviations divided by a lead-smoothed per-case spread (so the normalisation does not wobble from lead to lead)."""
    sd = dev.groupby(level=KEY).std()
    sd = smooth_leads(sd.drop(columns=0), sigma_h)
    sd[0] = np.nan
    sd = sd[dev.columns]
    sdm = sd.reindex(dev.index.droplevel("member")).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sdm > 1e-6, dev.to_numpy() / sdm, 0.0)
    z[:, 0] = 0.0
    return pd.DataFrame(z, index=dev.index, columns=dev.columns)


def members_q(z: pd.DataFrame, centre: pd.DataFrame, sigma: pd.DataFrame, r_up: pd.DataFrame, r_down: pd.DataFrame,
              c0: float, c1: float, name: str = "CCSQ") -> pd.DataFrame:
    leads = np.asarray(smo.LEADS0, dtype=float)
    k = np.maximum(c0 + c1 * leads / 100.0, 0.05)
    keys = z.index.droplevel("member")
    cen = centre.reindex(keys).to_numpy()
    sig = sigma.reindex(index=keys, columns=smo.LEADS0).to_numpy()
    zz = z.to_numpy()
    ru = r_up.reindex(index=keys, columns=smo.LEADS0).fillna(1.0).to_numpy()
    rd = r_down.reindex(index=keys, columns=smo.LEADS0).fillna(1.0).to_numpy()
    scale = np.where(zz >= 0, ru, rd)
    vals = cen + k[None, :] * sig * scale * zz
    out = pd.DataFrame(vals, index=z.index, columns=smo.LEADS0).drop(columns=0)
    out = out.stack().rename(name).reset_index().rename(columns={"level_3": "lead"})
    out[name] = out[name].clip(10, 200)
    return out


def fit_k_q(oof_members: pd.DataFrame, sigma: pd.DataFrame, r_up, r_down, sigma_dev: float, sigma_centre: float,
            max_cases: int = 8000, seed: int = 0):
    dev, centre = smo.prepare(oof_members, sigma_dev, sigma_centre)
    keys = dev.index.droplevel("member").unique()
    keys = keys[keys.isin(sigma.index)]
    if len(keys) > max_cases:
        keys = keys[np.random.default_rng(seed).choice(len(keys), max_cases, replace=False)]
    dev = dev[dev.index.droplevel("member").isin(keys)]
    z = standardised_smooth(dev)
    obs = oof_members.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"])["obs_vmax"]
    best = None
    for c0 in np.linspace(0.6, 2.0, 15):
        for c1 in np.linspace(-0.4, 0.4, 9):
            mm = members_q(z, centre, sigma, r_up, r_down, c0, c1).join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax", "CCSQ"])
            idx, x = sc.to_matrix(mm, "CCSQ")
            y = mm.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"]).loc[pd.MultiIndex.from_frame(idx), "obs_vmax"].to_numpy()
            crps = float(np.nanmean(sc.crps_fair(x, y)))
            if best is None or crps < best[0]:
                best = (crps, c0, c1)
    return best


def residual_quantiles(d: pd.DataFrame, feats, alphas=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Quantiles of the out-of-sample error e = best track - corrected mean (out-of-fold for 2018-24, final model for 2025).

    Unlike M1.5's own quantile models (fit in-sample on intensity, hence too narrow and nearly symmetric), these are fit to
    realistic out-of-sample errors, so their half-widths can be skewed (e.g. towards stronger-than-forecast outcomes)."""
    d = d.copy()
    d["resid"] = d["obs_vmax"] - d["tcca_mean"]
    for a in alphas:
        oof, final = _loso(d, feats, "resid", {**PARAMS, "objective": "quantile", "alpha": a}, 500)
        col = f"e{int(round(100 * a))}"
        d[col] = oof
        te = d["season"].isin(TEST)
        d.loc[te, col] = final.predict(d.loc[te, feats])
    return d
