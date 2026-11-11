"""M1.5 consensus correction: LightGBM residual models on top of an equal-weight early-aid consensus.

target = best-track intensity at t0 + lead  -  ref (mean of available early aids), one model across all leads.
Variant A uses aid/ensemble/CARQ features only; variant B adds operational SHIPS predictors.
RI cases (observed +30 kt in the 24 h ending at the valid time) are weighted x2, as in the paper.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COLS = ["storm_id", "t0", "valid", "season"]
NON_FEATURES = {"verifiable", "ri", "has_ships"}
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)
N_ROUNDS = 600
QUANTILES = (0.1, 0.5, 0.9)
CLIP = {"vmax": (10.0, 200.0), "pmin": (850.0, 1025.0)}
TARGETS = {"vmax": ("obs_vmax", "ref_vmax"), "pmin": ("obs_pmin", "ref_pmin")}


def feature_columns(df: pd.DataFrame, variant: str) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLS and c not in NON_FEATURES
            and not c.startswith(("obs_", "obs0_")) and pd.api.types.is_numeric_dtype(df[c])]
    if variant == "A":
        cols = [c for c in cols if not c.startswith("ships")]
    return cols


def training_rows(df: pd.DataFrame, var: str) -> pd.DataFrame:
    obs, ref = TARGETS[var]
    return df[df["verifiable"] & df[obs].notna() & df[ref].notna()]


def fit(train: pd.DataFrame, features: list[str], var: str, objective: str = "regression", alpha: float | None = None,
        rounds: int = N_ROUNDS) -> lgb.Booster:
    obs, ref = TARGETS[var]
    y = train[obs] - train[ref]
    w = np.where(train["ri"] == 1, 2.0, 1.0)
    params = {**PARAMS, "objective": objective, **({"alpha": alpha} if alpha is not None else {})}
    return lgb.train(params, lgb.Dataset(train[features], y, weight=w), num_boost_round=rounds)


def predict(model: lgb.Booster, df: pd.DataFrame, features: list[str], var: str) -> np.ndarray:
    _, ref = TARGETS[var]
    return np.clip(df[ref].to_numpy() + model.predict(df[features]), *CLIP[var])


def leave_one_season_out(df: pd.DataFrame, variant: str, var: str, seasons, quantiles: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions for each season in `seasons`, training on all other seasons in `seasons`."""
    rows = training_rows(df, var)
    rows = rows[rows["season"].isin(list(seasons))]
    feats = feature_columns(df, variant)
    out = []
    for s in seasons:
        tr, te = rows[rows["season"] != s], rows[rows["season"] == s]
        if te.empty:
            continue
        pred = te[ID_COLS + ["lead"]].copy()
        pred["mean"] = predict(fit(tr, feats, var), te, feats, var)
        if quantiles:
            for q in QUANTILES:
                pred[f"q{int(q * 100)}"] = predict(fit(tr, feats, var, "quantile", q), te, feats, var)
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def train_test(df: pd.DataFrame, variant: str, var: str, train_seasons, test_season,
               quantiles: bool = True) -> tuple[pd.DataFrame, lgb.Booster]:
    rows = training_rows(df, var)
    feats = feature_columns(df, variant)
    tr = rows[rows["season"].isin(list(train_seasons))]
    te = rows[rows["season"].isin(np.atleast_1d(test_season))]  # one season or several (combined test)
    model = fit(tr, feats, var)
    pred = te[ID_COLS + ["lead"]].copy()
    pred["mean"] = predict(model, te, feats, var)
    if quantiles:
        for q in QUANTILES:
            pred[f"q{int(q * 100)}"] = predict(fit(tr, feats, var, "quantile", q), te, feats, var)
    return pred, model


def as_cases(pred: pd.DataFrame, features: pd.DataFrame, tech: str, var: str = "vmax") -> pd.DataFrame:
    """Convert predictions into the `evaluate.pair` case format so they can be compared with ATCF aids."""
    obs = TARGETS[var][0]
    f = features.set_index(["storm_id", "t0", "lead"])
    p = pred.set_index(["storm_id", "t0", "lead"])
    c = pd.DataFrame({"fcst": p["mean"], "obs": f.loc[p.index, obs], "ri": f.loc[p.index, "ri"].astype(bool),
                      "obs_vmax": f.loc[p.index, "obs_vmax"]}).reset_index()
    from tcic.evaluate import CATEGORY_BINS, CATEGORY_LABELS

    c["category"] = pd.cut(c["obs_vmax"], CATEGORY_BINS, labels=CATEGORY_LABELS, right=False)
    c = c.rename(columns={"t0": "init", "lead": "tau"}).drop(columns="obs_vmax")
    c["err"] = c["fcst"] - c["obs"]
    c["tech"] = tech
    c["valid"] = c["init"] + pd.to_timedelta(c["tau"], unit="h")
    return c
