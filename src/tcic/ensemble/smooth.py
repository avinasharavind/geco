"""Physically smoother consensus-centred ensembles (CCS).

The first consensus-centred version (CCC) scaled each member's raw deviation from the ensemble mean independently at every
lead. That amplified tracker noise and the members' different starting intensities, giving unphysical 12-h jumps. CCS:
  1. anchored deviation   d_i(tau) = (v_i(tau) - mean(tau)) - (v_i(t0) - mean(t0)), so every member starts at the
                          operational intensity and spread grows with lead;
  2. temporal smoothing   Gaussian kernel along each member's own trajectory (sigma in hours, NaN-aware; 6 h in use), d_i(0) = 0;
  3. smooth spread scale  lambda(tau) = exp(a + b * tau / 100), fit by minimising out-of-fold CRPS;
  4. smooth centre        M1.5's corrected mean, anchored at the CARQ intensity at t0 and lightly smoothed.
Member value: CCS_i(tau) = centre(tau) + lambda(tau) * d_i(tau).
"""
import numpy as np
import pandas as pd

from tcic.ensemble import score as sc

LEADS0 = [0] + list(range(12, 169, 12))
KEY = ["storm_id", "t0"]


def _gauss_smooth(w: pd.DataFrame, sigma_h: float) -> pd.DataFrame:
    """NaN-aware Gaussian smoothing across lead columns; result is NaN wherever the input is NaN."""
    if not sigma_h:
        return w
    leads = np.asarray(w.columns, dtype=float)
    x = w.to_numpy(dtype=float)
    ok = np.isfinite(x)
    k = np.exp(-0.5 * ((leads[None, :] - leads[:, None]) / sigma_h) ** 2)  # (target, source)
    num = np.nan_to_num(x) @ k.T
    den = ok.astype(float) @ k.T
    out = np.where(ok, num / np.maximum(den, 1e-12), np.nan)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def prepare(d: pd.DataFrame, sigma_dev: float = 18.0, sigma_centre: float = 12.0):
    """Smoothed anchored deviations (index storm, t0, member; columns LEADS0) and the smoothed centre (index storm, t0)."""
    d = d[d["tcca_mean"].notna()]
    first = d.drop_duplicates(KEY + ["member"])
    mean_t0 = first.groupby(KEY)["v_t0"].mean().rename("mean_t0")
    dd = d.merge(mean_t0, on=KEY)
    dd["dev"] = (dd["v"] - dd["ens_mean"]) - (dd["v_t0"] - dd["mean_t0"])
    w = dd.pivot_table(index=KEY + ["member"], columns="lead", values="dev").reindex(columns=LEADS0)
    w[0] = 0.0
    dev = _gauss_smooth(w, sigma_dev)
    dev[0] = 0.0
    c = d.drop_duplicates(KEY + ["lead"]).pivot_table(index=KEY, columns="lead", values="tcca_mean").reindex(columns=LEADS0)
    c[0] = d.drop_duplicates(KEY).set_index(KEY)["vmax0"].reindex(c.index)
    centre = _gauss_smooth(c, sigma_centre)
    centre[0] = c[0]
    return dev, centre


def members(dev: pd.DataFrame, centre: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Long table (storm, t0, member, lead, CCS) for leads > 0."""
    lam = np.exp(a + b * np.asarray(LEADS0, dtype=float) / 100.0)
    cen = centre.reindex(dev.index.droplevel("member"))
    vals = cen.to_numpy() + lam[None, :] * dev.to_numpy()
    out = pd.DataFrame(vals, index=dev.index, columns=LEADS0).drop(columns=0)
    out = out.stack().rename("CCS").reset_index().rename(columns={"level_3": "lead"})
    out["CCS"] = out["CCS"].clip(10, 200)
    return out


def fit(oof: pd.DataFrame, sigma_dev: float, sigma_centre: float, max_cases: int = 8000, seed: int = 0):
    """Grid-search (a, b) for lambda(tau) = exp(a + b tau/100) minimising mean fair CRPS on out-of-fold cases."""
    dev, centre = prepare(oof, sigma_dev, sigma_centre)
    keys = dev.index.droplevel("member").unique()
    if len(keys) > max_cases:
        keys = keys[np.random.default_rng(seed).choice(len(keys), max_cases, replace=False)]
    dev = dev[dev.index.droplevel("member").isin(keys)]
    obs = oof.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"])["obs_vmax"]
    best = None
    for a in np.log(np.linspace(0.6, 4.0, 18)):
        for b in np.linspace(-0.6, 0.6, 13):
            m = members(dev, centre, a, b)
            m = m.join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax"])
            idx, x = sc.to_matrix(m.rename(columns={"t0": "t0"}), "CCS")
            y = m.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"]).loc[pd.MultiIndex.from_frame(idx), "obs_vmax"].to_numpy()
            crps = float(np.nanmean(sc.crps_fair(x, y)))
            if best is None or crps < best[0]:
                best = (crps, a, b)
    return best


def roughness(traces: pd.DataFrame, value: str, step_h: int = 12, flip_kt: float = 10.0) -> dict:
    """Step-to-step behaviour of member traces (one row per storm, t0, member, lead incl. 0)."""
    t = traces[traces["lead"] % step_h == 0].sort_values(KEY + ["member", "lead"])
    g = t.groupby(KEY + ["member"])[value]
    dv = g.diff()
    nxt = dv.groupby([t[k] for k in KEY + ["member"]]).shift(-1)
    flip = (np.sign(dv) != np.sign(nxt)) & (dv.abs() >= flip_kt) & (nxt.abs() >= flip_kt)
    a = dv.abs().dropna()
    return {"mean |Δ|": a.mean(), "p95 |Δ|": a.quantile(0.95), "p99 |Δ|": a.quantile(0.99), "share |Δ| > 30 kt": (a > 30).mean(),
            "flip rate": flip[nxt.notna() & dv.notna()].mean(), "steps": int(len(a))}


# ---------------------------------------------------------------- spread normalisation (CCSN)
# The raw ensemble's overall dispersion changes with model upgrades (e.g. ECMWF members are ~2x more spread in 2025 than
# 2018-24), so a fixed lambda learned on past seasons mis-scales new ones. CCSN rescales each case's smoothed deviations by
# the typical raw spread over a trailing window of *past* cases only (works the same way in real time), keeps a tunable
# share of case-to-case (flow-dependent) variation, and applies a calibrated absolute spread s(tau):
#   z_i(tau)      = d_i(tau) / sd_case(tau) * (sd_case(tau) / ref(tau, t0)) ** gamma
#   CCSN_i(tau)   = centre(tau) + k * s(tau) * z_i(tau),   s(tau) = A * (tau / 12) ** p   (error grows fast, then levels off)
# gamma = 0: every case gets the same spread; gamma = 1: spread proportional to the case's raw spread relative to recent
# cases. k is a user "sharpness" knob (1 = CRPS-optimal).

def case_spread(dev: pd.DataFrame) -> pd.DataFrame:
    """Per (storm, t0): std across members of the smoothed anchored deviations, per lead."""
    return dev.groupby(level=KEY).std()


def trailing_reference(sd: pd.DataFrame, window_days: int = 120, min_cases: int = 30) -> pd.DataFrame:
    """For each case, the median case spread per lead over cases initialised in the previous `window_days` (strictly before
    t0); falls back to all earlier cases when the window has fewer than `min_cases`. NaN when there is no history."""
    sd = sd.sort_index(level="t0")
    t0s = sd.index.get_level_values("t0")
    out = np.full(sd.shape, np.nan)
    vals = sd.to_numpy()
    for i, t in enumerate(t0s):
        past = t0s < t
        win = past & (t0s >= t - pd.Timedelta(days=window_days))
        use = win if win.sum() >= min_cases else past
        if use.sum() >= min_cases:
            out[i] = np.nanmedian(vals[use], axis=0)
    return pd.DataFrame(out, index=sd.index, columns=sd.columns)


def normalised(dev: pd.DataFrame, sd: pd.DataFrame, ref: pd.DataFrame, gamma: float) -> pd.DataFrame:
    keys = dev.index.droplevel("member")
    s_case = sd.reindex(keys).to_numpy()
    r = ref.reindex(keys).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = dev.to_numpy() / s_case * (s_case / r) ** gamma
    z[:, 0] = 0.0
    return pd.DataFrame(z, index=dev.index, columns=dev.columns)


def spread_curve(A: float, p: float) -> np.ndarray:
    return A * (np.maximum(np.asarray(LEADS0, dtype=float), 1.0) / 12.0) ** p


def members_n(z: pd.DataFrame, centre: pd.DataFrame, A: float, p: float, k: float = 1.0, name: str = "CCSN") -> pd.DataFrame:
    s = spread_curve(A, p)
    cen = centre.reindex(z.index.droplevel("member"))
    vals = cen.to_numpy() + k * s[None, :] * z.to_numpy()
    out = pd.DataFrame(vals, index=z.index, columns=LEADS0).drop(columns=0)
    out = out.stack().rename(name).reset_index().rename(columns={"level_3": "lead"})
    out[name] = out[name].clip(10, 200)
    return out


def fit_n(oof: pd.DataFrame, history: pd.DataFrame, sigma_dev: float, sigma_centre: float, gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
          max_cases: int = 8000, seed: int = 0):
    """Fit (gamma, a, b) on out-of-fold cases; the trailing reference uses `history` (all cases from the source, any season,
    but each case only sees cases initialised before it)."""
    dev_all, _ = prepare(history, sigma_dev, sigma_centre)
    sd_all = case_spread(dev_all)
    ref_all = trailing_reference(sd_all)
    dev, centre = prepare(oof, sigma_dev, sigma_centre)
    keys = dev.index.droplevel("member").unique()
    keys = keys[np.isfinite(ref_all.reindex(keys)[12].to_numpy())]
    if len(keys) > max_cases:
        keys = keys[np.random.default_rng(seed).choice(len(keys), max_cases, replace=False)]
    dev = dev[dev.index.droplevel("member").isin(keys)]
    obs = oof.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"])["obs_vmax"]
    best = None
    for g in gammas:
        z = normalised(dev, sd_all, ref_all, g)
        for A in np.linspace(5.0, 16.0, 23):
            for pw in np.linspace(0.0, 0.8, 9):
                m = members_n(z, centre, A, pw).join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax", "CCSN"])
                idx, x = sc.to_matrix(m, "CCSN")
                y = m.drop_duplicates(KEY + ["lead"]).set_index(KEY + ["lead"]).loc[pd.MultiIndex.from_frame(idx), "obs_vmax"].to_numpy()
                crps = float(np.nanmean(sc.crps_fair(x, y)))
                if best is None or crps < best[0]:
                    best = (crps, g, A, pw)
    return best, (sd_all, ref_all)
