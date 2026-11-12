"""GECO (GEFS Ensemble, Consensus-Oriented): GEFS member shapes around NHC's HCCA consensus.

member_i(tau) = HCCA(tau) + k(tau) * sigma_hat(case, tau) * z_i(tau), with z_i the smoothed, t0-anchored, standardised member
deviations and sigma_hat the predicted spread (as in CCSR). The centre is HCCA as issued (no lead smoothing; leads between HCCA's
12-h values are interpolated, past its last lead the last HCCA - consensus offset is carried forward). Full fitted spread.
HCCA is in NHC's public real-time a-decks, so GECO runs at all four cycles; cases without HCCA fall back to CCSR.
IECO is the same construction on ECMWF's own ENS member tracks (scripts/ens_ieco.py).
Chosen on nine seasons of NHC-GEFS members (docs/ensemble_multiseason.md, where it is the CCSHu variant).
"""
import math

import numpy as np
import pandas as pd

from tcic.config import INTERIM_DIR
from tcic.data.atcf import merge_interpolated
from tcic.ensemble import calib
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

CASE, KEY = sc.CASE, smo.KEY
SIG_DEV, SIG_CEN = 6.0, 0.0  # member-deviation smoothing as CCSR; centre unsmoothed


def hcca() -> pd.DataFrame:
    A = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", ["HCCA", "HCC2"])], columns=["storm_id", "init", "tech", "tau", "vmax"])
    A["storm_id"] = A["storm_id"].astype(str)
    A = merge_interpolated(A)
    A = A[A["tech"] == "HCCA"].rename(columns={"init": "t0", "tau": "lead", "vmax": "hcca"})
    return A.drop_duplicates(CASE)[CASE + ["hcca"]]


def hcca_centre(t: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """Swap tcca_mean for HCCA: HCCA - consensus interpolated along lead, carried forward past HCCA's last lead; no HCCA -> dropped."""
    c = t.drop_duplicates(CASE)[CASE + ["tcca_mean"]].merge(h, on=CASE, how="left").sort_values(CASE)
    c["adj"] = (c["hcca"] - c["tcca_mean"]).groupby([c["storm_id"], c["t0"]]).transform(lambda x: x.interpolate(limit_area="inside").ffill())
    c = c.dropna(subset=["adj"])
    c["tcca_mean"] = c["tcca_mean"] + c["adj"]
    return t.drop(columns=["tcca_mean"]).merge(c[CASE + ["tcca_mean"]], on=CASE)


def fit_k(oof: pd.DataFrame, sig: pd.DataFrame):
    """k(tau) = c0 + c1 tau/100 on out-of-fold CRPS; oof must already carry the HCCA centre."""
    ones = sig * 0 + 1.0
    return calib.fit_k_q(oof, ones, sig, sig, SIG_DEV, SIG_CEN)


def build(t: pd.DataFrame, sig: pd.DataFrame, c0: float, c1: float, name: str = "GECO") -> pd.DataFrame:
    """Members (KEY, member, lead, name) for a member table that already carries the HCCA centre in tcca_mean."""
    ones = sig * 0 + 1.0
    dev, cen = smo.prepare(t, SIG_DEV, SIG_CEN)
    z = calib.standardised_smooth(dev)
    z = z[z.index.droplevel("member").isin(sig.index)]
    return calib.members_q(z, cen, ones, sig, sig, c0, c1, name=name)


def products(seasons=None) -> dict[str, pd.DataFrame]:
    """GECO and IECO members, 2018-26 (each season out of sample to 2024): CASE, member, season, age, obs_vmax, RAW, v_t0, vmax0 and
    the product column. GECO comes from scripts/ens_multiseason.py (run age, raw start from the member table), IECO from scripts/ens_ieco.py."""
    from tcic.config import DATA_DIR, PRODUCTS

    r = DATA_DIR / "results" / "ensemble"
    out = {}
    for name, (_, f) in PRODUCTS.items():
        d = pd.read_parquet(r / f)
        if seasons is not None:
            d = d[d["season"].isin(list(seasons))]
        if name == "GECO":
            cols = KEY + ["member", "v_t0", "vmax0", "age"]
            meta = pd.concat([pd.read_parquet(r / f"{p}_nhc_gefs.parquet", columns=cols) for p in ("oof", "test")]).drop_duplicates(KEY + ["member"])
            d = d.assign(member=d["member"].astype(str)).merge(meta.assign(member=meta["member"].astype(str)), on=KEY + ["member"], how="left")
        else:
            d = d.rename(columns={"v": "RAW"})
        out[name] = d[CASE + ["member", "season", "age", "obs_vmax", "RAW", "v_t0", "vmax0", name]]
    return out


def dressed(p: pd.DataFrame) -> pd.DataFrame:
    """A point forecast (columns CASE, obs, mean) as a Gaussian with its per-lead RMS error in the other 2018-24 seasons (all of them for 2025-26): the fair
    probabilistic yardstick for HCCA ("dressed HCCA")."""
    p = p.assign(season=p["t0"].dt.year, e=p["mean"] - p["obs"])
    sq = p.groupby(["season", "lead"])["e"].agg(lambda e: (e ** 2).sum()).unstack("lead")
    n = p.groupby(["season", "lead"]).size().unstack("lead")
    tr = [s for s in calib.TRAIN if s in sq.index]
    sd = {s: np.sqrt(sq.loc[[t for t in tr if t != s]].sum() / n.loc[[t for t in tr if t != s]].sum()) for s in sq.index}
    sigma = np.array([sd[s][L] for s, L in zip(p["season"], p["lead"])])
    zz = (p["obs"] - p["mean"]) / sigma
    phi, Phi = np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), 0.5 * (1 + np.vectorize(math.erf)(zz / np.sqrt(2)))
    crps = sigma * (zz * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return p.assign(crps=crps, std=sigma, inside=(np.abs(zz) <= 1.2816).astype(float))[CASE + ["obs", "mean", "crps", "std", "inside"]]
