"""GECO (GEFS Ensemble, Consensus-Oriented): GEFS member shapes around NHC's HCCA consensus.

member_i(tau) = HCCA(tau) + k(tau) * sigma_hat(case, tau) * z_i(tau), with z_i the smoothed, t0-anchored, standardised member
deviations and sigma_hat the predicted spread (as in CCSR). The centre is HCCA as issued (no lead smoothing; leads between HCCA's
12-h values are interpolated, past its last lead the last HCCA - consensus offset is carried forward). Full fitted spread.
HCCA is in NHC's public real-time a-decks, so GECO runs at all four cycles; cases without HCCA fall back to CCSR.
Chosen on nine seasons of NHC-GEFS members (docs/ensemble_multiseason.md, where it is the CCSHu variant).
"""
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
