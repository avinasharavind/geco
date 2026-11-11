import numpy as np
import pandas as pd
import pytest

from tcic import evaluate as ev
from tcic.config import RAW_DIR
from tcic.data import atcf

MELISSA_B = RAW_DIR / "atcf" / "2025" / "bal132025.dat.gz"
needs_data = pytest.mark.skipif(not MELISSA_B.exists(), reason="ATCF 2025 decks not downloaded")


def test_latlon_parsing():
    assert atcf._latlon("163N") == pytest.approx(16.3)
    assert atcf._latlon("761W") == pytest.approx(-76.1)
    assert atcf._latlon("05S") == pytest.approx(-0.5)
    assert np.isnan(atcf._latlon(""))


@needs_data
def test_melissa_best_track():
    b = atcf.read_deck(MELISSA_B)
    best = b[b["tech"] == "BEST"]
    assert set(best["storm_id"]) == {"AL132025"}
    assert best["vmax"].max() >= 150  # Category 5 peak
    assert best["mslp"].min() < 900
    peak = best.loc[best["vmax"].idxmax()]
    assert 15 < peak["lat"] < 20 and -80 < peak["lon"] < -74  # south of Jamaica


@needs_data
def test_adeck_has_early_aids():
    a = atcf.read_deck(RAW_DIR / "atcf" / "2025" / "aal132025.dat.gz")
    techs = set(a["tech"])
    assert {"CARQ", "OFCL", "IVCN", "HCCA", "DSHP", "LGEM", "AVNI", "HFAI"} <= techs
    assert not a.duplicated(["init", "tech", "tau"]).any()  # wind-radii lines collapsed


def test_merge_interpolated_prefers_6h():
    t0 = pd.Timestamp("2025-10-27 06:00")
    t1 = pd.Timestamp("2025-10-27 12:00")
    df = pd.DataFrame({
        "storm_id": ["AL132025"] * 3, "init": [t0, t0, t1], "tech": ["EMXI", "EMX2", "EMX2"],
        "tau": [12, 12, 12], "vmax": [100.0, 90.0, 95.0],
    })
    m = atcf.merge_interpolated(df)
    assert set(m["tech"]) == {"EMXI"}
    assert m.set_index("init").loc[t0, "vmax"] == 100.0  # 6-h version kept
    assert m.set_index("init").loc[t1, "vmax"] == 95.0  # 12-h fallback used


def _synthetic_cases():
    rows = []
    for s in range(10):
        for i in range(4):
            for tech, e in (("A", 5.0), ("B", 10.0)):
                rows.append({"storm_id": f"S{s}", "init": pd.Timestamp("2025-09-01") + pd.Timedelta(hours=6 * i),
                             "tau": 24, "tech": tech, "err": e * (1 if i % 2 else -1)})
    rows.append({"storm_id": "S0", "init": pd.Timestamp("2025-10-01"), "tau": 24, "tech": "A", "err": 50.0})  # A only
    return pd.DataFrame(rows)


def test_homogeneous_drops_unmatched():
    c = ev.homogeneous(_synthetic_cases(), ["A", "B"])
    assert len(c) == 80 and c["err"].abs().max() == 10


def test_overall_and_difference():
    c = ev.homogeneous(_synthetic_cases(), ["A", "B"])
    o = ev.overall(c, ["A", "B"], n_boot=200)
    assert o.loc["A", "mae"] == pytest.approx(5) and o.loc["B", "mae"] == pytest.approx(10)
    assert o.loc["A", "bias"] == pytest.approx(0)
    d, lo, hi = ev.mae_difference(c, "A", "B", n_boot=200)
    assert d == pytest.approx(-5) and lo == pytest.approx(-5) and hi == pytest.approx(-5)


FEATURES = RAW_DIR.parent / "features" / "consensus_features.parquet"


@pytest.mark.skipif(not FEATURES.exists(), reason="consensus features not built")
def test_consensus_features_are_early():
    """Leakage audit: ensemble runs are at least 6 h (GEFS) / 12 h (ECMWF) old at t0; no truth columns used as features."""
    from tcic.consensus import model as m

    f = pd.read_parquet(FEATURES)
    assert set(f["gefs_age"].dropna().unique()) <= {6, 12}
    assert set(f["ecens_age"].dropna().unique()) <= {12, 18}
    for variant in ("A", "B"):
        cols = m.feature_columns(f, variant)
        assert not any(c.startswith(("obs_", "obs0_")) or c in ("ri", "verifiable", "valid") for c in cols)
    assert not any(c.startswith("ships") for c in m.feature_columns(f, "A"))
    # SHIPS rows come from the run at t0 itself (computed along OFCI, available ~t0+2 h)
    assert f.loc[f["has_ships"] == 1, "ships0_sst"].notna().all()


def test_crps_fair_sorted_matches_pairwise():
    from tcic.ensemble import score as sc

    rng = np.random.default_rng(1)
    x = rng.normal(50, 15, size=(200, 31))
    x[rng.random(x.shape) < 0.2] = np.nan  # ragged member counts
    y = rng.normal(50, 20, size=200)
    assert np.allclose(sc.crps_fair(x, y), sc._crps_fair_pairwise(x, y))
