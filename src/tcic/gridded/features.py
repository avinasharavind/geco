"""M2 gridded features: aggregate the pilot's per-member tracks and environment to one row per (storm, t0, lead).

Early-cycle mapping (same rule as M1.5): a 00Z run of age `AGE[model]` hours at t0 serves t0 = init + age, and the
forecast for lead tau at t0 comes from run lead tau + age. GEFS 00Z -> t0 06Z (6 h old); IFS-ENS 00Z -> t0 12Z (12 h old).
"""
import numpy as np
import pandas as pd

from tcic.config import FEATURES_DIR, RI_THRESHOLD_KT
from tcic.pilot.extract import PILOT_DIR

AGE = {"gefs": 6, "ifs": 12}
LEADS = list(range(12, 169, 12))
CORE_VARS = {"ifs": ["t2m", "d2m", "z500", "z850", "z925", "t850", "t925"], "gefs": ["t2m", "rh2m", "pwat", "z500"]}


def _load(model: str, name: str) -> pd.DataFrame:
    files = sorted((PILOT_DIR / model).glob("*/done.json"))
    return pd.concat([pd.read_parquet(f.parent / f"{name}.parquet") for f in files], ignore_index=True)


def member_stats(model: str) -> pd.DataFrame:
    """Per (storm, init, run lead): statistics across tracked members."""
    t = _load(model, "tracks")
    t = t.sort_values(["storm_id", "init", "member", "lead"])
    g = t.groupby(["storm_id", "init", "member"])
    t["dv12"] = t["vmax"] - g["vmax"].shift(2)  # leads are 6-hourly
    t["dv24"] = t["vmax"] - g["vmax"].shift(4)
    t["ri"] = (t["dv24"] >= RI_THRESHOLD_KT).where(t["dv24"].notna())
    for v in CORE_VARS[model]:
        t[f"cr_{v}"] = t[f"core_{v}"] - t[f"ring_{v}"]  # inner core minus 200-500 km ring (warm core, moist core, ...)
    n_members = t["member"].max() + 1
    k = t.groupby(["storm_id", "init", "lead"])
    out = pd.DataFrame({
        "g_frac_tracked": k["lat"].count() / n_members,
        "g_vmax_mean": k["vmax"].mean(), "g_vmax_std": k["vmax"].std(), "g_vmax_p10": k["vmax"].quantile(0.1),
        "g_vmax_p50": k["vmax"].median(), "g_vmax_p90": k["vmax"].quantile(0.9), "g_vmax_max": k["vmax"].max(),
        "g_pmin_mean": k["pmin"].mean(), "g_pmin_std": k["pmin"].std(), "g_pmin_min": k["pmin"].min(),
        "g_gust_mean": k["gust"].mean(), "g_rmw_med": k["rmw_km"].median(), "g_depth_mean": k["depth"].mean(),
        "g_dv12_mean": k["dv12"].mean(), "g_dv24_mean": k["dv24"].mean(), "g_ri_frac": k["ri"].mean(),
        "g_lat_std": k["lat"].std(), "g_lon_std": k["lon"].std(), "g_lat_mean": k["lat"].mean(), "g_lon_mean": k["lon"].mean(),
    })
    for v in CORE_VARS[model]:
        out[f"g_cr_{v}"] = k[f"cr_{v}"].mean()
        out[f"g_core_{v}"] = k[f"core_{v}"].mean()
    return out.reset_index()


def environment(model: str) -> pd.DataFrame:
    e = _load(model, "env")
    keep = [c for c in e.columns if c not in ("lat_mean", "lon_mean", "n_members")]
    e = e[keep].rename(columns={c: f"e_{c}" for c in keep if c not in ("storm_id", "init", "lead", "env_member")})
    return e


def build(model: str) -> pd.DataFrame:
    """One row per (storm, t0, tau) for this model's runs, with per-lead and initial-time gridded features."""
    age = AGE[model]
    ms = member_stats(model).merge(environment(model), on=["storm_id", "init", "lead"], how="left")
    ms["t0"] = ms["init"] + pd.Timedelta(hours=age)
    feat_cols = [c for c in ms.columns if c.startswith(("g_", "e_"))]
    # state at t0 (run lead = age) and 12 h earlier in the run, for "initial offset" and recent trend features
    init_state = ms[ms["lead"] == age].set_index(["storm_id", "t0"])[["g_vmax_mean", "g_pmin_mean", "g_frac_tracked"]]
    init_state.columns = ["g0_vmax_mean", "g0_pmin_mean", "g0_frac_tracked"]
    rows = ms.assign(tau=ms["lead"] - age)
    rows = rows[rows["tau"].isin(LEADS)]
    rows = rows[["storm_id", "t0", "tau", "env_member"] + feat_cols].rename(columns={"tau": "lead"})
    rows = rows.join(init_state, on=["storm_id", "t0"])
    rows["model"] = model
    return rows


def assemble(model: str) -> pd.DataFrame:
    """Join gridded features to the M1.5 rows (CARQ, aids, targets, verification flags) at matching (storm, t0, lead)."""
    m15 = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet")
    g = build(model)
    df = m15.merge(g, on=["storm_id", "t0", "lead"], how="inner")
    df["g_offset0"] = df["g0_vmax_mean"] - df["vmax0"]  # model intensity at t0 minus CARQ (resolution bias)
    df["g_poffset0"] = df["g0_pmin_mean"] - df["pmin0"]
    df["g_anchor_vmax"] = df["g_vmax_mean"] - df["g_offset0"]  # ensemble-mean intensity, shifted to match CARQ at t0
    df["g_anchor_pmin"] = df["g_pmin_mean"] - df["g_poffset0"]
    df["env_pf1"] = (df["env_member"] == "pf1").astype(int)
    return df


if __name__ == "__main__":
    for mdl in ("ifs", "gefs"):
        d = assemble(mdl)
        v = d[d["verifiable"]]
        d.to_parquet(FEATURES_DIR / f"gridded_{mdl}.parquet", index=False)
        print(f"{mdl}: {len(d):,} rows, {len(v):,} verifiable, {v[['storm_id', 't0']].drop_duplicates().shape[0]} storm-cycles, "
              f"t0 hours {sorted(d['t0'].dt.hour.unique())}, seasons {v.groupby('season').size().to_dict()}")
        print(f"   tracked frac by lead: {v.groupby('lead')['g_frac_tracked'].mean().round(2).iloc[::2].to_dict()}")
        print(f"   anchor MAE vs obs: raw ens-mean {(v['g_vmax_mean'] - v['obs_vmax']).abs().mean():.1f} kt (bias {(v['g_vmax_mean'] - v['obs_vmax']).mean():+.1f}), "
              f"CARQ-shifted {(v['g_anchor_vmax'] - v['obs_vmax']).abs().mean():.1f} kt, aid mean {(v['ref_vmax'] - v['obs_vmax']).abs().mean():.1f} kt")
