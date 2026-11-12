"""Member-level tables for the corrected intensity ensembles: one row per (storm, t0, lead, member).

Two sources, same early-cycle rule as M1.5 (newest run that is old enough at t0; leads shifted by the run's age):
- "nhc":   NHC tracker output for GEFS members AP01-30 (age 6 or 12 h, all cycles, 2018-2026) and ECMWF members EE01-50
           (age 12 or 18 h, 2020-2025; absent from 2026 real-time decks). Intensity and track only.
- "tf":    ECMWF's own ENS tracks (BUFR, public in real time; identical to NHC's EE01-50), age 6 h at every cycle from 2023,
           NHC's archived EE01-50 before (age 12 or 18 h). Writes members_tf_ecmwf.parquet; built by scripts/ens_ieco.py.
- "pilot": our own tracker on the extracted IFS-ENS (51 members, 00Z runs at t0 = 12Z) and GEFS (31 members, 00Z at t0 = 06Z)
           runs, 2024-2026, with per-member structure (RMW, MSLP depth, gust, inner-core minus ring) and the run's environment.
Storm state, targets and verification flags come from the M1.5 table (CARQ at t0, final/provisional b-deck truth).
"""
import numpy as np
import pandas as pd

from tcic.config import FEATURES_DIR, INTERIM_DIR, RI_THRESHOLD_KT
from tcic.gridded.features import AGE as PILOT_AGE, CORE_VARS, environment
from tcic.pilot.extract import PILOT_DIR

LEADS = list(range(12, 169, 12))
NEEDED = sorted({0, *LEADS, *[l - 12 for l in LEADS], *[l - 24 for l in LEADS]})
NHC_MEMBERS = {"gefs": ([f"AP{i:02d}" for i in range(1, 31)], (6, 12)), "ecmwf": ([f"EE{i:02d}" for i in range(1, 51)], (12, 18))}
STATE = ["vmax0", "pmin0", "dv6", "dv12", "dv24", "u_motion", "v_motion", "lat0", "lon0", "dist2land0", "basin_ep", "doy"]
ROW = ["storm_id", "t0", "lead", "season", "obs_vmax", "obs_dv24", "verifiable", "ri"] + STATE


def _long(wide_v: pd.DataFrame, wide_p: pd.DataFrame | None) -> pd.DataFrame:
    """wide (storm, t0, member) x relative-lead tables -> long rows per lead with own-trajectory features."""
    out = []
    for lead in LEADS:
        if lead not in wide_v:
            continue
        d = pd.DataFrame({"v": wide_v[lead]})
        d["v_m12"] = wide_v.get(lead - 12)
        d["v_m24"] = wide_v.get(lead - 24)
        d["v_t0"] = wide_v.get(0)
        if wide_p is not None and lead in wide_p:
            d["p"] = wide_p[lead]
            d["p_t0"] = wide_p.get(0)
        d["lead"] = lead
        out.append(d.dropna(subset=["v"]))
    return pd.concat(out).reset_index()


def _decks(members: list[str]) -> pd.DataFrame:
    a = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", members)], columns=["storm_id", "init", "tech", "tau", "vmax", "mslp"])
    a["storm_id"], a["tech"], a["tau"] = a["storm_id"].astype(str), a["tech"].astype(str), a["tau"].astype(int)
    return a


def _aged(a: pd.DataFrame, ages) -> pd.DataFrame:
    """Member tracks (storm_id, init, tech, tau, vmax, mslp) -> long rows at t0 = init + age, newest eligible run first."""
    parts, have = [], None
    for age in ages:
        e = a.assign(t0=a["init"] + pd.Timedelta(hours=age), rel=a["tau"] - age)
        e = e[e["rel"].isin(NEEDED)]
        keys = e[["storm_id", "t0"]].drop_duplicates()
        if have is not None:
            keys = keys.merge(have, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
        e = e.merge(keys, on=["storm_id", "t0"])
        wv = e.pivot_table(index=["storm_id", "t0", "tech"], columns="rel", values="vmax")
        wp = e.pivot_table(index=["storm_id", "t0", "tech"], columns="rel", values="mslp")
        d = _long(wv, wp).rename(columns={"tech": "member"})
        d["age"] = age
        parts.append(d)
        have = keys if have is None else pd.concat([have, keys])
    return pd.concat(parts, ignore_index=True)


def nhc(model: str) -> pd.DataFrame:
    members, ages = NHC_MEMBERS[model]
    d = _aged(_decks(members), ages)
    d["source"], d["model"] = "nhc", model
    return d


def ecmwf_tf() -> pd.DataFrame:
    """ECMWF ENS perturbed members from ECMWF's own BUFR tracks (data/interim/ecmwf_tf.parquet: 2023-01 on, all four runs, public in
    real time), plus NHC's archived copies of the same tracks (EE01-50, identical values, mostly 00/12Z runs) for runs the BUFR set
    lacks. Newest run at least 6 h old: 6 h when the 06/18Z runs exist (2023 on), else 12 or 18 h."""
    t = pd.read_parquet(INTERIM_DIR / "ecmwf_tf.parquet", filters=[("ftype", "==", "pf")], columns=["storm_id", "init", "member", "lead", "vmax", "pmin"])
    t = t.assign(tech="EE" + t["member"].astype(str).str.zfill(2), tau=t["lead"].astype(int)).rename(columns={"pmin": "mslp"})
    t = t[["storm_id", "init", "tech", "tau", "vmax", "mslp"]]
    a = _decks(NHC_MEMBERS["ecmwf"][0])
    a = a.merge(t[["storm_id", "init"]].drop_duplicates(), how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    d = _aged(pd.concat([t, a], ignore_index=True), (6, 12, 18))
    d["source"], d["model"] = "tf", "ecmwf"
    return d


def pilot(model: str) -> pd.DataFrame:
    age = PILOT_AGE[model]
    files = sorted((PILOT_DIR / model).glob("*/done.json"))
    t = pd.concat([pd.read_parquet(f.parent / "tracks.parquet") for f in files], ignore_index=True)
    t["t0"] = t["init"] + pd.Timedelta(hours=age)
    t["rel"] = t["lead"] - age
    t = t[t["rel"].isin(NEEDED)]
    for v in CORE_VARS[model]:
        t[f"cr_{v}"] = t[f"core_{v}"] - t[f"ring_{v}"]
    idx = ["storm_id", "t0", "member"]
    wv = t.pivot_table(index=idx, columns="rel", values="vmax")
    wp = t.pivot_table(index=idx, columns="rel", values="pmin")
    d = _long(wv, wp)
    struct = ["rmw_km", "depth", "gust"] + [f"cr_{v}" for v in CORE_VARS[model]]
    s = t[t["rel"].isin(LEADS)][idx + ["rel"] + struct].rename(columns={"rel": "lead"})
    d = d.merge(s, on=idx + ["lead"], how="left")
    env = environment(model)
    env["t0"] = env["init"] + pd.Timedelta(hours=age)
    env = env.assign(lead=env["lead"] - age).drop(columns=["init", "env_member"])
    d = d.merge(env, on=["storm_id", "t0", "lead"], how="left")
    d["member"] = d["member"].astype(str)
    d["age"], d["source"], d["model"] = age, "pilot", model
    return d


def finish(d: pd.DataFrame) -> pd.DataFrame:
    """Add ensemble context, storm state, targets; drop rows without a usable CARQ state."""
    g = d.groupby(["storm_id", "t0", "lead"])["v"]
    d["ens_mean"], d["ens_std"], d["ens_n"] = g.transform("mean"), g.transform("std"), g.transform("count")
    d["ens_rank"] = g.rank(pct=True)
    d["z"] = (d["v"] - d["ens_mean"]) / d["ens_std"].replace(0, np.nan)
    d["dv12_own"], d["dv24_own"] = d["v"] - d["v_m12"], d["v"] - d["v_m24"]
    d["ri_own"] = (d["dv24_own"] >= RI_THRESHOLD_KT).astype(float).where(d["dv24_own"].notna())
    d["ens_ri_frac"] = d.groupby(["storm_id", "t0", "lead"])["ri_own"].transform("mean")
    rows = pd.read_parquet(FEATURES_DIR / "consensus_features.parquet", columns=ROW)
    d = d.merge(rows, on=["storm_id", "t0", "lead"], how="inner")
    d["off0"] = d["v_t0"] - d["vmax0"]  # member's intensity at t0 minus the operational (CARQ) intensity
    d["v_shift"] = d["v"] - d["off0"]  # member intensity shifted to match CARQ at t0
    return d


def build(source: str, model: str) -> pd.DataFrame:
    d = finish({"nhc": nhc, "pilot": pilot, "tf": lambda m: ecmwf_tf()}[source](model))
    d.to_parquet(FEATURES_DIR / f"members_{source}_{model}.parquet", index=False)
    return d


if __name__ == "__main__":
    for source, model in (("nhc", "gefs"), ("nhc", "ecmwf"), ("pilot", "ifs"), ("pilot", "gefs")):
        d = build(source, model)
        v = d[d["verifiable"]]
        print(f"{source}/{model}: {len(d):,} member rows ({len(v):,} verifiable), {v[['storm_id', 't0']].drop_duplicates().shape[0]} storm-cycles, "
              f"members/case {v['ens_n'].median():.0f}, seasons {sorted(v['season'].unique())}, raw member MAE {(v['v'] - v['obs_vmax']).abs().mean():.1f} kt, "
              f"ens-mean bias {(v['ens_mean'] - v['obs_vmax']).mean():+.1f} kt")
