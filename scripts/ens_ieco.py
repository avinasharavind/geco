"""IECO from ECMWF's own ENS tracks: the GECO construction (src/tcic/ensemble/geco.py) on the 50 perturbed IFS-ENS members, 2018-2026.

Members: members.ecmwf_tf() - ECMWF's BUFR tracks (public in real time on the AWS open-data mirror, scripts/fetch_ecmwf_tracks.py),
with NHC's archived copies (EE01-50, identical) for earlier runs. Newest run at least 6 h old: 6 h at every cycle from 2023 (the 06/18Z
runs are posted ~6 h after their start, the 00/12Z runs ~7-7.6 h, so before each advisory), 12 or 18 h before (00/12Z runs only).
Centre HCCA as issued, spread sigma_hat, k(tau) refit leaving the scored season out for 2018-24 (as docs/ensemble_multiseason.md);
2025-26 use k fit on all of 2018-24. Cases without HCCA are dropped.

Yardsticks: dressed HCCA (HCCA as a Gaussian with its per-lead RMS error from the other 2018-24 seasons) on all nine seasons, and
FNV3 at early timing (newest run 6-24 h old) on 2025-26. GECO (same cases, from multiseason_nhc_gefs.parquet) is a reference column.
Writes data/features/members_tf_ecmwf.parquet, data/results/ensemble/ieco_multiseason.parquet, docs/ensemble_ieco.md.
"""
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, ROOT
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import calib, geco, members
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
SEASONS = list(range(2018, 2027))
t_start = time.time()

d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
TCCA = pd.concat([pd.read_parquet(DATA_DIR / "results" / "m15" / f) for f in ("oof_vmax_A.parquet", "test_vmax_A.parquet")])
TCCA = TCCA[["storm_id", "t0", "lead", "mean"]].rename(columns={"mean": "tcca_mean"})
H = geco.hcca()

m = members.build("tf", "ecmwf")
m = geco.hcca_centre(m.merge(TCCA, on=CASE), H)
m = m[m["season"].isin(SEASONS)]
print(f"members: {len(m):,} rows, {m[KEY].drop_duplicates().shape[0]:,} cycles ({time.time() - t_start:.0f} s)", flush=True)

parts, ks = [], []
train = m[m["season"].isin(calib.TRAIN)]
k_all = geco.fit_k(train, sig)
for s in SEASONS:
    crps, c0, c1 = geco.fit_k(train[train["season"] != s], sig) if s in calib.TRAIN else k_all
    g = m[m["season"] == s]
    parts.append(geco.build(g, sig, c0, c1, name="IECO"))
    ks.append({"season": s, "c0": c0, "c1": c1})
    print(f"{s}: k = {c0:.2f} {c1:+.2f} τ/100 ({time.time() - t_start:.0f} s)", flush=True)
cols = CASE + ["member", "season", "age", "obs_vmax", "v", "v_shift", "v_t0", "vmax0", "tcca_mean"]
ie = m[cols].merge(pd.concat(parts, ignore_index=True), on=KEY + ["member", "lead"])
ie = ie[ie["lead"].isin(LEADS) & ie["obs_vmax"].notna()].dropna(subset=["IECO"])
ie.to_parquet(R / "ieco_multiseason.parquet", index=False)

S = {"IECO": sc.ensemble_case_scores(ie, "IECO"), "raw ECMWF members (offset to t0)": sc.ensemble_case_scores(ie, "v_shift")}
g = pd.read_parquet(R / "multiseason_nhc_gefs.parquet", columns=CASE + ["member", "obs_vmax", "GECO"])
S["GECO (reference)"] = sc.ensemble_case_scores(g, "GECO")


obs = ie.drop_duplicates(CASE)[CASE + ["obs_vmax"]]
hp = obs.merge(H, on=CASE)
S["HCCA dressed"] = geco.dressed(hp.assign(obs=hp["obs_vmax"], mean=hp["hcca"]))

# FNV3 at early timing (newest run 6-24 h old, leads shifted to t0), where available (2025-26)
fn = fnv3_data.load_test()
c = obs[KEY].drop_duplicates().merge(fn[["storm_id", "init"]].drop_duplicates(), on="storm_id")
c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))].sort_values("init").groupby(KEY).tail(1)
f = c.merge(fn, on=["storm_id", "init"])
f["lead"] = f["lead"] - ((f["t0"] - f["init"]) / pd.Timedelta(hours=1)).astype(int)
f = f[f["lead"].isin(LEADS)].merge(obs, on=CASE)
S["FNV3 early"] = sc.ensemble_case_scores(f, "vmax")

age = ie.drop_duplicates(CASE)[CASE + ["age"]]


def common(names) -> pd.DataFrame:
    keys = None
    for k in names:
        keys = S[k][CASE] if keys is None else keys.merge(S[k][CASE], on=CASE)
    return keys


def ci(a: pd.DataFrame, b: pd.DataFrame) -> str:
    dd, lo, hi = sc.storm_bootstrap_diff(a, b, "crps")
    return f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if hi < 0 or lo > 0 else "")


def table(names, samples, refs) -> pd.DataFrame:
    keys = common(names)
    Hs = {k: S[k].merge(keys, on=CASE).merge(age, on=CASE).assign(season=lambda x: x["t0"].dt.year) for k in names}
    rows = []
    for label, sub in samples:
        x = {k: h[sub(h)] for k, h in Hs.items()}
        i = x["IECO"]
        err = i["mean"] - i["obs"]
        row = {"sample": label, "storms": i["storm_id"].nunique(), "cases": len(i), "6-h-old share": (i["age"] == 6).mean()}
        row.update({f"CRPS {k}": h["crps"].mean() for k, h in x.items()})
        row.update({"IECO coverage": i["inside"].mean(), "IECO spread/error": i["std"].mean() / np.sqrt((err ** 2).mean()), "IECO bias": err.mean()})
        row.update({f"IECO − {r}": ci(i, x[r]) for r in refs})
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample"), Hs


by_season = [(str(s), lambda x, s=s: x["season"] == s) for s in SEASONS]
groups = [("2018–24", lambda x: x["season"] <= 2024), ("2025–26", lambda x: x["season"] >= 2025), ("all 2018–26", lambda x: x["season"] > 0),
          ("run 6 h old", lambda x: x["age"] == 6), ("run 12–18 h old", lambda x: x["age"] > 6)]
T1, H1 = table(["IECO", "HCCA dressed", "GECO (reference)", "raw ECMWF members (offset to t0)"], by_season + groups, ["HCCA dressed", "GECO (reference)"])
T2, H2 = table(["IECO", "HCCA dressed", "FNV3 early", "GECO (reference)"],
               [("2025", lambda x: x["season"] == 2025), ("2026", lambda x: x["season"] == 2026), ("2025–26", lambda x: x["season"] >= 2025)],
               ["HCCA dressed", "FNV3 early", "GECO (reference)"])
lead1 = pd.concat({k: h.groupby("lead")["crps"].mean() for k, h in H1.items()}, axis=1).T
lead2 = pd.concat({k: h.groupby("lead")["crps"].mean() for k, h in H2.items()}, axis=1).T
T1.to_csv(R / "ieco_multiseason_table.csv")
T2.to_csv(R / "ieco_fnv3_table.csv")

txt = ("# IECO from ECMWF's own ensemble tracks, 2018–2026\n\n"
       "IECO = the GECO construction (member shapes × k(τ) × σ̂ around HCCA as issued, full spread) on the 50 perturbed IFS-ENS members, "
       "using ECMWF's own tracker output: the BUFR track files on the AWS open-data mirror (public in real time; 2023 on, all four runs) and "
       "NHC's archived copies of the same tracks (EE01–50) before. Newest run at least 6 h old (6 h at every cycle from 2023; 12–18 h before). "
       "k(τ) refit leaving the scored season out for 2018–24; 2025–26 use k fit on 2018–24. No gridded data.\n\n"
       "Yardsticks: dressed HCCA (HCCA as a Gaussian with its per-lead RMS error from the other 2018–24 seasons) and FNV3 at early timing "
       "(newest run 6–24 h old; 2025–26 only). GECO is shown for reference: it is IECO's twin (GEFS shapes), not a competitor. "
       "Identical cases within each table, leads 12–120 h every 12 h, all cycles; best track (2026: working best track). Fair CRPS in kt; "
       "`*` = 95% storm-block bootstrap CI excludes 0. Coverage = share inside the 10–90% range (ideal 0.80).\n\n"
       "## Against dressed HCCA, by season\n\n" + T1.to_markdown(floatfmt=".2f")
       + "\n\n## Against FNV3 (early), 2025–26\n\n" + T2.to_markdown(floatfmt=".2f")
       + "\n\n## CRPS by lead, all 2018–26\n\n" + lead1.to_markdown(floatfmt=".1f")
       + "\n\n## CRPS by lead, 2025–26 with FNV3\n\n" + lead2.to_markdown(floatfmt=".1f")
       + "\n\n## k(τ) = c0 + c1 τ/100 per scored season\n\n" + pd.DataFrame(ks).set_index("season").to_markdown(floatfmt=".2f")
       + f"\n\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_ieco.md").write_text(txt)
print(txt)
