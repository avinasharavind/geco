"""NHC-GEFS CCSR season by season, 2018-2026: is the gain over the raw GEFS members (and the point baselines) stable across years?

CCSH: the same with HCCA as the centre (HCCA is in the public real-time a-decks); GECO: HCCA centre without lead smoothing.
For 2018-24 every ingredient is out of sample for the season being scored: the consensus centre (real-time M1.5 variant A) and the
predicted spread sigma_hat are leave-one-season-out already, and k(tau) is refit here leaving that season out too (the production
CCSR fits k on all of 2018-24). 2025-26 use the production models (trained on 2018-24). Model choices (smoothing widths, the CCSR
design) were made looking at 2025, so 2025 is not perfectly clean either.
GEFS changed under us: v11 (20 members, AP01-20) until 2020-09, v12 (30 members) after; NHC's member tracker changed with it.
Point baselines (HCCA, IVCN, OFCL, the consensus centre) are scored with CRPS = absolute error, so all rows share one scale.
Writes data/results/ensemble/multiseason_nhc_gefs.parquet and docs/ensemble_multiseason.md.
"""
import math
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, INTERIM_DIR, ROOT
from tcic.data.atcf import merge_interpolated
from tcic.ensemble import calib, geco
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
SEASONS = list(range(2018, 2027))
POINT = ["HCCA", "IVCN", "OFCL"]
t_start = time.time()

d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
ones = sig * 0 + 1.0
oof, test = pd.read_parquet(R / "oof_nhc_gefs.parquet"), pd.read_parquet(R / "test_nhc_gefs.parquet")
base = ["RAW", "v_shift"]
allm = pd.concat([oof, test[oof.columns.intersection(test.columns)]], ignore_index=True)
allm = allm[allm["season"].isin(SEASONS)]


A = pd.read_parquet(INTERIM_DIR / "adecks.parquet", filters=[("tech", "in", POINT + ["HCC2", "IVC2", "OFC2"])], columns=["storm_id", "init", "tech", "tau", "vmax"])
A["storm_id"] = A["storm_id"].astype(str)
A = merge_interpolated(A)
A = A[A["tech"].isin(POINT)].rename(columns={"init": "t0", "tau": "lead"}).drop_duplicates(CASE + ["tech"])
H_HCCA = geco.hcca()
hcca_centre = lambda t: geco.hcca_centre(t, H_HCCA)


def built(members: pd.DataFrame, c0: float, c1: float, name: str, sig_c: float) -> pd.DataFrame:
    dev, cen = smo.prepare(members, 6.0, sig_c)
    z = calib.standardised_smooth(dev)
    z = z[z.index.droplevel("member").isin(sig.index)]
    out = calib.members_q(z, cen, ones, sig, sig, c0, c1, name=name)
    out[f"{name}80"] = calib.members_q(z, cen, ones * 0.8, sig, sig, c0, c1, name="x")["x"].to_numpy()
    return out


VARIANTS = {"CCSR": (lambda t: t, 12.0), "CCSH": (hcca_centre, 12.0), "GECO": (hcca_centre, 0.0)}  # centre, its lead smoothing (h)
parts, ks = [], []
for name, (centre, sig_c) in VARIANTS.items():
    o = centre(oof[oof["season"].isin(calib.TRAIN)])
    k_all = calib.fit_k_q(o, ones, sig, sig, 6.0, sig_c)
    vp = []
    for s in SEASONS:
        crps, c0, c1 = calib.fit_k_q(o[o["season"] != s], ones, sig, sig, 6.0, sig_c) if s in calib.TRAIN else k_all
        g = centre(allm[allm["season"] == s])
        vp.append(g[KEY + ["member", "lead"]].merge(built(g, c0, c1, name, sig_c), on=KEY + ["member", "lead"]))
        ks.append({"variant": name, "season": s, "c0": c0, "c1": c1})
        print(f"{name} {s}: k = {c0:.2f} + {c1:.2f} τ/100 ({time.time() - t_start:.0f} s)", flush=True)
    parts.append(pd.concat(vp, ignore_index=True))
m = allm.merge(parts[0], on=KEY + ["member", "lead"]).merge(parts[1], on=KEY + ["member", "lead"]).merge(parts[2], on=KEY + ["member", "lead"])
m = m[m["lead"].isin(LEADS) & m["obs_vmax"].notna()].dropna(subset=["CCSR", "CCSH", "GECO"])
m[CASE + ["member", "season", "obs_vmax", "tcca_mean"] + base + ["CCSR", "CCSR80", "CCSH", "CCSH80", "GECO", "GECO80"]].to_parquet(R / "multiseason_nhc_gefs.parquet", index=False)

# per-case scores; point forecasts (CRPS = |error|) on the same cases
S = {v: sc.ensemble_case_scores(m, v) for v in base + ["CCSR", "CCSR80", "CCSH", "CCSH80", "GECO", "GECO80"]}
obs = m.drop_duplicates(CASE)[CASE + ["obs_vmax", "tcca_mean"]]
for tech in POINT + ["centre"]:
    p = obs.rename(columns={"tcca_mean": "fc"}) if tech == "centre" else obs.merge(A[A["tech"] == tech][CASE + ["vmax"]].rename(columns={"vmax": "fc"}), on=CASE)
    S[tech] = p.assign(obs=p["obs_vmax"], mean=p["fc"], crps=(p["fc"] - p["obs_vmax"]).abs(), std=0.0, inside=np.nan)[CASE + ["obs", "mean", "crps", "std", "inside"]]
for v in ("CCSR", "CCSH", "GECO"):
    S[f"{v} mean"] = S[v].assign(crps=S[v]["abs_err_mean"], std=0.0, inside=np.nan)[CASE + ["obs", "mean", "crps", "std", "inside"]]


def dressed(p: pd.DataFrame) -> pd.DataFrame:
    """A point forecast made probabilistic: Gaussian with a per-lead spread equal to its RMS error in the other seasons."""
    p = p.assign(season=p["t0"].dt.year, e=p["mean"] - p["obs"])
    sq = p.groupby(["season", "lead"])["e"].agg(lambda e: (e ** 2).sum()).unstack("lead")
    n = p.groupby(["season", "lead"]).size().unstack("lead")
    train = [s for s in calib.TRAIN if s in sq.index]
    sd = {s: np.sqrt(sq.loc[[t for t in train if t != s]].sum() / n.loc[[t for t in train if t != s]].sum()) for s in sq.index}
    sigma = np.array([sd[s][L] for s, L in zip(p["season"], p["lead"])])
    zz = (p["obs"] - p["mean"]) / sigma
    phi, Phi = np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), 0.5 * (1 + np.vectorize(math.erf)(zz / np.sqrt(2)))
    crps = sigma * (zz * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    inside = (np.abs(zz) <= 1.2816).astype(float)
    return p.assign(crps=crps, std=sigma, inside=inside)[CASE + ["obs", "mean", "crps", "std", "inside"]]


S["HCCA dressed"], S["OFCL dressed"] = dressed(S["HCCA"]), dressed(S["OFCL"])
keys = None
for s in S.values():
    keys = s[CASE].dropna() if keys is None else keys.merge(s[CASE], on=CASE)
H = {k: s.merge(keys, on=CASE).assign(season=lambda x: x["t0"].dt.year) for k, s in S.items()}
NAMES = {"RAW": "raw members", "v_shift": "raw, offset to t0", "CCSR": "CCSR", "CCSR80": "CCSR80", "centre": "consensus centre (point)",
         "HCCA": "HCCA (point)", "IVCN": "IVCN (point)", "OFCL": "OFCL (point)",
         "CCSR mean": "CCSR mean (point)", "CCSH": "CCSH (HCCA centre)", "CCSH80": "CCSH80", "CCSH mean": "CCSH mean (point)", "GECO": "GECO (HCCA centre, unsmoothed)", "GECO80": "GECO80", "GECO mean": "GECO mean (point)", "HCCA dressed": "HCCA dressed", "OFCL dressed": "OFCL dressed"}


def ci(a, b, sub):
    dd, lo, hi = sc.storm_bootstrap_diff(H[a][sub(H[a])], H[b][sub(H[b])], "crps")
    return f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if hi < 0 or lo > 0 else "")


rows = []
members = m.groupby("season")["member"].nunique()
for label, sub in [(str(s), lambda x, s=s: x["season"] == s) for s in SEASONS] + [("2018–24", lambda x: x["season"] <= 2024), ("2025–26", lambda x: x["season"] >= 2025), ("all", lambda x: x["season"] > 0)]:
    c = H["CCSR"][sub(H["CCSR"])]
    err = c["mean"] - c["obs"]
    row = {"season": label, "storms": c["storm_id"].nunique(), "cases": len(c),
           "members": members.get(int(label), np.nan) if label.isdigit() else np.nan}
    row.update({NAMES[k]: h[sub(h)]["crps"].mean() for k, h in H.items()})
    row.update({"CCSR coverage": c["inside"].mean(), "raw coverage": H["v_shift"][sub(H["v_shift"])]["inside"].mean(),
                "CCSR spread/error": c["std"].mean() / np.sqrt((err ** 2).mean()), "CCSR bias": err.mean(),
                "CCSR − raw (offset)": ci("CCSR", "v_shift", sub), "CCSR − HCCA dressed": ci("CCSR", "HCCA dressed", sub),
                "CCSR − OFCL dressed": ci("CCSR", "OFCL dressed", sub), "CCSR mean − HCCA (MAE)": ci("CCSR mean", "HCCA", sub),
                "CCSH coverage": H["CCSH"][sub(H["CCSH"])]["inside"].mean(), "CCSH − CCSR": ci("CCSH", "CCSR", sub),
                "CCSH − HCCA dressed": ci("CCSH", "HCCA dressed", sub), "CCSH − OFCL dressed": ci("CCSH", "OFCL dressed", sub),
                "CCSH80 − CCSH": ci("CCSH80", "CCSH", sub), "GECO coverage": H["GECO"][sub(H["GECO"])]["inside"].mean(),
                "GECO − CCSH": ci("GECO", "CCSH", sub), "GECO − CCSR": ci("GECO", "CCSR", sub), "GECO − HCCA dressed": ci("GECO", "HCCA dressed", sub),
                "GECO mean − HCCA (MAE)": ci("GECO mean", "HCCA", sub),
                "CCSR − centre": ci("CCSR", "centre", sub)})
    rows.append(row)
T = pd.DataFrame(rows).set_index("season")
T.to_csv(R / "multiseason_nhc_gefs_table.csv")
lead = pd.concat({NAMES[k]: h.groupby("lead")["crps"].mean() for k, h in H.items()}, axis=1).T
by_season_lead = H["CCSR"].groupby(["season", "lead"])["crps"].mean().unstack("lead")

score_cols = [NAMES[k] for k in H]
txt = ("# NHC-GEFS CCSR across seasons, 2018–2026\n\nCCSH = CCSR with the centre swapped for HCCA (NHC's early corrected consensus, public in real time); same members and predicted spread, k refit leaving the season out. GECO = CCSH without the 12-h lead smoothing of the centre (HCCA used as issued); the chosen GEFS product (src/tcic/ensemble/geco.py).\n\n"
       "Fair CRPS (kt; point forecasts scored as |error|) on identical cases: every NHC synoptic cycle with NHC's GEFS member tracks "
       "(newest run ≥ 6 h old), HCCA, IVCN and OFCL all present; “dressed” = the point forecast as a Gaussian with its own per-lead RMS error from the other 2018–24 seasons (the fair probabilistic baseline: an ensemble always beats its own point forecast in CRPS); leads 12–120 h every 12 h; best track (2026: working best track). "
       "2018–24 are fully out of sample (consensus centre, spread model and k all leave-one-season-out); 2025–26 use the production models. "
       "GEFS v11 (20 members) until 2020-09, v12 (30) after. `*` = 95% storm-block bootstrap CI excludes 0.\n\n"
       "## Scores by season\n\n" + T[["storms", "cases", "members"] + score_cols].to_markdown(floatfmt=".2f")
       + "\n\n## Calibration and differences (CCSR)\n\n"
       + T[["CCSR coverage", "raw coverage", "CCSR spread/error", "CCSR bias", "CCSR − raw (offset)", "CCSR − centre", "CCSR − HCCA dressed", "CCSR − OFCL dressed", "CCSR mean − HCCA (MAE)"]].to_markdown(floatfmt=".2f")
       + "\n\n## HCCA-centred version (CCSH)\n\n" + T[["CCSH coverage", "CCSH − CCSR", "CCSH − HCCA dressed", "CCSH − OFCL dressed", "CCSH80 − CCSH"]].to_markdown(floatfmt=".2f")
       + "\n\n## HCCA centre without lead smoothing (GECO)\n\n" + T[["GECO coverage", "GECO − CCSH", "GECO − CCSR", "GECO − HCCA dressed", "GECO mean − HCCA (MAE)"]].to_markdown(floatfmt=".2f")
       + "\n\nCoverage = share of cases inside the 10–90% range (ideal 0.80).\n\n## CRPS by lead, all seasons\n\n" + lead.to_markdown(floatfmt=".1f")
       + "\n\n## CCSR CRPS by season and lead\n\n" + by_season_lead.to_markdown(floatfmt=".1f")
       + "\n\n## k(τ) = c0 + c1 τ/100 per held-out season\n\n" + pd.DataFrame(ks).set_index(["variant", "season"]).to_markdown(floatfmt=".2f")
       + f"\n\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_multiseason.md").write_text(txt)
print(txt)
