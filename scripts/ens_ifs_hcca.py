"""IFS-ENS member shapes around HCCA (built exactly like GECO), and GEFS + IFS-ENS shapes pooled around HCCA, against GECO, CCSG (the gridded-centre IFS-ENS ensemble),
FNV3 early and dressed HCCA on identical 12Z cycles, 2025-26. Question: do IFS-ENS scenario shapes add anything once the centre is HCCA?

k(tau) for the new variants is fit on 2024 out-of-fold cases (the only out-of-fold IFS-ENS season); GECO keeps its own k (2018-24).
Members on each side are standardised within their own ensemble, so the pooled set weights each model's scenarios by member count
(30 GEFS + 51 IFS-ENS). The "IFS-ENS around HCCA" variant became IECO (scripts/ens_geco.py). Nothing existing is modified; writes docs/ensemble_ifs_hcca.md.
"""
import math
import time
import warnings

import numpy as np
import pandas as pd

from tcic.config import DATA_DIR, ROOT, TEST_LABEL, TEST_SEASONS
from tcic.data import fnv3 as fnv3_data
from tcic.ensemble import calib, geco
from tcic.ensemble import score as sc
from tcic.ensemble import smooth as smo

warnings.filterwarnings("ignore")
R = DATA_DIR / "results" / "ensemble"
CASE, KEY = sc.CASE, smo.KEY
LEADS = list(range(12, 121, 12))
t_start = time.time()
d = pd.read_parquet(DATA_DIR / "results" / "calibration" / "spread_ri_table.parquet")
sig = calib.smooth_leads(d.pivot_table(index=KEY, columns="lead", values="sigma_hat"), 18.0)
ones = sig * 0 + 1.0
h = geco.hcca()


def shapes(t: pd.DataFrame, tag: str):
    """z (standardised, anchored, smoothed member deviations) and the HCCA centre, members prefixed by tag."""
    dev, cen = smo.prepare(geco.hcca_centre(t[t["t0"].dt.hour == 12], h), geco.SIG_DEV, geco.SIG_CEN)
    z = calib.standardised_smooth(dev)
    z = z[z.index.droplevel("member").isin(sig.index)]
    z.index = z.index.set_levels(z.index.levels[2].astype(str).map(lambda m: f"{tag}{m}"), level=2)
    return z, cen


def fit_k(z: pd.DataFrame, cen: pd.DataFrame, obs: pd.Series):
    best = None
    for c0 in np.linspace(0.6, 2.0, 15):
        for c1 in np.linspace(-0.4, 0.4, 9):
            mm = calib.members_q(z, cen, ones, sig, sig, c0, c1, name="X").join(obs, on=KEY + ["lead"]).dropna(subset=["obs_vmax", "X"])
            crps = sc.ensemble_case_scores(mm, "X")["crps"].mean()
            if best is None or crps < best[0]:
                best = (crps, c0, c1)
    return best


src = {n: (pd.read_parquet(R / f"oof_{n}.parquet"), pd.read_parquet(R / f"test_{n}.parquet")) for n in ("nhc_gefs", "pilot_ifs")}
obs_all = pd.concat([pd.concat(v) for v in src.values()]).drop_duplicates(CASE).set_index(CASE)["obs_vmax"]
fits, built = {}, {}
for label, parts in {"IFS-ENS around HCCA": [("pilot_ifs", "I")], "GEFS + IFS-ENS around HCCA": [("nhc_gefs", "G"), ("pilot_ifs", "I")]}.items():
    zs_tr, zs_te = [], []
    for n, tag in parts:
        oof, test = src[n]
        z, cen_tr = shapes(oof[oof["season"] == 2024], tag)
        zs_tr.append(z)
        z, cen_te = shapes(test, tag)
        zs_te.append(z)
    both_tr = set.intersection(*[set(z.index.droplevel("member").unique()) for z in zs_tr])
    both_te = set.intersection(*[set(z.index.droplevel("member").unique()) for z in zs_te])
    ztr = pd.concat([z[z.index.droplevel("member").isin(both_tr)] for z in zs_tr]).sort_index()
    zte = pd.concat([z[z.index.droplevel("member").isin(both_te)] for z in zs_te]).sort_index()
    crps, c0, c1 = fit_k(ztr, cen_tr, obs_all)
    fits[label] = (c0, c1)
    built[label] = calib.members_q(zte, cen_te, ones, sig, sig, c0, c1, name="X").join(obs_all, on=KEY + ["lead"])
    print(f"{label}: k = {c0:.2f} {c1:+.2f} τ/100 (2024 OOF CRPS {crps:.2f}; {time.time() - t_start:.0f} s)", flush=True)

g, i = src["nhc_gefs"][1], src["pilot_ifs"][1]
g, i = g[g["t0"].dt.hour == 12], i[i["t0"].dt.hour == 12]
S = {"GECO (GEFS around HCCA)": sc.ensemble_case_scores(g[g["lead"].isin(LEADS)].dropna(subset=["GECO", "obs_vmax"]), "GECO"),
     "CCSG (IFS-ENS around gridded consensus)": sc.ensemble_case_scores(i[i["lead"].isin(LEADS)].dropna(subset=["CCSG", "obs_vmax"]), "CCSG")}
for k, m in built.items():
    S[k] = sc.ensemble_case_scores(m[m["lead"].isin(LEADS)].dropna(subset=["X", "obs_vmax"]), "X")
# FNV3 early: newest run 6-24 h old
fn = fnv3_data.load_test()
c = i[KEY].drop_duplicates().merge(fn[["storm_id", "init"]].drop_duplicates(), on="storm_id")
c = c[(c["init"] <= c["t0"] - pd.Timedelta(hours=6)) & (c["init"] >= c["t0"] - pd.Timedelta(hours=24))].sort_values("init").groupby(KEY).tail(1)
f = c.merge(fn, on=["storm_id", "init"])
f["lead"] = f["lead"] - ((f["t0"] - f["init"]) / pd.Timedelta(hours=1)).astype(int)
f = f[f["lead"].isin(LEADS)].join(obs_all, on=CASE).dropna(subset=["obs_vmax"])
S["FNV3 early"] = sc.ensemble_case_scores(f, "vmax")
# dressed HCCA: Gaussian with HCCA's 2018-24 RMS error by lead (all cycles)
hh = h.join(obs_all, on=CASE).dropna()
hh["e"] = hh["hcca"] - hh["obs_vmax"]
sd = hh[hh["t0"].dt.year <= 2024].groupby("lead")["e"].apply(lambda e: np.sqrt((e ** 2).mean()))
p = hh[hh["t0"].dt.year.isin(TEST_SEASONS) & (hh["t0"].dt.hour == 12) & hh["lead"].isin(LEADS)].copy()
s_ = p["lead"].map(sd).to_numpy()
zz = (p["obs_vmax"] - p["hcca"]).to_numpy() / s_
Phi = 0.5 * (1 + np.vectorize(math.erf)(zz / np.sqrt(2)))
p["crps"] = s_ * (zz * (2 * Phi - 1) + 2 * np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
S["HCCA dressed"] = p.assign(obs=p["obs_vmax"], mean=p["hcca"], std=s_, inside=(np.abs(zz) <= 1.2816).astype(float))[CASE + ["obs", "mean", "crps", "std", "inside"]]

keys = None
for s in S.values():
    keys = s[CASE] if keys is None else keys.merge(s[CASE], on=CASE)
H = {k: s.merge(keys, on=CASE) for k, s in S.items()}
rows = []
for label, years in ((TEST_LABEL, list(TEST_SEASONS)), ("2025", [2025]), ("2026", [2026])):
    for k, hk in H.items():
        hk = hk[hk["t0"].dt.year.isin(years)]
        err = hk["mean"] - hk["obs"]
        row = {"sample": label, "ensemble": k, "CRPS": hk["crps"].mean(), "MAE of mean": err.abs().mean(), "bias": err.mean(),
               "coverage": hk["inside"].mean(), "spread/error": hk["std"].mean() / np.sqrt((err ** 2).mean())}
        for ref in ("GECO (GEFS around HCCA)", "HCCA dressed"):
            if k != ref:
                ref_h = H[ref][H[ref]["t0"].dt.year.isin(years)]
                dd, lo, hi = sc.storm_bootstrap_diff(hk, ref_h, "crps")
                row[f"− {ref.split(' (')[0]}"] = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if hi < 0 or lo > 0 else "")
        rows.append(row)
lead = pd.concat({k: hk.groupby("lead")["crps"].mean() for k, hk in H.items()}, axis=1).T
txt = ("# IFS-ENS shapes around HCCA, alone and pooled with GEFS (12Z cycles, " + TEST_LABEL + ")\n\n"
       "All HCCA-centred variants use GECO's construction (anchored, 6-h-smoothed, standardised member deviations × k × σ̂ around HCCA as "
       "issued, full spread). GEFS members are NHC's tracks from the 06Z run; IFS-ENS members are ours from the 00Z run. k for the new "
       "variants fit on 2024 out-of-fold cases: " + "; ".join(f"{k}: {a:.2f} {b:+.2f} τ/100" for k, (a, b) in fits.items())
       + f". Identical cases ({keys['storm_id'].nunique()} storms, {len(keys)} cases), leads 12–120 h, 95% storm-block bootstrap; `*` = CI excludes 0.\n\n"
       + pd.DataFrame(rows).set_index(["sample", "ensemble"]).to_markdown(floatfmt=".2f") + "\n\n## CRPS by lead, " + TEST_LABEL + "\n\n"
       + lead.to_markdown(floatfmt=".1f") + f"\n\n_Run time {time.time() - t_start:.0f} s._\n")
(ROOT / "docs" / "ensemble_ifs_hcca.md").write_text(txt)
print(txt)
