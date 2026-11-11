"""M1 sanity checks: reproduce NHC's 2024 Atlantic intensity verification and the paper's 2025 NA+EP OFCL/FNV3 numbers."""
import pandas as pd

from tcic import evaluate as ev
from tcic.config import INTERIM_DIR
from tcic.data.atcf import merge_interpolated

pd.set_option("display.width", 200)
A = pd.read_parquet(INTERIM_DIR / "adecks.parquet")
A["storm_id"] = A["storm_id"].astype(str)
A = merge_interpolated(A)  # 6-h and 12-h interpolated aids count as one early aid, as in NHC verification
A["tech"] = A["tech"].astype(str).replace({"AVNI": "GFSI"})  # NHC report name for the interpolated GFS

# NHC 2024 report, Table 4 (OFCL, all Atlantic TCs) and Table 5a (homogeneous early models)
NHC_T4 = {"mae": [5.4, 8.0, 9.9, 11.4, 12.1, 13.1, 13.7, 12.8], "bias": [0.5, -0.1, -1.5, -2.5, -3.0, -2.7, -0.2, 4.4],
          "n": [311, 276, 244, 214, 185, 157, 112, 79]}
NHC_T5A = {
    "OFCL": [5.4, 8.0, 9.9, 11.4, 12.0, 13.3, 13.8, 12.8], "OCD5": [8.3, 13.4, 18.4, 22.7, 23.8, 25.4, 28.1, 19.2],
    "HWFI": [7.5, 9.7, 11.4, 12.3, 12.8, 13.2, 15.8, 17.7], "HMNI": [7.2, 9.1, 11.1, 12.1, 12.5, 13.0, 14.9, 15.6],
    "HFAI": [7.4, 10.2, 12.4, 13.9, 14.8, 16.3, 18.6, 17.5], "HFBI": [7.3, 10.1, 11.4, 14.3, 15.8, 16.8, 16.2, 18.8],
    "DSHP": [7.2, 10.3, 13.0, 14.0, 14.8, 16.3, 15.5, 15.3], "LGEM": [7.5, 10.5, 13.1, 14.0, 13.6, 14.3, 14.6, 14.1],
    "IVCN": [6.3, 8.4, 10.1, 11.5, 12.0, 12.6, 13.3, 12.2], "HCCA": [6.0, 7.8, 9.6, 11.3, 12.1, 12.9, 14.4, 14.8],
    "GFSI": [7.6, 11.0, 13.7, 15.6, 16.8, 17.8, 18.7, 17.6], "EMXI": [8.6, 12.8, 16.3, 18.8, 20.3, 21.3, 21.5, 21.6],
    "n": [306, 272, 240, 210, 181, 153, 111, 79]}


def compare(ours: pd.Series, ref: list, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"ours": ours.round(1).values, "NHC": ref}, index=list(ev.NHC_LEADS)).T
    print(f"\n{label}\n{df.to_string()}")
    return df


truth_nhc = ev.load_truth("nhc")
al24 = A[A["storm_id"].str.startswith("AL") & (A["init"].dt.year == 2024)]

# --- Table 4: OFCL, all Atlantic TCs, 2024 ---
c = ev.pair(al24, truth_nhc, ["OFCL"])
t = ev.by_lead(c).loc["OFCL"]
print("=== NHC 2024 Table 4 (OFCL, Atlantic) ===")
compare(t["mae"], NHC_T4["mae"], "MAE (kt)")
compare(t["bias"], NHC_T4["bias"], "bias (kt)")
compare(t["n"].astype(float), NHC_T4["n"], "cases")

# --- Table 5a: homogeneous early models ---
techs = [k for k in NHC_T5A if k != "n"]
c = ev.homogeneous(ev.pair(al24, truth_nhc, techs), techs)
t = ev.by_lead(c, techs)
print("\n=== NHC 2024 Table 5a (homogeneous early models, Atlantic) ===")
rows = []
for tech in techs:
    ours = t.loc[tech, "mae"].reindex(list(ev.NHC_LEADS))
    rows.append(pd.Series((ours.values - pd.Series(NHC_T5A[tech]).values).round(1), index=list(ev.NHC_LEADS), name=tech))
print("our MAE minus NHC MAE (kt):")
print(pd.DataFrame(rows).to_string())
print("cases ours:", t.loc[techs[0], "n"].reindex(list(ev.NHC_LEADS)).tolist(), " NHC:", NHC_T5A["n"])

# --- Paper check: 2025 NA+EP, IBTrACS truth, leads 12-168 h, OFCL vs GDMN (FNV3) homogeneous ---
truth_ib = ev.load_truth("ibtracs")
y25 = A[A["init"].dt.year == 2025]
leads = range(12, 169, 12)
print("\n=== Paper check: 2025 NA+EP, IBTrACS, leads 12-168 h, homogeneous OFCL vs GDMN ===")
for require_tc in (False, True):
    c = ev.homogeneous(ev.pair(y25, truth_ib, ["OFCL", "GDMN"], leads=leads, require_tc=require_tc), ["OFCL", "GDMN"])
    o = ev.overall(c, ["OFCL", "GDMN"])
    print(f"\nrequire TC at both times = {require_tc}  (paper: OFCL 10.7, FNV3 10.7 kt; bias OFCL +0.6, FNV3 -4.5)")
    print(o.round(1).to_string())
    d = ev.mae_difference(c, "OFCL", "GDMN")
    print(f"OFCL - GDMN MAE: {d[0]:+.2f} kt [{d[1]:+.2f}, {d[2]:+.2f}]")
