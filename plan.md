# Early-cycle TC intensity correction for physical models (NA + EP)

## Goal
Allen et al. (2026, "AIFS-TC") showed that a simple learned additive correction to a model's tracker intensity can reach FNV3/OFCL
skill. AIFS has been done; this project asks whether the same idea works for **physical-model guidance**: GEFS, IFS-ENS, GFS, HAFS
and ECMWF. Their raw bias is much smaller, so the gain is uncertain.

Design rules:
- Public data only.
- **Early-cycle only:** use guidance available within ~3 h of synoptic time, like HCCA/IVCN, so it can run in near real time
  and be compared fairly with OFCL.
- North Atlantic and East Pacific.
- Build up in stages: a cheap statistical consensus from ATCF aids first (our own HCCA-style model), then gridded features.
  Probabilistic and multi-model calibration come later.

## Resources and constraints (measured 2026-10-16)
- **Machine:** DGX Spark, shared. aarch64, GB10, 20 cores, 121 GB RAM, 1.7 TB free.
- **Compute cap 40%:** run heavy jobs under `systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=48G`, with `nice`/`ionice`.
- **Network:** internet goes over **Wi-Fi (RedRover, ~287 Mbit/s link)**, measured at ~20–23 MB/s. The 1 Gbps wired port is a private LAN.
  Use at most 85%: throttle downloads to **~17 MB/s** with a token bucket in code. Parallel streams don't raise the ceiling.
- **aarch64 wheels:** `dynamical-catalog`, `icechunk`, `zarr` and `gribberish` all install. They are currently installed in `../tc_prediction/.venv`;
  this project gets its own environment.

## Data sources
| What | Where | Notes |
|---|---|---|
| Best track | IBTrACS v04r01 (NA/EP); NHC b-decks for the current season | vmax = USA_WIND (fallback WMO_WIND/0.88); pmin = USA_PRES (fallback WMO_PRES) |
| Guidance, initial conditions, baselines | NHC ATCF a-decks: `ftp.nhc.noaa.gov/atcf/archive/YYYY/` (archive), `atcf/aid_public` (real time) | CARQ, OFCL, early "I" aids, IVCN, HCCA, DSHP/LGEM, GDMN, AP01–30 |
| SHIPS environment | Real-time SHIPS text, NCAR RAL TC Guidance Project: `hurricanes.ral.ucar.edu/realtime/plots/{northatlantic,northeastpacific}/YYYY/{al,ep}NNYYYY/stext/*_ships.txt` (2016→); current season also at NHC `atcf/lsdiag/` | Operational predictors along the OFCI track, so no leakage; ~10 KB/file, ~150 MB for 2018–25 |
| GEFS, 31 members, surface | dynamical `noaa-gefs-forecast-35-day` (native tiles 31 mem × 64 leads × 4°; 00Z; 2020-10→) | ~18.6 MB/var per storm patch, all members |
| GEFS control, pressure levels | dynamical `noaa-gefs-forecast-16-day-0-5-degree-virtual` (full-globe chunks; source noaa-gefs-pds, us-east-1) | ~0.25 MB/field |
| IFS-ENS, 51 members, surface | dynamical `ecmwf-ifs-ens-forecast-15-day-0-25-degree` (native tiles 51 mem × 85 leads × 8°; 00Z; 2024-04→) | ~16 MB/var per storm patch |
| IFS-ENS control, pressure levels | ECMWF open data, **GCS mirror** `storage.googleapis.com/ecmwf-open-data/YYYYMMDD/HHz/ifs/0p25/enfo/` (2024-04-02→, `.index` byte ranges; decode with `gribberish`) | u,v,t,q,r,gh,vo,d,w × 13 levels; skt, tcwv; ~0.4–0.7 MB/field; 18 MB/s with 16 threads (AWS eu-central-1 is latency-bound) |
| Later: GEFSv12 reforecast | `noaa-gefs-retrospective` (us-east-1), 2000–19, 00Z, 5 members, `.idx` per message | 2,713 storm days; ~0.7–1.2 TB |

Access notes:
- Open dynamical datasets only via `dynamical_catalog.open(id, chunks=None[, group="pressure_level"])`. Raw `data.dynamical.org`
  URLs are retired.
- `.sel` downloads every chunk it touches. On virtual stores a patch costs the full globe; on native stores it costs only the tiles touched.

## M0: Setup and probes (½ day)
- Repo setup: `git init`, `uv`, `pyproject.toml`, package `src/tcic/`, `tests/`. Shared settings go in `src/tcic/config.py`: paths, rate limits,
  basin and season lists.
- `src/tcic/io/throttle.py`: a token-bucket download limiter (default 17 MB/s) and the `systemd-run` wrapper script `scripts/run_capped.sh`.
- Probes (a few hundred MB in total):
  - Throughput from eu-central-1 (ECMWF open data).
  - GEFS virtual variable names (u/v/t/RH/skin T) and whether chunks are per member.
  - What the a-deck archive actually contains: which early aids, which years, and whether ECMWF aids (EMXI/EMX2/EEMN) are present.
  - Where the real-time SHIPS diagnostic files are archived.

## M1: Ground truth, ATCF parsing, evaluation harness (1–2 days) — DONE 2026-10-16 (`docs/m1_results.md`)
Reproduced NHC 2024 Tables 4 and 5a: exact case counts, MAE within ±0.2 kt. The paper's 2025 OFCL/FNV3 values were reproduced within the CIs.
Early aids: 6-h and 12-h interpolated versions are merged (`atcf.merge_interpolated`), as in NHC verification.

- `tcic/data/ibtracs.py`: NA/EP 2015→, 6-hourly synoptic times.
- `tcic/data/atcf.py`: one a-deck/b-deck parser that writes a tidy parquet with columns (storm, init, tech, lead, lat, lon, vmax, pmin).
- `tcic/evaluate.py`:
  - Homogeneous samples; MAE and bias by lead (12–168 h), by intensity category, and for RI (≥30 kt/24 h).
  - Storm-block bootstrap 95% CIs.
  - Verify against IBTrACS (and b-decks for the current season).
- **Check:** OFCL and GDMN on 2025 NA+EP give MAE ≈ 10–11 kt (paper: 10.7), and recompute NHC's published 2024 verification
  for OFCL, IVCN and HCCA to within ~0.5 kt.

## M1.5: Early-cycle statistical consensus ("our HCCA") (2–3 days, <1 GB) — DONE 2026-10-16 (`docs/m15_summary.md`)
**Result:** TCCA/TCCB match HCCA/IVCN overall (CV 10.2 vs 10.1 kt; 2025 test 11.0–11.1 vs 11.1/11.4; OFCL 10.4), all differences not significant.
On RI cases they significantly beat HCCA (CV −2.3 kt, 2025 −3.3 kt). They are better than HCCA/OFCL at 96–120 h and worse at 12 h. They tie with FNV3 early (GDMI) to 168 h.
SHIPS adds ≤0.2 kt (not significant). Pressure MAE is 8.8 mb (2025). The 10–90% quantiles are under-dispersed (69% coverage).
Implementation notes: the pmin reference is CARQ pmin + the GEFS mean pressure change, since interpolated aids carry no pressure. Ensemble members use the newest run
that is ≥6 h old (GEFS) or ≥12 h old (ECMWF).

**Cases:** NHC synoptic times t0 (00/06/12/18Z), NA+EP, 2018–2026. Seasons are chosen to cover the HWRF→HAFS switch in 2023.

**Inputs, all early (available by t0 + ~3 h):**
- **NHC interpolated aids:** AVNI (GFS), AEMI (GEFS mean), HFAI/HFBI (HAFS; HWFI before 2023), CTCI, EMXI/EMX2 and EGRI if present,
  DSHP, LGEM, plus IVCN and HCCA as baselines. The exact list is fixed in M1 from what the archive holds and NHC's HCCA member list.
- **Our own early versions** of aids that have no "I" form, such as AP01–30 (GEFS members):
  - Take the previous cycle's run (t0 − 6 h, or t0 − 12 h for ECMWF) and shift its leads by the age.
  - Store the intensity offset to CARQ at t0 as a feature rather than applying NHC's decaying offset.
  - This rule is also used for gridded data in M2.
- **CARQ at t0:** initial-condition intensity, plus the change over the previous 12 and 24 h, plus position and motion.

**Rows:** one per (t0, lead), leads 12–168 h. Target: best-track vmax (and, separately, pmin) at t0 + lead, written as a residual from a reference.
The reference is the early aid consensus (IVCN-like mean of available members). This keeps the "additive correction" framing.

**Features:**
- Per-aid forecast intensity at this lead, and its 12/24 h tendency.
- Each aid's offset from CARQ.
- Missing aids left as NaN (LightGBM handles them).
- Aid spread and GEFS member spread.
- Fraction of members with a ≥30 kt/24 h increase.
- Forecast position from the consensus track, lead, day of year, basin, and distance to land (land mask).

**Two variants:**
- **A, no SHIPS:** the features above only.
- **B, + SHIPS:** A plus operational SHIPS environmental predictors: shear, SST/MPI, RH, OHC, upper-level divergence, and others.
  **Leakage rule:** use only the *real-time operational* SHIPS diagnostics. The developmental dataset computes predictors along
  the best track from analyses ("perfect prog"), so it leaks the future. If no operational archive exists, B uses only the
  lead-0 SHIPS predictors from the developmental file, valid at t0. Verify which is possible in M0.

**Model:**
- LightGBM for the mean, plus quantile objectives (10/50/90).
- One model across all leads; RI cases weighted ×2.
- Leave-one-season-out CV over 2018–2024. 2025 is the held-out test (comparable to the paper); 2026 is pseudo-operational
  (train on everything before it).

**Evaluation:** homogeneous comparison of A and B against IVCN, HCCA, OFCL, DSHP/LGEM, the raw early aids, and GDMN where available.

**Success:** A ≤ IVCN, and B ≈ HCCA or better, within the CIs. The gap between A and B shows how much the environment matters before any gridded data.

## M2: Overnight gridded pilot (IFS-ENS + GEFS) — EXTRACTION DONE 2026-10-20 17:43 (694/694 runs, 6.1 GB; `docs/m2_pilot.md`). FIRST MODELS DONE (`docs/m2_summary.md`):
IFS-ENS gridded fields add significant skill over the same-data consensus (−0.75 kt over all seasons, −1.6 kt in 2025). GEFS fields add nothing beyond the AP-track features.
Gridded-only models trail HCCA by about 3 kt with two training seasons. Training takes ~15 s.
Resumable and capped. 694 runs, ~0.75 TB, ~15 h of network-bound work in the 20:00–08:00 window. The tracker is validated against NHC member tracks.

**Question:** do gridded model fields add skill beyond M1.5 for physical models?

**Cases:** 2024-04 → 2026-10 NA+EP, same early rule as M1.5.
- GEFS: the 00Z runs serve t0 = 06Z (6 h old).
- IFS-ENS: the 00Z runs serve t0 = 12Z (12 h old).
- About 340 runs and ~710 storm-runs per model.
- Leads 0–168 h every 6 h of the run (shifted to t0).
- Train on 2024 + 2026, test on 2025, plus 3-fold leave-one-season-out.

**Tracker (`tcic/tracker.py`):**
- Start from the CARQ / best-track position at the run's init time.
- Each member: MSLP minimum near a first guess (persistence plus steering flow), refined with 10 m wind and 850 hPa vorticity.
- Output: lat/lon, pmin, vmax (max 10 m wind within ~300 km), and a lost-storm flag.
- Validate against the a-deck AEMN/AP## (and EEMN if present) tracks.

**Fields kept:**
| Source | Fields |
|---|---|
| IFS-ENS native, **all 51 members**, ±4° box | msl, u10, v10, 10 m gust, t2m, d2m, z500/850/925, t850/925 |
| IFS-ENS control, open-data GRIB, full globe → 16°×16° patches saved | u/v 200/500/850; t 200/500; r 500/700; q 850; vo 850; d 200; skt; tcwv |
| GEFS native, **all 31 members**, ±4° box | msl, u10, v10, gust, t2m, rh2m, PWAT, z500 |
| GEFS control, 0.5° virtual, full globe → patches saved | same environment set; vo/d from u/v; skin T as SST |

**Fields dropped:**
- Precipitation and precipitation type.
- Radiation and cloud.
- 80/100 m winds and temperature.
- Surface pressure.
- Max/min t2m.
- gh above 500 hPa.
- w; hydrometeors and ozone.
- Pressure-level fields for perturbed members.

**Volume:** ~0.37 TB, which is **~6 h at 17 MB/s**: one night. Run IFS-ENS first so a partial run is still usable.
The run must be resumable and work through one (model, init) at a time. Write per-member tracks, features (parquet) and patches (zarr, float16) to `data/`.
Raw GRIB is never kept.

**Features:**
- The paper-style set: anchor intensity; CARQ intensity and its trend; shear 200–850; RH 500–700; SST and empirical MPI; MPI − vmax;
  vo850; d200; tcwv; motion; distance to land.
- Ensemble features: member mean, median and spread of vmax/pmin, member RI fraction, and track spread.
- The M1.5 aid features for the same t0, so the model can learn how gridded and consensus information combine.

**Models:** LightGBM residual models for IFS-ENS and GEFS separately, plus a pooled model with `model_id`, plus a version with
M1.5 features.

**Success (2025 test):**
- Corrected beats the raw ensemble mean significantly.
- It is at least as good as IVCN/DSHP/LGEM.
- It is ideally within the CI of OFCL/GDMN.
- It beats M1.5-A/B, which is what justifies the gridded data.

**Expect:** a smaller gain than the paper's ×3. IFS-ENS already reached 913 hPa for Melissa, against AIFS-ENS's 961. Training data is
thin (~500 storm-runs), so treat this pilot as directional.

## Corrected intensity ensembles (2026-10-20; `docs/ensemble_results.md`, `docs/fnv3_compare.md`, `docs/case_plots.pdf`)
- Member-level tables from NHC member tracks (GEFS AP01–30 2018–26, ECMWF EE01–50 2020–25) and our pilot tracks (IFS-ENS, GEFS 2024–26).
- **Consensus-centred (CCC):** M1.5's corrected mean + each member's raw deviation from the raw mean, scaled per lead (fit on out-of-fold seasons).
  On 2025 the CRPS is ≈8.1–8.9 kt, on par with raw FNV3 members (8.2–8.8; differences not significant). It is better calibrated than FNV3
  (FNV3 covers 60% in its 10–90% range and is 5 kt low); the ECMWF-based CCC is too wide. RI Brier is 0.082–0.086 vs FNV3's 0.076–0.079 (climatology 0.10).
- **Case plots:** our member traces are noisy step to step (12-h jumps of tens of kt) and less physically coherent than FNV3. Cause: per-lead
  scaling (λ≈2) of raw deviations amplifies tracker noise and initial-offset differences.
- **Smoothed version CCS (2026-10-21; `src/tcic/ensemble/smooth.py`, `docs/ensemble_smooth.md`):** deviations anchored at t0, Gaussian-smoothed (σ = 18 h),
  λ(τ) = exp(a + bτ/100), CARQ-anchored smoothed centre. The 12-h zig-zag rate drops from 0.10–0.13 (CCC) to ≤0.001 (best track 0.043, FNV3 0.053).
  CRPS cost is ≤0.15 kt (not significant). It stays on par with FNV3 (CRPS within ±0.3 early, +0.7 vs late; not significant). FNV3's 6-h zig-zag rate is 0.078,
  13× the best track's. **Trade-off:** RI probability from member counts loses its skill (Brier ≈ climatology, 0.088–0.100), and the IFS-ENS/ECMWF
  version is still too wide (91–93% coverage), with fast early spread growth. Next: a separately calibrated RI probability, and a better constraint on IFS spread.
- **Spread-normalised CCSN (2026-10-21; `docs/ensemble_spread.md`):** ECMWF raw member spread roughly doubled from 2024 to 2025 (likely an IFS upgrade),
  so a fixed scale learned on old seasons over-inflated 2025. CCSN rescales by the trailing-120-day median raw spread (past cases only), keeps a fitted share γ
  of case-specific spread (GEFS 0.5–0.75, our IFS-ENS 0), and applies a power-law spread s(τ) = A(τ/12)^p. 2025: IFS-ENS spread/RMSE 1.33 → 0.85, coverage 0.91 → 0.75,
  CRPS 9.01 → 8.75; NHC-ECMWF CRPS −0.60 (significant). GEFS is essentially unchanged (it was already calibrated). CCSN85 (85% spread) is a sharper option at ≤0.1 kt CRPS cost.
  All versions remain level with FNV3 on CRPS. Remaining gap: spread is only weakly case-specific (spread–error correlation 0.27–0.34 vs FNV3's 0.47–0.50).
- **FNV3 member source:** Weather Lab download `https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble/paired/csv/FNV3_YYYY_MM_DDTHH_00_paired.csv`
  (historical data CC BY 4.0). Its member mean matches NHC GDMN to within 1 kt. `src/tcic/data/fnv3.py`; 2025 is in `data/interim/fnv3_2025.parquet`.

- **Final ensemble CCSR80 (2026-10-21; was CCSR85; `docs/ensemble_calibration.md`, `docs/ensemble_smoothspread.md`):** 6-h smoothed, t0-anchored member
  deviations normalised per forecast (shape only). Size from a spread model (LightGBM predicting the corrected consensus's |error| from
  M1.5 features; trained on 2018–24 out-of-fold errors; 2025 corr 0.44) smoothed along lead. Member = consensus mean + 0.80·k(τ)·σ̂·z (80% of the CRPS-optimal spread, chosen as a compromise between 85% and 70%).
  2025: CRPS 8.2/8.5 kt (GEFS/IFS-ENS cycles) vs FNV3 8.5/8.9 (not significant); coverage 73%, spread/error 0.84, nearly unbiased;
  member steps calmer than the best track. CCSR85 (≈76%) and CCSR70 (≈66%) remain as columns for comparison. An asymmetric-spread variant was worse
  and is not used. **RI model** (LightGBM + isotonic, `src/tcic/ensemble/calib.py`): 2025 Brier 0.077 vs climatology 0.102 (25% skill,
  AUC 0.88), on par with FNV3 members (24%). Both under-forecast in the RI-rich 2025 (11% vs 6% climatology).
- **Experiments after CCSR80 (2026-10-21; all additive, nothing existing changed):**
  - *Multi-model ensemble* (`scripts/ens_multimodel.py`, `docs/ensemble_multimodel.md`, `test_multimodel.parquet`): pool GEFS + ECMWF
    member shapes (NHC decks) around the same centre and σ̂, k refit. 2025 CRPS 8.26 vs 8.30/8.34 single-model (−0.08 [−0.24, +0.04]),
    coverage 0.78 vs 0.68, RI-run upper misses 0.19 vs 0.21–0.24. Equal-weight (30+30) is identical. Promising, not significant.
  - *RI model* (`scripts/ri_improve.py`, `docs/ri_improvements.md`, `ri_variants.parquet`): SHIPS environment (R-B) no gain;
    in-season base-rate adaptation selects itself away (N0→10⁴); **NHC's operational RI guidance** (SHIPS-RII/logistic/Bayesian/
    consensus/DTOPS, parsed by `ships.parse_ri` into `data/interim/ships_ri.parquet`) gives a small significant OOF gain
    (Brier −0.0010 [−0.0018, −0.0001]) and 2025 24-h skill 0.25→0.29. Candidate for adoption (not yet the default).
    Our RI model beats SHIPS-RII and the RI consensus at 24 h; DTOPS beats it in 2025 only.
  - *RI-informed upper tail* (`scripts/ens_ritail.py`, `docs/ensemble_ritail.md`): stretch the upper half by 1 + a·P_RI. NHC-deck
    sources: −0.03/−0.05 CRPS (n.s.), upper misses 0.23→0.20; pilot sources overfit (a fitted on one season) and get worse. Not adopted.
  - *Pilot multi-model* (`scripts/ens_multimodel_pilot.py`, `docs/ensemble_multimodel_pilot.md`): our GEFS used 12 h old (t0 = 12Z,
    new `members_pilot_gefs12.parquet`) pooled with our IFS-ENS: CRPS 8.59 vs 8.60 (−0.01 [−0.06, +0.03], 112 forecasts). With a
    shared centre and σ̂ the two models' shapes are nearly interchangeable; pooling mainly smooths coverage.
  - New-data options and order: `docs/new_data_plan.md`.
- **Decisions (2026-10-22):** the goal is the best *correction of the physical ensembles*. FNV3 is a benchmark only, never an input.
  The RI side model is paused (RII variant not adopted). A real-time system is the goal, and an older IFS-ENS run in the early window is
  acceptable. Next: GEFS gridded back to 2020-10 (475 runs, ~200 GB, ~6–7 h active; not started, because the Spark is unstable), then a 2026 test.

## Later stages (open-ended; decide after M2)
- **More training data:**
  - GEFSv12 reforecast 2000–19: ~0.7–1.2 TB, about 12–20 h at 17 MB/s. Its initial conditions come from reanalysis, so add a source flag.
  - Pool across models.
  - GFS 2021→ via native + virtual stores (~0.4 TB).
  - IFS-ENS keeps accumulating.
- **Members:** AIFS-ENS (2025-07→, too short now); deterministic IFS (open-data `oper`); corrected HAFS.
- **Probabilistic ensemble:**
  - Each member gives quantiles; combine by lead-dependent quantile averaging with weights fit on out-of-fold CRPS.
  - Split-conformal adjustment per lead.
  - RI probabilities.
  - Baselines: OFCL plus its historical error distribution, the raw GEFS/IFS-ENS ensembles, and the best single member.
  - Check reliability (PIT) and Brier score versus RI25/30/35.
- **CNN:** the environment patches saved in M2 make it possible without downloading again. Per-member inner-core patches would need more.
- **Compute:**
  - If the Spark network is limiting, look into a wired campus connection (3–5×) or external compute.
  - On AWS, bulk extraction takes hours for ~$50–150. Run in us-west-2 or us-east-1, next to the data. Keep patches in S3 and bring back only features.
- **Real time:**
  - Trigger on new `aid_public` a-decks and dynamical's pipeline status feed.
  - The M1.5 consensus runs first; it needs no gridded data.
  - Gridded members are added as they arrive.
  - Output ATCF-format lines and a plot.
- **Non-stationarity:** model upgrades (HWRF→HAFS, IFS cycles, GFS/GEFS versions). Consider era flags and recency weighting.

## Verification (per milestone)
- **M0:** the probes report numbers; the throttle holds ≤17 MB/s; capped jobs stay ≤8 cores.
- **M1:** parser unit tests on known storms (Melissa 2025 AL13); OFCL/GDMN/IVCN/HCCA MAEs match the paper and NHC verification.
- **M1.5:** CV and test tables with CIs for A and B against baselines; a leakage audit (feature timestamps ≤ t0 + 3 h).
- **M2:** tracker versus a-deck tracks (median position error, intensity correlation); extraction log of MB per run; skill tables;
  a Melissa case-study plot like paper Fig. 5.

## M0 findings (details in `docs/m0_probes.md`)
- The early aids IVCN, HCCA, DSHP/LGEM, AVNI, AEMI, HWFI and CTCI cover ≥65–100% of cycles in every season 2018–25. HFAI/HFBI start in 2023.
  ECMWF EMXI/EMX2/EEMN are public from 2020 at ~half of cycles (2018–19: ~4%). GDMN/GDMI are 2025 only. GEFS members AP01–30 are present from 2020
  (AP01–20 before).
- The operational SHIPS archive was found (NCAR TCGP, above).
- IFS-ENS pressure levels: use the GCS mirror with ~16 threads.
- GEFS 0.5° virtual: inits at 00/06/12/18Z, chunks per member and full-globe; pressure-level variables wind_u/v, temperature, relative_humidity,
  specific_humidity, geopotential_height and absolute_vorticity at 31 levels. Skin T is `temperature_surface`.

## Open items
- **Real-time caveat (found 2026-10-18):** ECMWF aids (EMX/EEMN/EE##, and so EMXI/EMNI) are absent from the 2026 real-time `aid_public`
  a-decks but present in the archive afterwards. A real-time M1.5 must drop the ECMWF features, or take ECMWF from open data via our own tracker (M2).
  Re-evaluate M1.5 without ECMWF features to see what this costs.
- M1.5 follow-ups: the 12-h deficit, quantile calibration, the 2026 pseudo-operational season, and a CV-only hyperparameter search (see `docs/m15_summary.md`).
- HCCA membership by year. 2024: AEMI, GFSI(=AVNI), CTCI, DSHP, EGRI, EMNI, EMXI, HWFI, HMNI, LGEM (NHC 2024 report, Table 1).
  Check earlier years' reports if membership drift matters.
- Whether the SHIPS archive has gaps (storms or cycles missing) relative to the a-deck DSHP cycles.

## 2026-10-25: GEFS back to 2020-10, combined 2025–26 test, real-time consensus
- **GEFS extension:** 492 more 00Z runs (2020-10-01 → 2023-11-26; 839 GEFS runs in total), 0 failures, 7.5 h.
  - Gridded GEFS features still add nothing: −0.02 kt [−0.22, +0.18] over 2020–26. GEFSv12 reforecasts are deprioritised.
  - Pilot GEFS ensemble k is refit on 2020–24: 1.3 − 0.1τ/100 (was 1.6 − 0.2τ/100 on 2024 alone). The five-season fit is kept.
- **Test = 2025 + 2026 combined** (`config.TEST_SEASONS`).
  - Training: 2018–24 for the consensus, the spread model and the NHC-deck ensembles; 2020–24 for pilot GEFS; 2024 for pilot IFS-ENS.
  - 2026 inputs come from `scripts/refresh_season.py`, verified against NHC's working best track.
- **Real-time consensus (now the default):** NHC's real-time decks have no ECMWF aids (EMXI, EMNI, EE01–50).
  - `consensus.features.realtime` drops all ECMWF inputs and recomputes the reference. The with-ECMWF table is kept as
    `consensus_features_with_ecmwf.parquet`.
  - Gain: −0.65 kt over 2025–26, −0.38 kt even in 2025 (`docs/m15_realtime.md`).
  - Remaining gap: 2026 is still +2.05 kt worse than HCCA [0.26, 3.88] and 3.8 kt too weak. 2025 is −0.23 kt.
- **IFS-ENS gridded features** now help much more: −1.04 kt (2024–26), −1.45 kt (2025–26), about −2 kt on RI. Our IFS-ENS extraction is the
  only real-time ECMWF information.
- **Ensembles vs FNV3 (2025–26):**
  - Combined: +0.66 (GEFS) / +0.51 (IFS-ENS) CRPS, not significant.
  - 2025: −0.27 / −0.62 (ours better).
  - 2026: +1.28 / +1.25 (ours about 5 kt too weak).
- **Pipeline:** `scripts/rerun_chain.sh` runs it as a detached, resumable service; launchers set the user-bus environment after a reconnect.
- **Backup:** everything from before these changes is in `data/backup/2026-10-24_before_gefs5_test2526/`.
- **Next:**
  1. Centre the IFS-ENS ensemble on the A+G (gridded) consensus.
  2. Investigate the 2026 shortfall once the final best tracks are out.
  3. Real-time prototype.
- **2026-10-26: our IFS-ENS ensemble is now centred on the gridded consensus (A+G)** (`scripts/ens_gridcentre.py`, column CCSG80,
  `config.FINAL_ENSEMBLE`).
  - vs the track-only centre: −0.59 CRPS [−1.21, +0.11] over 2025–26; −0.88 in 2026. Bias −3.6 → −1.3 kt.
  - vs FNV3 early: −0.08 [−1.24, +1.04] over 2025–26 (2025: −0.77; 2026: +0.37).
  - A+G is leave-one-season-out over 2024–26: each test season is predicted by a model trained on the other two, never on itself.
  - Rollback: set FINAL_ENSEMBLE["pilot_ifs"] = "CCSR80" and rerun fnv3 / fig_ens / cases.

## 2026-10-29: GECO / IECO, nine-season test
- GEFS spread factor: full fitted spread (the 80% trim cost ~0.1 kt on every test), user decision.
- Nine seasons of NHC-GEFS (2018–26, 336 storms, all cycles; `scripts/ens_multiseason.py`, `docs/ensemble_multiseason.md`):
  everything out of sample per season (consensus, spread model, k refit leaving the season out). Fair baseline = "dressed" HCCA/OFCL
  (Gaussian with out-of-season RMS error by lead); a point forecast's |error| is not comparable with an ensemble CRPS.
- Centring on HCCA (public in real time) and using it unsmoothed is best: **GECO** (GEFS Ensemble, Consensus-Oriented),
  `src/tcic/ensemble/geco.py`, `scripts/ens_geco.py`. Beats dressed HCCA in all nine seasons (−0.36 kt [−0.41, −0.29]); beats our
  consensus centre by −0.34 kt; beats FNV3 early on 2025–26 by −0.78 kt [−1.60, −0.01]. k = 1.2. CCSR fallback when HCCA is missing (0.3%).
- **IECO** = the former CCSG80 (IFS-ENS around the gridded consensus, 80% spread). Full spread would be ~0.1 kt better (78% coverage); open.
- `FINAL_ENSEMBLE`: nhc_gefs/pilot_gefs → GECO, pilot_ifs → IECO. Case plots and report figures now show GECO on NHC's GEFS tracks
  (30 members; 12Z figures use the 06Z run). Report updated (new section 6.2). Chain steps `geco`, `multiseason` added.
- IECO switched to full spread (user, 2026-10-29); IECO80 kept as a comparison column. case_plots.pdf gained 11 GECO-only cases from
  2018–24 (out-of-sample members from the nine-season run; leads 12–120 h only). Two-page GECO summary: docs/report/geco_summary.pdf.
- IFS-ENS shapes around HCCA (`scripts/ens_ifs_hcca.py`, `docs/ensemble_ifs_hcca.md`; 12Z cycles 2025–26, 1,589 cases): IFS-ENS
  around HCCA +0.09 kt CRPS vs GECO (significant, also with GECO's k = 1.2), pooled GEFS+IFS-ENS +0.08–0.09 (significant). Both beat
  dressed HCCA (−0.28) and IECO. IFS-ENS shapes add nothing once the centre is HCCA (its run is 12 h old vs 6 h for GEFS). User:
  gridded gains would need a CNN (no compute now); a no-gridded-data system is the preferred outcome.
- 2026-11-03: IECO redefined (user) as the HCCA-centred IFS-ENS ensemble, built like GECO (`scripts/ens_geco.py`; k = 1.50 − 0.30 τ/100
  fit on 2024). 2025–26 12Z: CRPS 7.7, −0.65 kt [−1.53, +0.20] vs FNV3 early, +0.1 vs GECO. Old gridded-centre version kept as column
  CCSG (`scripts/ens_gridcentre.py`). Report, tables, figures and case plots regenerated.

## 2026-11-11/12: IECO from ECMWF's own tracks (official); report trimmed to GECO / IECO
- ECMWF ENS per-member TC tracks (BUFR "tf" files) downloaded from the AWS open-data mirror, 2023-01-18 → 2026-11-10, all four runs
  (5,533 files, 2.5 GB; `scripts/fetch_ecmwf_tracks.py`, paced + 503 backoff). Decoder `src/tcic/data/ecmwf_tf.py` (eccodes) →
  data/interim/ecmwf_tf.parquet (NHC-numbered storms). Members 1–50 pf, 51 control, 52 HRES. Values identical to NHC's archived EE01–50.
  ECMWF posts the tracks ~6 h (06/18Z) and 6.5–7.6 h (00/12Z; ~8.7 h in 2024) after the run starts: the 6-h-old run is in time.
- **IECO** (official) = GECO construction on the 50 ECMWF members (`members.ecmwf_tf()`: BUFR from 2023, NHC archive before; newest run
  ≥ 6 h old; `scripts/ens_ieco.py`, `docs/ensemble_ieco.md`). k = 1.2. 2018–26 (240 storms): −0.28 kt [−0.34, −0.21] vs dressed HCCA,
  significant in every season from 2020 (2018/19: one storm each); +0.04 vs GECO (n.s.). 2025–26 vs FNV3 early: −0.74 [−1.59, +0.07]
  (2026: −1.06*). The old IECO (our tracker, 12Z only) and its comparisons (`ens_ifs_hcca.py`, `ens_gridcentre.py`) are superseded.
- `config.FINAL_ENSEMBLE` replaced by `PRODUCTS` (GECO, IECO → nine-season member files); loader `geco.products()`, shared `geco.dressed()`.
  `fnv3_compare.py`, `report_figures_ens.py`, `case_plots.py` (one cycle for every panel; IECO panels for older storms too) rewritten.
  Chain steps `ecmwf_tf`, `ieco` added; the separate 12Z case-plot file dropped.
- Report rewritten around GECO / IECO only (21 → 12 pages; old version in git). On the common 2025–26 sample both are ~0.8 kt better than
  FNV3 early, borderline over both seasons (GECO −0.80 [−1.65, +0.02]), significant in 2026. GECO summary updated.
