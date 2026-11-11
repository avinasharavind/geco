# Consensus-centred corrected ensembles (final: GECO / IECO) vs FNV3 members, 2025–26
Ours = consensus-centred, spread-calibrated members (CCC): M1.5's corrected mean + each member's raw deviation, scaled per lead. FNV3 = raw Google DeepMind Weather Lab members (50), early (newest run ≥ 6 h old, like ours) or late (run at t0; an upper bound). CRPS/quantile score/MAE in kt. 95% storm-block bootstrap CIs.

## IFS-ENS cycles (t0 = 12Z)

56 storms, 1586 cases, leads 12–120 h, homogeneous.

| system                                         |   CRPS |   quantile score |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-----------------------------------------------|-------:|-----------------:|--------------:|---------------:|------------------:|--------------:|
| Ours IFS-ENS IECO                              |   7.65 |             3.48 |         10.77 |          -1.13 |              0.80 |          0.95 |
| FNV3 late (age [np.int64(0)] h)                |   8.15 |             3.77 |         11.38 |          -5.71 |              0.61 |          0.67 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |   8.31 |             3.82 |         11.62 |          -5.87 |              0.62 |          0.69 |

By season (same cases)

|                                                |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:-----------------------------------------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| Ours IFS-ENS IECO                              |             8.52 |            -2.48 |                 0.79 |             7.08 |            -0.25 |                 0.80 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |             8.92 |            -5.53 |                 0.59 |             7.91 |            -6.09 |                 0.64 |
| FNV3 late (age [np.int64(0)] h)                |             8.38 |            -5.21 |                 0.59 |             8.00 |            -6.03 |                 0.61 |
- CRPS Ours IFS-ENS IECO − FNV3 early (age [np.int64(6), np.int64(12)] h) = -0.66 [-1.55, +0.18]
- CRPS Ours IFS-ENS IECO − FNV3 late (age [np.int64(0)] h) = -0.50 [-1.53, +0.46]

CRPS by lead

|                                                |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-----------------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| Ours IFS-ENS IECO                              |  4.4 |  6.0 |  7.2 |  8.0 |  8.7 |  9.0 |  8.9 |  9.9 |  10.3 |  10.4 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |  6.1 |  6.8 |  7.6 |  8.4 |  8.7 |  9.3 |  9.6 | 10.6 |  10.5 |  11.0 |
| FNV3 late (age [np.int64(0)] h)                |  6.3 |  6.8 |  7.4 |  8.1 |  8.9 |  9.4 |  9.4 | 10.5 |   9.7 |   9.7 |

RI probability (leads 36–120 h; each system's own cases)

|                   |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:------------------|--------:|---------:|-----------:|-------------:|---------:|
| Ours IFS-ENS IECO |   0.069 |    0.051 |      0.095 |        0.086 | 1206.000 |
| FNV3 early        |   0.062 |    0.047 |      0.088 |        0.080 | 1097.000 |
| FNV3 late         |   0.065 |    0.051 |      0.096 |        0.086 | 1183.000 |

## IFS-ENS cycles, 2025 only, with NHC's ECMWF members

29 storms, 600 cases, leads 12–120 h, homogeneous.

| system                                         |   CRPS |   quantile score |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-----------------------------------------------|-------:|-----------------:|--------------:|---------------:|------------------:|--------------:|
| NHC-ECMWF CCSR80                               |   8.29 |             3.80 |         11.56 |          -1.18 |              0.67 |          0.72 |
| FNV3 late (age [np.int64(0)] h)                |   8.35 |             3.87 |         11.39 |          -5.07 |              0.60 |          0.61 |
| Ours IFS-ENS IECO                              |   8.54 |             3.87 |         12.02 |          -2.42 |              0.79 |          0.88 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |   8.81 |             4.07 |         12.14 |          -5.28 |              0.60 |          0.61 |

By season (same cases)

|                                                |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |
|:-----------------------------------------------|-----------------:|-----------------:|---------------------:|
| Ours IFS-ENS IECO                              |             8.54 |            -2.42 |                 0.79 |
| NHC-ECMWF CCSR80                               |             8.29 |            -1.18 |                 0.67 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |             8.81 |            -5.28 |                 0.60 |
| FNV3 late (age [np.int64(0)] h)                |             8.35 |            -5.07 |                 0.60 |
- CRPS Ours IFS-ENS IECO − FNV3 early (age [np.int64(6), np.int64(12)] h) = -0.26 [-2.18, +1.32]
- CRPS Ours IFS-ENS IECO − FNV3 late (age [np.int64(0)] h) = +0.20 [-1.64, +1.81]
- CRPS NHC-ECMWF CCSR80 − FNV3 early (age [np.int64(6), np.int64(12)] h) = -0.52 [-2.47, +1.22]
- CRPS NHC-ECMWF CCSR80 − FNV3 late (age [np.int64(0)] h) = -0.06 [-1.82, +1.74]

CRPS by lead

|                                                |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-----------------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| Ours IFS-ENS IECO                              |  4.4 |  6.2 |  7.8 |  9.7 | 10.4 | 11.3 | 11.0 | 12.6 |  11.7 |  11.6 |
| NHC-ECMWF CCSR80                               |  4.6 |  6.6 |  8.2 |  9.6 | 10.0 | 10.5 |  9.4 | 11.8 |  10.5 |  10.4 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |  6.4 |  7.6 |  8.1 |  9.2 |  8.8 |  9.7 | 10.0 | 13.0 |  12.1 |  11.9 |
| FNV3 late (age [np.int64(0)] h)                |  6.8 |  7.7 |  7.6 |  8.5 |  8.2 |  9.4 |  9.2 | 11.4 |  10.1 |  10.4 |

RI probability (leads 36–120 h; each system's own cases)

|                   |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:------------------|--------:|---------:|-----------:|-------------:|---------:|
| Ours IFS-ENS IECO |   0.069 |    0.051 |      0.095 |        0.086 | 1206.000 |
| NHC-ECMWF CCSR80  |   0.091 |    0.032 |      0.114 |        0.101 |  387.000 |
| FNV3 early        |   0.062 |    0.047 |      0.088 |        0.080 | 1097.000 |
| FNV3 late         |   0.065 |    0.051 |      0.096 |        0.086 | 1183.000 |

## GEFS cycles (t0 = 06Z)

56 storms, 1613 cases, leads 12–120 h, homogeneous.

| system                           |   CRPS |   quantile score |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:---------------------------------|-------:|-----------------:|--------------:|---------------:|------------------:|--------------:|
| NHC-GEFS GECO                    |   7.43 |             3.44 |         10.52 |          -1.21 |              0.76 |          0.88 |
| Ours GEFS GECO                   |   7.52 |             3.49 |         10.58 |          -1.13 |              0.74 |          0.87 |
| FNV3 early (age [np.int64(6)] h) |   8.06 |             3.72 |         11.24 |          -5.67 |              0.63 |          0.71 |
| FNV3 late (age [np.int64(0)] h)  |   8.14 |             3.74 |         11.34 |          -5.99 |              0.62 |          0.67 |

By season (same cases)

|                                  |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:---------------------------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| Ours GEFS GECO                   |             8.34 |            -2.99 |                 0.72 |             6.98 |             0.09 |                 0.76 |
| NHC-GEFS GECO                    |             8.27 |            -2.99 |                 0.74 |             6.87 |            -0.05 |                 0.78 |
| FNV3 early (age [np.int64(6)] h) |             8.42 |            -5.72 |                 0.61 |             7.83 |            -5.64 |                 0.65 |
| FNV3 late (age [np.int64(0)] h)  |             8.23 |            -5.01 |                 0.61 |             8.08 |            -6.63 |                 0.62 |
- CRPS Ours GEFS GECO − FNV3 early (age [np.int64(6)] h) = -0.55 [-1.59, +0.38]
- CRPS Ours GEFS GECO − FNV3 late (age [np.int64(0)] h) = -0.63 [-1.60, +0.30]
- CRPS NHC-GEFS GECO − FNV3 early (age [np.int64(6)] h) = -0.64 [-1.64, +0.29]
- CRPS NHC-GEFS GECO − FNV3 late (age [np.int64(0)] h) = -0.72 [-1.67, +0.23]

CRPS by lead

|                                  |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:---------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| Ours GEFS GECO                   |  4.2 |  6.0 |  7.1 |  7.8 |  8.3 |  8.2 |  8.7 |  9.9 |  10.9 |  10.8 |
| NHC-GEFS GECO                    |  4.2 |  5.9 |  7.1 |  7.7 |  8.2 |  8.1 |  8.4 |  9.5 |  10.6 |  10.6 |
| FNV3 early (age [np.int64(6)] h) |  6.2 |  6.8 |  7.5 |  7.8 |  8.9 |  8.7 |  9.2 |  9.5 |  10.4 |  10.0 |
| FNV3 late (age [np.int64(0)] h)  |  6.1 |  6.5 |  7.3 |  7.8 |  8.7 |  9.2 |  9.7 | 10.1 |  11.3 |  10.3 |

RI probability (leads 36–120 h; each system's own cases)

|                |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:---------------|--------:|---------:|-----------:|-------------:|---------:|
| Ours GEFS GECO |   0.069 |    0.060 |      0.089 |        0.081 | 1220.000 |
| NHC-GEFS GECO  |   0.069 |    0.060 |      0.090 |        0.082 | 1225.000 |
| FNV3 early     |   0.064 |    0.054 |      0.087 |        0.080 | 1124.000 |
| FNV3 late      |   0.065 |    0.048 |      0.086 |        0.079 | 1168.000 |

## All 2025–26 cycles (NHC-tracker GEFS members)

59 storms, 4751 cases, leads 12–120 h, homogeneous.

| system                                         |   CRPS |   quantile score |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-----------------------------------------------|-------:|-----------------:|--------------:|---------------:|------------------:|--------------:|
| NHC-GEFS GECO                                  |   7.47 |             3.47 |         10.59 |          -1.17 |              0.76 |          0.88 |
| FNV3 late (age [np.int64(0)] h)                |   8.03 |             3.70 |         11.18 |          -5.56 |              0.62 |          0.68 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |   8.32 |             3.84 |         11.64 |          -5.82 |              0.63 |          0.70 |

By season (same cases)

|                                                |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:-----------------------------------------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| NHC-GEFS GECO                                  |             8.21 |            -2.58 |                 0.74 |             6.99 |            -0.26 |                 0.77 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |             8.70 |            -5.59 |                 0.61 |             8.08 |            -5.97 |                 0.64 |
| FNV3 late (age [np.int64(0)] h)                |             8.35 |            -5.16 |                 0.60 |             7.82 |            -5.82 |                 0.63 |
- CRPS NHC-GEFS GECO − FNV3 early (age [np.int64(6), np.int64(12)] h) = -0.85 [-1.67, -0.03] **significant**
- CRPS NHC-GEFS GECO − FNV3 late (age [np.int64(0)] h) = -0.56 [-1.39, +0.27]

CRPS by lead

|                                                |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-----------------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| NHC-GEFS GECO                                  |  4.2 |  5.9 |  7.2 |  7.8 |  8.3 |  8.6 |  8.8 |  9.5 |  10.3 |  10.4 |
| FNV3 early (age [np.int64(6), np.int64(12)] h) |  6.4 |  7.0 |  7.6 |  8.3 |  9.0 |  9.2 |  9.9 | 10.0 |  10.3 |  10.4 |
| FNV3 late (age [np.int64(0)] h)                |  6.2 |  6.7 |  7.3 |  7.8 |  8.6 |  9.0 |  9.4 |  9.9 |  10.2 |  10.0 |

RI probability (leads 36–120 h; each system's own cases)

|               |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:--------------|--------:|---------:|-----------:|-------------:|---------:|
| NHC-GEFS GECO |   0.068 |    0.057 |      0.093 |        0.084 | 4880.000 |
| FNV3 early    |   0.063 |    0.049 |      0.088 |        0.080 | 4407.000 |
| FNV3 late     |   0.064 |    0.052 |      0.092 |        0.083 | 3537.000 |
