# GECO and IECO vs FNV3 members and dressed HCCA, 2025–26

GECO / IECO: GEFS / IFS-ENS member shapes around HCCA (newest member run ≥ 6 h old). FNV3 = raw Google DeepMind Weather Lab members (50), early (newest run ≥ 6 h old, like ours) or late (run at t0; an upper bound, not available in time). Dressed HCCA = HCCA as a Gaussian with its out-of-season RMS error by lead. Fair CRPS / MAE in kt, all cycles. 95% storm-block bootstrap CIs.

## GECO (GEFS)

59 storms, 4749 cases, leads 12–120 h, homogeneous.

| system       |   CRPS |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-------------|-------:|--------------:|---------------:|------------------:|--------------:|
| GECO         |   7.46 |         10.59 |          -1.16 |              0.76 |          0.88 |
| FNV3 early   |   8.32 |         11.63 |          -5.81 |              0.63 |          0.70 |
| FNV3 late    |   8.03 |         11.18 |          -5.56 |              0.62 |          0.68 |
| HCCA dressed |   7.80 |         10.58 |          -1.11 |              0.84 |          0.98 |

By season (same cases)

|              |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:-------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| GECO         |             8.21 |            -2.58 |                 0.74 |             6.98 |            -0.24 |                 0.77 |
| FNV3 early   |             8.70 |            -5.59 |                 0.61 |             8.08 |            -5.95 |                 0.64 |
| FNV3 late    |             8.35 |            -5.16 |                 0.60 |             7.82 |            -5.82 |                 0.63 |
| HCCA dressed |             8.55 |            -2.56 |                 0.82 |             7.32 |            -0.17 |                 0.86 |

- 2025–26: CRPS GECO − HCCA dressed = -0.34 [-0.49, -0.20] **significant**
- 2025: CRPS GECO − HCCA dressed = -0.33 [-0.61, -0.12] **significant**
- 2026: CRPS GECO − HCCA dressed = -0.34 [-0.51, -0.19] **significant**
- 2025–26: CRPS GECO − FNV3 early = -0.86 [-1.69, -0.04] **significant**
- 2025: CRPS GECO − FNV3 early = -0.49 [-2.27, +0.95]
- 2026: CRPS GECO − FNV3 early = -1.10 [-1.94, -0.21] **significant**
- 2025–26: CRPS GECO − FNV3 late = -0.56 [-1.40, +0.26]
- 2025: CRPS GECO − FNV3 late = -0.14 [-1.92, +1.34]
- 2026: CRPS GECO − FNV3 late = -0.84 [-1.67, +0.02]

CRPS by lead

|              |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| GECO         |  4.2 |  5.9 |  7.2 |  7.8 |  8.3 |  8.6 |  8.8 |  9.4 |  10.3 |  10.4 |
| FNV3 early   |  6.4 |  7.0 |  7.6 |  8.3 |  9.0 |  9.2 |  9.9 | 10.0 |  10.3 |  10.4 |
| FNV3 late    |  6.2 |  6.7 |  7.3 |  7.8 |  8.6 |  9.0 |  9.4 |  9.9 |  10.2 |  10.0 |
| HCCA dressed |  4.4 |  6.2 |  7.5 |  8.1 |  8.8 |  9.0 |  9.3 |  9.9 |  10.6 |  10.9 |

RI probability from member counts (leads 36–120 h; each system's own cases)

|            |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:-----------|--------:|---------:|-----------:|-------------:|---------:|
| GECO       |   0.068 |    0.057 |      0.093 |        0.084 | 4867.000 |
| FNV3 early |   0.063 |    0.049 |      0.088 |        0.080 | 4407.000 |
| FNV3 late  |   0.064 |    0.052 |      0.092 |        0.083 | 3537.000 |

## IECO (IFS-ENS)

58 storms, 4721 cases, leads 12–120 h, homogeneous.

| system       |   CRPS |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-------------|-------:|--------------:|---------------:|------------------:|--------------:|
| IECO         |   7.54 |         10.59 |          -1.22 |              0.76 |          0.88 |
| FNV3 early   |   8.34 |         11.66 |          -5.85 |              0.62 |          0.69 |
| FNV3 late    |   8.05 |         11.21 |          -5.58 |              0.62 |          0.68 |
| HCCA dressed |   7.81 |         10.59 |          -1.14 |              0.84 |          0.98 |

By season (same cases)

|              |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:-------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| IECO         |             8.38 |            -2.68 |                 0.75 |             6.99 |            -0.27 |                 0.77 |
| FNV3 early   |             8.71 |            -5.60 |                 0.61 |             8.11 |            -6.00 |                 0.64 |
| FNV3 late    |             8.35 |            -5.18 |                 0.60 |             7.85 |            -5.85 |                 0.63 |
| HCCA dressed |             8.56 |            -2.57 |                 0.82 |             7.33 |            -0.21 |                 0.86 |

- 2025–26: CRPS IECO − HCCA dressed = -0.28 [-0.42, -0.13] **significant**
- 2025: CRPS IECO − HCCA dressed = -0.18 [-0.39, -0.02] **significant**
- 2026: CRPS IECO − HCCA dressed = -0.34 [-0.52, -0.17] **significant**
- 2025–26: CRPS IECO − FNV3 early = -0.81 [-1.67, +0.05]
- 2025: CRPS IECO − FNV3 early = -0.33 [-1.99, +1.18]
- 2026: CRPS IECO − FNV3 early = -1.12 [-1.96, -0.23] **significant**
- 2025–26: CRPS IECO − FNV3 late = -0.51 [-1.40, +0.35]
- 2025: CRPS IECO − FNV3 late = +0.02 [-1.69, +1.57]
- 2026: CRPS IECO − FNV3 late = -0.86 [-1.74, +0.00]

CRPS by lead

|              |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| IECO         |  4.2 |  6.0 |  7.2 |  7.9 |  8.4 |  8.7 |  9.0 |  9.7 |  10.4 |  10.5 |
| FNV3 early   |  6.4 |  7.0 |  7.7 |  8.3 |  9.0 |  9.2 |  9.9 | 10.0 |  10.3 |  10.4 |
| FNV3 late    |  6.3 |  6.7 |  7.3 |  7.8 |  8.6 |  9.0 |  9.4 | 10.0 |  10.2 |   9.9 |
| HCCA dressed |  4.4 |  6.2 |  7.5 |  8.2 |  8.8 |  9.0 |  9.3 |  9.9 |  10.6 |  10.9 |

RI probability from member counts (leads 36–120 h; each system's own cases)

|            |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:-----------|--------:|---------:|-----------:|-------------:|---------:|
| IECO       |   0.066 |    0.048 |      0.090 |        0.082 | 4451.000 |
| FNV3 early |   0.063 |    0.049 |      0.088 |        0.080 | 4407.000 |
| FNV3 late  |   0.064 |    0.052 |      0.092 |        0.083 | 3537.000 |

## GECO and IECO on their common cases

58 storms, 4721 cases, leads 12–120 h, homogeneous.

| system       |   CRPS |   MAE of mean |   bias of mean |   10–90% coverage |   spread/RMSE |
|:-------------|-------:|--------------:|---------------:|------------------:|--------------:|
| GECO         |   7.48 |         10.60 |          -1.19 |              0.76 |          0.88 |
| IECO         |   7.54 |         10.59 |          -1.22 |              0.76 |          0.88 |
| FNV3 early   |   8.34 |         11.66 |          -5.85 |              0.62 |          0.69 |
| FNV3 late    |   8.05 |         11.21 |          -5.58 |              0.62 |          0.68 |
| HCCA dressed |   7.81 |         10.59 |          -1.14 |              0.84 |          0.98 |

By season (same cases)

|              |   (2025, 'CRPS') |   (2025, 'bias') |   (2025, 'coverage') |   (2026, 'CRPS') |   (2026, 'bias') |   (2026, 'coverage') |
|:-------------|-----------------:|-----------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| GECO         |             8.23 |            -2.59 |                 0.74 |             6.99 |            -0.28 |                 0.77 |
| IECO         |             8.38 |            -2.68 |                 0.75 |             6.99 |            -0.27 |                 0.77 |
| FNV3 early   |             8.71 |            -5.60 |                 0.61 |             8.11 |            -6.00 |                 0.64 |
| FNV3 late    |             8.35 |            -5.18 |                 0.60 |             7.85 |            -5.85 |                 0.63 |
| HCCA dressed |             8.56 |            -2.57 |                 0.82 |             7.33 |            -0.21 |                 0.86 |

- 2025–26: CRPS GECO − HCCA dressed = -0.34 [-0.48, -0.19] **significant**
- 2025: CRPS GECO − HCCA dressed = -0.33 [-0.61, -0.13] **significant**
- 2026: CRPS GECO − HCCA dressed = -0.34 [-0.50, -0.18] **significant**
- 2025–26: CRPS GECO − FNV3 early = -0.87 [-1.73, -0.03] **significant**
- 2025: CRPS GECO − FNV3 early = -0.48 [-2.16, +1.00]
- 2026: CRPS GECO − FNV3 early = -1.12 [-1.94, -0.22] **significant**
- 2025–26: CRPS GECO − FNV3 late = -0.57 [-1.45, +0.28]
- 2025: CRPS GECO − FNV3 late = -0.13 [-1.82, +1.39]
- 2026: CRPS GECO − FNV3 late = -0.86 [-1.72, +0.01]
- 2025–26: CRPS IECO − HCCA dressed = -0.28 [-0.42, -0.13] **significant**
- 2025: CRPS IECO − HCCA dressed = -0.18 [-0.39, -0.02] **significant**
- 2026: CRPS IECO − HCCA dressed = -0.34 [-0.52, -0.17] **significant**
- 2025–26: CRPS IECO − FNV3 early = -0.81 [-1.67, +0.05]
- 2025: CRPS IECO − FNV3 early = -0.33 [-1.99, +1.18]
- 2026: CRPS IECO − FNV3 early = -1.12 [-1.96, -0.23] **significant**
- 2025–26: CRPS IECO − FNV3 late = -0.51 [-1.40, +0.35]
- 2025: CRPS IECO − FNV3 late = +0.02 [-1.69, +1.57]
- 2026: CRPS IECO − FNV3 late = -0.86 [-1.74, +0.00]

CRPS by lead

|              |   12 |   24 |   36 |   48 |   60 |   72 |   84 |   96 |   108 |   120 |
|:-------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| GECO         |  4.3 |  6.0 |  7.2 |  7.9 |  8.3 |  8.6 |  8.8 |  9.5 |  10.3 |  10.4 |
| IECO         |  4.2 |  6.0 |  7.2 |  7.9 |  8.4 |  8.7 |  9.0 |  9.7 |  10.4 |  10.5 |
| FNV3 early   |  6.4 |  7.0 |  7.7 |  8.3 |  9.0 |  9.2 |  9.9 | 10.0 |  10.3 |  10.4 |
| FNV3 late    |  6.3 |  6.7 |  7.3 |  7.8 |  8.6 |  9.0 |  9.4 | 10.0 |  10.2 |   9.9 |
| HCCA dressed |  4.4 |  6.2 |  7.5 |  8.2 |  8.8 |  9.0 |  9.3 |  9.9 |  10.6 |  10.9 |

RI probability from member counts (leads 36–120 h; each system's own cases)

|            |   Brier |   mean P |   obs freq |   Brier clim |    cases |
|:-----------|--------:|---------:|-----------:|-------------:|---------:|
| GECO       |   0.068 |    0.057 |      0.093 |        0.084 | 4867.000 |
| IECO       |   0.066 |    0.048 |      0.090 |        0.082 | 4451.000 |
| FNV3 early |   0.063 |    0.049 |      0.088 |        0.080 | 4407.000 |
| FNV3 late  |   0.064 |    0.052 |      0.092 |        0.083 | 3537.000 |
