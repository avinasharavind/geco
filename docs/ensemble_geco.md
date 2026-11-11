# GECO / IECO (GEFS / IFS-ENS members around HCCA), 2025–26

GECO (IECO) = GEFS (IFS-ENS) member shapes (smoothed, anchored, standardised deviations) around NHC's HCCA as issued, scaled by the predicted spread σ̂ and k(τ) fit on the out-of-fold training seasons; full spread. Forecasts without HCCA fall back to CCSR. GEFS: all cycles; IFS-ENS: 12Z cycles. Leads 12–120 h. Multi-season evaluation: docs/ensemble_multiseason.md.

|                          |   CRPS |   MAE of mean |   bias |   coverage |   spread/error | − CCSR               | k(τ)             |   CCSR fallback share |
|:-------------------------|-------:|--------------:|-------:|-----------:|---------------:|:---------------------|:-----------------|----------------------:|
| ('NHC-GEFS', 'RAW')      |  22.05 |         25.40 | -24.95 |       0.23 |           0.21 | nan                  | nan              |                nan    |
| ('NHC-GEFS', 'CCSR')     |   8.40 |         12.04 |  -2.95 |       0.74 |           0.86 | nan                  | nan              |                nan    |
| ('NHC-GEFS', 'GECO')     |   7.61 |         10.80 |  -1.12 |       0.76 |           0.88 | -0.79 [-1.87, +0.24] | 1.20 +0.00 τ/100 |                  0.00 |
| ('Ours GEFS', 'RAW')     |  20.92 |         24.32 | -23.45 |       0.24 |           0.22 | nan                  | nan              |                nan    |
| ('Ours GEFS', 'CCSR')    |   8.52 |         12.08 |  -3.15 |       0.72 |           0.81 | nan                  | nan              |                nan    |
| ('Ours GEFS', 'GECO')    |   7.55 |         10.60 |  -1.14 |       0.75 |           0.88 | -0.97 [-2.23, +0.11] | 1.20 +0.00 τ/100 |                  0.00 |
| ('Ours IFS-ENS', 'RAW')  |  17.92 |         22.79 | -22.05 |       0.43 |           0.41 | nan                  | nan              |                nan    |
| ('Ours IFS-ENS', 'CCSR') |   8.71 |         12.25 |  -2.91 |       0.80 |           0.94 | nan                  | nan              |                nan    |
| ('Ours IFS-ENS', 'IECO') |   7.70 |         10.79 |  -1.11 |       0.81 |           0.96 | -1.01 [-2.16, +0.06] | 1.50 -0.30 τ/100 |                  0.00 |

_Run time 82 s._
