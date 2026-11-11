# IFS-ENS ensemble centred on the gridded consensus (CCSG)

CCSG = CCSR (full spread) with the centre taken from the consensus correction with IFS-ENS gridded features (A+G, leave-one-season-out over 2024–26) instead of the track-only consensus. k = 1.60 + -0.30 τ/100 (refit on 2024). 12Z cycles, leads 12–120 h, identical cases (56 storms, 1590 cases).

|                                           |   CRPS |   MAE of mean |   bias |   coverage |   spread/error | CRPS − FNV3 early    | CRPS − Track-only centre   |
|:------------------------------------------|-------:|--------------:|-------:|-----------:|---------------:|:---------------------|:---------------------------|
| ('2025–26', 'Track-only centre (CCSR80)') |   8.81 |         12.25 |  -3.61 |       0.70 |           0.74 | +0.51 [-0.96, +2.01] | nan                        |
| ('2025–26', 'Gridded centre (CCSG)')      |   8.13 |         11.64 |  -1.30 |       0.78 |           1.00 | -0.17 [-1.26, +0.86] | -0.67 [-1.34, +0.02]       |
| ('2025–26', 'CCSG80 (80% spread)')        |   8.21 |         11.64 |  -1.30 |       0.67 |           0.80 | -0.08 [-1.24, +1.04] | -0.59 [-1.21, +0.11]       |
| ('2025–26', 'FNV3 early')                 |   8.30 |         11.60 |  -5.86 |       0.62 |           0.69 | nan                  | -0.51 [-2.01, +0.96]       |
| ('2025', 'Track-only centre (CCSR80)')    |   8.27 |         11.61 |  -1.04 |       0.75 |           0.82 | -0.62 [-2.40, +1.11] | nan                        |
| ('2025', 'Gridded centre (CCSG)')         |   8.10 |         11.54 |   0.78 |       0.80 |           1.07 | -0.80 [-2.51, +0.87] | -0.17 [-1.16, +0.78]       |
| ('2025', 'CCSG80 (80% spread)')           |   8.12 |         11.54 |   0.78 |       0.69 |           0.86 | -0.77 [-2.52, +0.97] | -0.15 [-1.11, +0.87]       |
| ('2025', 'FNV3 early')                    |   8.89 |         12.20 |  -5.52 |       0.59 |           0.61 | nan                  | +0.62 [-1.11, +2.40]       |
| ('2026', 'Track-only centre (CCSR80)')    |   9.16 |         12.67 |  -5.29 |       0.66 |           0.69 | +1.25 [-0.60, +3.51] | nan                        |
| ('2026', 'Gridded centre (CCSG)')         |   8.16 |         11.70 |  -2.67 |       0.76 |           0.95 | +0.25 [-0.95, +1.71] | -1.00 [-1.99, -0.10]       |
| ('2026', 'CCSG80 (80% spread)')           |   8.28 |         11.71 |  -2.65 |       0.66 |           0.76 | +0.37 [-0.90, +1.94] | -0.88 [-1.80, +0.00]       |
| ('2026', 'FNV3 early')                    |   7.91 |         11.21 |  -6.09 |       0.64 |           0.75 | nan                  | -1.25 [-3.51, +0.60]       |

_Run time 5 s._
