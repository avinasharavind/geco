# Storm-specific spread and calibrated RI probability (2025–26)

## 1. Spread model: how well does the predicted error σ̂ track the actual error of the corrected consensus?

| sample              |   corr(σ̂, |error|) |   corr(lead-only, |error|) |   corr within lead (σ̂) |   corr within lead (M1.5 10–90% width) |
|:--------------------|--------------------:|---------------------------:|------------------------:|---------------------------------------:|
| out-of-fold 2018–24 |                0.50 |                       0.34 |                    0.39 |                                   0.36 |
| 2025–26             |                0.40 |                       0.24 |                    0.33 |                                   0.33 |

2025 cases binned by predicted error (all leads):

| bin         |   cases |   predicted |   actual |
|:------------|--------:|------------:|---------:|
| lowest 20%  |  1650.0 |         5.0 |      5.3 |
| 20–40%      |  1649.0 |         7.9 |      9.0 |
| 40–60%      |  1649.0 |        11.0 |     13.6 |
| 60–80%      |  1649.0 |        14.6 |     15.6 |
| highest 20% |  1650.0 |        21.0 |     19.0 |

## 2. Ensembles with storm-specific spread (2025–26, leads 12–120 h, homogeneous within each ensemble)

CCSP = corrected mean + k(τ)·σ̂·(standardised smoothed member deviations). Coverage by thirds of predicted error shows whether the interval holds its coverage in both calm and volatile situations (it should be ≈ 0.8 in every third).

|                                                  |   CRPS |   10–90% coverage |   spread/error |   spread–error corr. |   spread–error corr. within lead | coverage calm / middle / volatile third   | CRPS − CCSN [95% CI]   | CRPS − FNV3 [95% CI]   |
|:-------------------------------------------------|-------:|------------------:|---------------:|---------------------:|---------------------------------:|:------------------------------------------|:-----------------------|:-----------------------|
| ('Ours GEFS', 'CCSN (lead-only spread)')         |   8.28 |              0.74 |           0.87 |                 0.42 |                             0.36 | 0.84 / 0.69 / 0.68                        | nan                    | nan                    |
| ('Ours GEFS', 'CCSP (storm-specific spread)')    |   8.29 |              0.74 |           0.85 |                 0.42 |                             0.37 | 0.74 / 0.72 / 0.75                        | +0.01 [-0.20, +0.26]   | +0.19 [-1.25, +1.50]   |
| ('Ours GEFS', 'FNV3 members')                    |   8.11 |              0.63 |           0.70 |                 0.48 |                             0.45 | 0.65 / 0.63 / 0.62                        | nan                    | nan                    |
| ('Ours IFS-ENS', 'CCSN (lead-only spread)')      |   8.94 |              0.75 |           0.85 |                 0.26 |                             0.10 | 0.89 / 0.71 / 0.65                        | nan                    | nan                    |
| ('Ours IFS-ENS', 'CCSP (storm-specific spread)') |   8.75 |              0.78 |           0.90 |                 0.38 |                             0.32 | 0.79 / 0.73 / 0.81                        | -0.19 [-0.33, -0.07]   | +0.45 [-0.96, +1.89]   |
| ('Ours IFS-ENS', 'FNV3 members')                 |   8.30 |              0.62 |           0.69 |                 0.46 |                             0.43 | 0.66 / 0.59 / 0.60                        | nan                    | nan                    |
| ('NHC-GEFS', 'CCSN (lead-only spread)')          |   8.44 |              0.76 |           0.89 |                 0.39 |                             0.36 | 0.86 / 0.72 / 0.70                        | nan                    | nan                    |
| ('NHC-GEFS', 'CCSP (storm-specific spread)')     |   8.45 |              0.72 |           0.82 |                 0.39 |                             0.33 | 0.72 / 0.69 / 0.74                        | +0.01 [-0.16, +0.19]   | +0.08 [-1.24, +1.36]   |
| ('NHC-GEFS', 'FNV3 members')                     |   8.37 |              0.62 |           0.69 |                 0.47 |                             0.44 | 0.65 / 0.61 / 0.61                        | nan                    | nan                    |
| ('NHC-ECMWF', 'CCSN (lead-only spread)')         |   8.14 |              0.76 |           0.89 |                 0.31 |                             0.27 | 0.87 / 0.73 / 0.69                        | nan                    | nan                    |
| ('NHC-ECMWF', 'CCSP (storm-specific spread)')    |   8.02 |              0.77 |           0.94 |                 0.41 |                             0.36 | 0.73 / 0.76 / 0.83                        | -0.12 [-0.25, +0.01]   | -0.62 [-2.58, +1.04]   |
| ('NHC-ECMWF', 'FNV3 members')                    |   8.64 |              0.62 |           0.62 |                 0.48 |                             0.45 | 0.68 / 0.60 / 0.56                        | nan                    | nan                    |

## 3. Rapid-intensification probability (P of ≥ 30 kt rise in the 24 h ending at t0 + τ, τ = 24–120 h), 2025–26

Skill vs climatology = 1 − Brier / Brier(climatology), where climatology is the 2018–24 RI frequency. AUC measures ranking only.

|                                                                                  |   Brier |   skill vs climatology |     AUC |   mean P |   observed rate |    cases |   RI cases |
|:---------------------------------------------------------------------------------|--------:|-----------------------:|--------:|---------:|----------------:|---------:|-----------:|
| ('all 2025–26 cycles', 'Climatology')                                            |   0.084 |                  0.000 | nan     |    0.060 |           0.092 | 5342.000 |    490.000 |
| ('all 2025–26 cycles', 'RI model (calibrated)')                                  |   0.068 |                  0.192 |   0.874 |    0.066 |           0.092 | 5342.000 |    490.000 |
| ('all 2025–26 cycles', 'RI model (uncalibrated)')                                |   0.069 |                  0.182 |   0.877 |    0.055 |           0.092 | 5342.000 |    490.000 |
| ('all 2025–26 cycles', 'FNV3 members (share with +30 kt / 24 h)')                |   0.065 |                  0.234 |   0.860 |    0.052 |           0.092 | 5342.000 |    490.000 |
| ('Ours GEFS cycles (06Z)', 'Climatology')                                        |   0.083 |                  0.000 | nan     |    0.060 |           0.090 | 1344.000 |    121.000 |
| ('Ours GEFS cycles (06Z)', 'RI model (calibrated)')                              |   0.070 |                  0.153 |   0.852 |    0.066 |           0.090 | 1344.000 |    121.000 |
| ('Ours GEFS cycles (06Z)', 'RI model (uncalibrated)')                            |   0.071 |                  0.143 |   0.851 |    0.055 |           0.090 | 1344.000 |    121.000 |
| ('Ours GEFS cycles (06Z)', 'Ours GEFS members (share with +30 kt / 24 h)')       |   0.071 |                  0.140 |   0.799 |    0.053 |           0.090 | 1344.000 |    121.000 |
| ('Ours GEFS cycles (06Z)', 'FNV3 members (share with +30 kt / 24 h)')            |   0.065 |                  0.219 |   0.852 |    0.056 |           0.090 | 1344.000 |    121.000 |
| ('Ours IFS-ENS cycles (12Z)', 'Climatology')                                     |   0.085 |                  0.000 | nan     |    0.060 |           0.092 | 1315.000 |    121.000 |
| ('Ours IFS-ENS cycles (12Z)', 'RI model (calibrated)')                           |   0.068 |                  0.195 |   0.888 |    0.067 |           0.092 | 1315.000 |    121.000 |
| ('Ours IFS-ENS cycles (12Z)', 'RI model (uncalibrated)')                         |   0.068 |                  0.191 |   0.893 |    0.056 |           0.092 | 1315.000 |    121.000 |
| ('Ours IFS-ENS cycles (12Z)', 'Ours IFS-ENS members (share with +30 kt / 24 h)') |   0.069 |                  0.183 |   0.857 |    0.051 |           0.092 | 1315.000 |    121.000 |
| ('Ours IFS-ENS cycles (12Z)', 'FNV3 members (share with +30 kt / 24 h)')         |   0.065 |                  0.235 |   0.882 |    0.049 |           0.092 | 1315.000 |    121.000 |

Reliability, RI model (calibrated) (all 2025–26 cycles):

| p           |    cases |   mean forecast |   observed frequency |
|:------------|---------:|----------------:|---------------------:|
| [0.0, 0.05) | 4039.000 |           0.018 |                0.026 |
| [0.05, 0.1) |  324.000 |           0.064 |                0.139 |
| [0.1, 0.2)  |  511.000 |           0.150 |                0.249 |
| [0.2, 0.3)  |  171.000 |           0.239 |                0.444 |
| [0.3, 0.5)  |  160.000 |           0.377 |                0.450 |
| [0.5, 1.0)  |  137.000 |           0.590 |                0.489 |

Reliability, FNV3 members (all 2025–26 cycles):

| p           |    cases |   mean forecast |   observed frequency |
|:------------|---------:|----------------:|---------------------:|
| [0.0, 0.05) | 4097.000 |           0.005 |                0.029 |
| [0.05, 0.1) |  342.000 |           0.069 |                0.143 |
| [0.1, 0.2)  |  406.000 |           0.136 |                0.197 |
| [0.2, 0.3)  |  225.000 |           0.237 |                0.333 |
| [0.3, 0.5)  |  188.000 |           0.373 |                0.521 |
| [0.5, 1.0)  |   84.000 |           0.611 |                0.821 |

Next-24-h product only (τ = 24 h, 1043 cases, 106 RI): Brier calibrated model 0.068, climatology 0.093, AUC 0.89.

_Run time 93 s._
