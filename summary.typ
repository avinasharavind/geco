// python -c "import typst; typst.compile('summary.typ', output='summary.pdf', font_paths=['/usr/share/fonts'])"

#let ink2 = rgb("#52514e")
#let rule = rgb("#d6d5d0")
#set document(title: "GECO/IECO Summary")
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.75in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true, leading: 0.58em, spacing: 0.85em)
#show heading.where(level: 1): it => { v(0.35em); set text(11.5pt); it; v(0.1em) }
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt + ink2) } else { (bottom: 0.3pt + rule) }, inset: (x: 5pt, y: 3pt))
#show table: set text(8.5pt)
#show table: set par(justify: false)
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)

#align(center)[
  #text(15pt, weight: "bold")[Correcting Numerical Weather Model Ensembles for
                                
                            Tropical Cyclone Intensity Prediction]
  #v(0.25em)
]
#v(0.6em)

Existing numerical weather prediction (NWP) models are often separated based on their coverage, resolution, and strengths. 
In the realm of tropical cyclone (TC) forecasting, global NWP models are generally regarded as too spatially coarse to properly represent
the structure and intensity of a TC, instead being used mainly for track prediction while specialized, higher-resolution models are used for intensity
forecasting. This is especially true for global NWP ensembles, which provide a set of perturbed forecasts but often at a lower resolution than the single
deterministic forecast to save on compute costs. However, disregarding these ensemble members removes potentially valuable information about the uncertainty
present in a TC intensity forecast, ultimately weakening the information a forecaster has on hand to develop their prediction.

In this work, we present a framework for correcting these global NWP ensembles to produce a suite of realistic TC intensity forecasts while maintaining the
forecast uncertainty information in the perturbed members. We use this framework to introduce two new forecast systems: GECO, the GFS Ensemble, Consensus-Oriented,
and IECO, the IFS Ensemble, Consensus-Oriented. Both systems show competitive results alongside the state-of-the-art ensemble forecast system (Google DeepMind's FNV3)
with superior ensemble member spread and more physically consistent members, providing valuable information to forecasters about probabilistically extreme outcomes
for a given TC intensity forecast.

= Overall Framework
To generate corrected ensemble forecasts, we use the following procedure:
#v(0.1em)
* 1. Retrieve pre-corrected model intensity forecasts and other storm-specific variables. *
#v(0.1em)
These include the raw GEFS and IFS-ENS intensity forecasts (published by NOAA and the ECMWF, respectively), the National Hurricane Center's ATCF "A-deck" files 
containing other model forecast products, and information about the storm's current state (via CARQ). 
#v(0.1em)
*2. Calculate each member's spread within its respective ensemble. *
#v(0.1em)
For each ensemble member and at each forecast point, we calculate its position relative to the mean of that ensemble. As part of this step, we introduce an 18-hour
smoothing kernel to prevent the amplification of short-term variations between the ensemble's timesteps.
#v(0.1em)
*3. Predict the expected forecast spread at a given timestep. *
Here, we use a LightBGM model trained on storm information, timing information, and the spread in several of the "early" aids available in the NHC "A-deck" forecast
files to predict a scaling parameter $sigma_i$ at each forecast timestep $i$. 
#v(0.1em)
*4. Scale and re-center the ensemble. *
#v(0.1em)
Instead of training a new forecast to center our shifted ensemble, we simply use the HCCA consensus model routinely and automatically produced by the NHC and included
in the "A-deck" files. The HCCA routinely scores as one of the strongest intensity forecast models as a weighted consensus mean of many other forecast products. 
From the HCCA mean, we add each ensemble member's deviation from its mean, scaled by the forecast spread $sigma_i$ and a constant $k=1.2$. Thus, for ensemble member
m and its modeled forecast intensity at timestep i $m_i$, along with the HCCA mean at that timestep $H_i$, the corrected forecast is:

$C_(m,i) = H_i + k * sigma_i * m_i$. 

This process is repeated for all ensemble members, separately for both ensembles, to produce the new corrected forecast product.
#v(0.6em)

= Real-Time Application
#v(0.1em)
This framework is designed to operate entirely with near-real-time information, available publicly. GEFS forecasts take approximately 6 hours to become available, while
IFS-ENS forecasts take closer to 9 hours, meaning these corrected ensembles will become available approximately 9 hours and 12 hours after model initialization time, respectively.
While an operational implementation of GECO and IECO are not yet available, our intention is to make this available soon.
// ------------------------------------------------------------------ 1
= Results
To validate GECO and IECO, we compare its performance against a probabilistic "HCCA-ENS" (created by generating members from a Normal distribution centered on the HCCA whose standard deviation 
is the HCCA's error over the training period, 2018-2024), and against the FNV3 in 2025 and 2026 when the latter is present. Our primary metric is CRPS, an extension of MAE for ensemble 
systems, though we also consider more simple metrics of consistency. The below table summarizes some statistics; for CRPS, a lower score is better, and for its difference to the HCCA-ENS a 
more negative number is better. For Bias, a number closer to zero is better, and for spread/error, a number closer to 1 is better (Here, spread/error is the spread in the ensemble members 
divided by the RMSE of HCCA). For the "track behavior" columns, numbers closer to the Best Track row at the bottom are better.
#v(0.6em)
#figure(
  {
    set text(8pt)
    set par(justify: false)
    let sig = [\*]
    table(
      columns: 9,
      align: (x, y) => if x == 0 { left } else { center },
      stroke: none,
      inset: (x: 5pt, y: 3pt),
      table.hline(stroke: 0.7pt),
      table.vline(x: 1, stroke: 0.4pt), table.vline(x: 3, stroke: 0.4pt), table.vline(x: 7, stroke: 0.4pt),
      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*System*],
        table.cell(colspan: 2)[*2018–26*],
        table.cell(colspan: 4)[*2025–26*],
        table.cell(colspan: 2)[*Trace Behavior, 2025–26*],
        table.hline(start: 1, end: 9, stroke: 0.4pt),
        [CRPS], [− dressed HCCA], [CRPS], [− FNV3], [Bias], [Spread / error], [Zig-zag rate], [Mean |12-h Δv|],
      ),
      table.hline(stroke: 0.5pt),
      [*GECO* (GEFS)], [*7.04*#super[a]], [−0.36#sig], [*7.59*], [−0.80#super[c]], [−1.2], [0.87], [0.025], [9.1],
      [*IECO* (IFS-ENS)], [*7.19*#super[b]], [−0.28#sig], [*7.64*], [−0.74#super[c]], [−1.3], [0.87], [0.017], [8.9],
      [Dressed HCCA], [7.39#super[a] / 7.46#super[b]], [], [7.93], [], [−1.2], [0.96], [—], [—],
      [FNV3 (early)], [—], [], [8.39], [], [−6.0], [0.69], [0.072], [10.1],
      [Raw GEFS], [19.49#super[a]], [], [22.64], [], [−25.5], [0.20], [0.014], [5.2],
      [Raw IFS-ENS], [17.58#super[b]], [], [16.94], [], [−20.7], [0.43], [0.037], [7.0],
      [Best track], [], [], [], [], [], [], [0.033], [8.6],
      table.hline(stroke: 0.7pt),
    )
  },
  caption: [Summary of verification (kt; fair CRPS, lower is better; identical cases within each column group).

  #super[a] and #super[b]: Due to data availability, the GEFS and IFS-ENS training data have different sample sizes.

  \*Significant to a 95% confidence interval. 
  
  #super[c]Not significant over 2025–26; significant in 2026 alone.

  Zig-zag rate: two consecutive 12-h changes of opposite sign, each ≥ 10 kt],
)

= Case Studies
Several case studies were also investiaged; below is a figure for one from Hurricane Erin (N. Atlantic, 2025)
#figure(image("docs/report/figures/fig_case_1.svg", width: 100%), caption: [Hurricane Erin, 14 August 2025, two days before its rapid intensification.])
