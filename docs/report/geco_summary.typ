// Two-page summary of the GECO architecture. Build:
//   .venv/bin/python -c "import typst; typst.compile('docs/report/geco_summary.typ', output='docs/report/geco_summary.pdf', font_paths=['/usr/share/fonts'])"
#let ink2 = rgb("#52514e")
#let rule = rgb("#d6d5d0")
#set document(title: "GECO: architecture summary")
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.75in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true, leading: 0.58em, spacing: 0.85em)
#show heading.where(level: 1): it => { v(0.35em); set text(11.5pt); it; v(0.1em) }
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt + ink2) } else { (bottom: 0.3pt + rule) }, inset: (x: 5pt, y: 3pt))
#show table: set text(8.5pt)
#show table: set par(justify: false)
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)

#align(center)[
  #text(15pt, weight: "bold")[GECO: a calibrated GEFS intensity ensemble from public real-time data]
  #v(0.15em)
  #text(9.5pt, fill: ink2)[GEFS Ensemble, Consensus-Oriented · architecture summary · 12 November 2026 · North Atlantic and East Pacific]
]

= What it is
GECO turns the 30 GEFS member tracks that NHC publishes in real time into an intensity ensemble that is centred correctly, has a
realistic and storm-specific spread, and keeps each member's physically consistent scenario (when it intensifies, peaks and weakens). It
runs at every synoptic cycle (00, 06, 12, 18 UTC), uses only NHC's public files, and is ready within the early-cycle window, like NHC's
own consensus aids. Each output member is
$ "member"_i (tau) = "HCCA"(tau) + k dot hat(sigma)(tau) dot z_i (tau), $
where HCCA is NHC's corrected consensus forecast (the _centre_), $hat(sigma)$ is a predicted forecast error (the _size_ of the spread)
and $z_i$ is the member's normalised scenario from the GEFS (the _shape_). Each of the three comes from the source that does it best.
Its twin, IECO, is the same construction on the 50 member tracks that ECMWF publishes for its ensemble (IFS-ENS) on its open-data
mirror; over 2018–26 it scores within 0.04 kt of GECO.

= Inputs (all available by about $t_0$ + 3 h)
#table(columns: (auto, 1fr), align: left,
  table.header[*Input*][*Source and use*],
  [GEFS member tracks (AP01–30)], [NHC's public a-decks; the newest run at least 6 h old (the 00 UTC run serves the 06 UTC cycle), leads shifted by
    the run's age. Provides the scenario shapes $z_i$.],
  [HCCA], [NHC's corrected consensus, in the same public a-decks at every cycle. Provides the centre. It draws on ECMWF guidance
    that NHC does not publish in real time.],
  [Storm state (CARQ) and early aids], [NHC's operational analysis at $t_0$ and the twelve interpolated aids (GFS, GEFS mean, UKMET,
    Canadian, Navy, HWRF/HMON/HAFS, COAMPS-TC, DSHP, LGEM). Feed the spread model.],
  [Our consensus correction (TCCA)], [A LightGBM correction of the early-aid mean (no ECMWF, no SHIPS). Its output and quantile width
    feed the spread model, and it is the fallback centre when HCCA is missing (0.3% of 2025–26 forecasts).],
)

= Processing steps
*1. Scenario shapes.* For member $i$ at lead $tau$ (every 12 h to 168 h), take its departure from the ensemble mean and re-anchor it so
that every member starts together at $t_0$:
$d_i (tau) = [v_i (tau) - overline(v)(tau)] - [v_i (0) - overline(v)(0)]$. Smooth $d_i$ along lead with a 6-h Gaussian kernel (this gives
the members the step-to-step statistics of real best tracks) and divide by the forecast's own spread of smoothed departures, itself
smoothed along lead (18 h), to get $z_i$ with unit spread. The raw GEFS intensity level, about 25 kt too weak, is discarded; only the
shape is kept.

*2. Spread size.* A LightGBM regression predicts $hat(sigma)$, the expected absolute error of a consensus forecast for this storm, cycle
and lead. Features: the disagreement among the early aids, the spread of the GEFS member tracks, the storm's intensity, recent trend,
position and distance to land, the lead time, and our consensus correction's quantile width and predicted 24-h change. It is trained on
out-of-fold forecast errors from 2018–2024 (leave-one-season-out), so it learns from seven seasons of realistic errors. $hat(sigma)$ is
smoothed along lead (18 h) and floored at 2 kt.

*3. Centre.* HCCA as issued at its 12-hourly leads, with no further smoothing. Gaps between HCCA leads are interpolated; past HCCA's last
lead the last HCCA-minus-TCCA offset is carried forward.

*4. Combine.* $"member"_i = "HCCA" + k dot hat(sigma) dot z_i$, clipped to 10–200 kt. The factor $k = 1.2$ (constant with lead) converts a
mean absolute error into an ensemble spread; it is the value that minimises the fair CRPS on out-of-fold 2018–2024 cases. The full
fitted spread is used. Because the $z_i$ average to zero, the ensemble mean equals HCCA: GECO adds a distribution, not a new mean.

#table(columns: (auto, 1fr, auto), align: left,
  table.header[*Fitted component*][*Training data*][*Validation*],
  [Consensus correction (feeds $hat(sigma)$)], [2018–2024, leave-one-season-out], [2025–26 held out],
  [Spread model $hat(sigma)$], [out-of-fold errors 2018–2024, leave-one-season-out], [2025–26 held out],
  [Spread factor $k$], [out-of-fold CRPS 2018–2024], [refit without each season: 1.2–1.3],
  [Smoothing widths (6 h, 18 h)], [chosen on step-to-step statistics vs the best track], [—],
)

= How well it works
Verified against NHC's best track (2026: working best track), leads 12–120 h, all four cycles, identical cases for every system, 95%
storm-block bootstrap intervals. For 2018–2024 every fitted part is out of sample for the season scored. The fair probabilistic
yardstick is "dressed HCCA": HCCA as a normal distribution whose width at each lead is HCCA's own error in the other seasons.

#table(columns: (auto, 1fr, 1fr, 1fr), align: (left, right, right, right),
  table.header[*Fair CRPS (kt; lower is better)*][*2018–24*][*2025–26*][*All 2018–26*],
  [Raw GEFS members (offset to $t_0$)], [12.69], [14.61], [13.10],
  [Dressed OFCL], [7.49], [7.68], [7.53],
  [Dressed HCCA], [7.33], [7.58], [7.39],
  [*GECO*], [*6.97*], [*7.26*], [*7.04*],
  [GECO − dressed HCCA], [−0.36 [−0.43, −0.29]], [−0.32 [−0.43, −0.20]], [−0.36 [−0.41, −0.29]],
  [GECO 10–90% coverage (ideal 80%)], [77%], [76%], [76%],
)
#text(8.5pt)[336 storms, 27,623 cases. GECO beats dressed HCCA in each of the nine seasons separately, each significantly, across the
GEFS upgrade from 20 to 30 members (2020) and the HWRF-to-HAFS switch (2023).]

*Against FNV3 (Google DeepMind), 2025–26.* GECO's CRPS is about 0.8 kt lower than that of FNV3's 50 members taken at the same early
timing (7.6 against 8.4 kt): −0.86 kt [−1.69, −0.04] on GECO's own cases, −0.80 kt [−1.65, +0.02] on the 6,345 cases where IECO also exists;
significant in 2026 on its own. GECO is better or level at every lead to 120 h; its mean bias is
−1.2 kt against FNV3's −6.0 kt; its spread is 0.87 of its error against 0.69 for FNV3, which is over-confident.

*Physical consistency.* GECO members change by 9.0 kt per 12 h on average (best track 8.6 kt), zig-zag less often than the best track
(0.025 vs 0.034) and exceed 30 kt in 12 h about as often (2.1% vs 1.9%).

= Real-time operation and limits
At each cycle: read NHC's a-decks, take the newest GEFS run at least 6 h old, compute $z_i$, run the spread model, read HCCA, combine.
It is a few seconds of computation, with no gridded data. Arrival times of HCCA and the GEFS member tracks relative to the forecast deadline
have not yet been measured.

Limits: the mean is exactly HCCA, so GECO inherits HCCA's errors, including its low bias during rapid intensification. The spread
is symmetric about the centre, so rapid intensification still falls above the 90th percentile more often than nominal. The spread
model has not been retrained on HCCA's errors (it predicts our consensus correction's errors, and $k$ absorbs the difference).

#text(8.5pt, fill: ink2)[Code: `src/tcic/ensemble/geco.py`, `scripts/ens_geco.py`; nine-season evaluation `scripts/ens_multiseason.py`,
`docs/ensemble_multiseason.md`; IECO `scripts/ens_ieco.py`; FNV3 comparison `scripts/fnv3_compare.py`; full write-up `docs/report/report.pdf`.]
