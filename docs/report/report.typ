// Project write-up: GECO and IECO, calibrated intensity ensembles from public real-time physical-model guidance.
// Build:  .venv/bin/python scripts/report_figures.py && .venv/bin/python scripts/report_figures_ens.py && \
//         .venv/bin/python scripts/case_plots.py && \
//         .venv/bin/python -c "import typst; typst.compile('docs/report/report.typ', output='docs/report/report.pdf', font_paths=['/usr/share/fonts'])"
// (report_figures.py only for the verification-check numbers and figure in Appendix A.)

#let n = json("numbers.json")
#let e = json("numbers_ens.json")
#let f2(x) = str(calc.round(x, digits: 2)).replace("-", "−")
#let pct(x) = str(calc.round(100 * x, digits: 0)) + "%"
#let diff(d) = [#f2(d.at(0)) kt [#f2(d.at(1)), #f2(d.at(2))]]
#let mag(d) = f2(-d.at(0))

#let ink2 = rgb("#52514e")
#let rule = rgb("#d6d5d0")
#let accent = rgb("#2a78d6")

#set document(title: "GECO and IECO: calibrated intensity ensembles from public real-time physical-model guidance")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1",
  header: context { if counter(page).get().first() > 1 [#set text(8pt, fill: ink2); GECO and IECO · project write-up #h(1fr) 12 Nov 2026] })
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em, spacing: 0.95em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.6em); set text(13pt); it; v(0.2em) }
#show heading.where(level: 2): it => { v(0.3em); set text(11pt); it }
#show figure.caption: set text(9pt)
#show figure: set block(spacing: 1.1em)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt + ink2) } else { (bottom: 0.3pt + rule) }, inset: (x: 5pt, y: 3.5pt))
#show table: set text(8.5pt)
#show table: set par(justify: false)
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
#show link: set text(fill: accent)

#let csvtable(path, cols: auto, aligns: none) = {
  let rows = csv(path)
  let ncol = rows.at(0).len()
  table(
    columns: if cols == auto { (auto,) + (1fr,) * (ncol - 1) } else { cols },
    align: if aligns != none { aligns } else { (x, y) => if x == 0 { left } else { right } },
    table.header(..rows.at(0).map(h => strong(h))),
    ..rows.slice(1).flatten().map(c => if c.starts-with("-") { "−" + c.slice(1) } else { c.replace(" -", " −").replace("[-", "[−") }),
  )
}
#let note(body) = block(fill: rgb("#f3f2ee"), inset: 10pt, radius: 3pt, width: 100%, body)

// ------------------------------------------------------------------ title
#align(center)[
  #text(17pt, weight: "bold")[GECO and IECO: calibrated intensity ensembles\ from public real-time physical-model guidance]
  #v(0.3em)
  #text(10.5pt, fill: ink2)[Project write-up · status as of 12 November 2026 · North Atlantic and East Pacific]
]
#v(0.6em)

#note[
  *Summary.* The raw intensity forecasts of the physical ensembles, GEFS and the ECMWF ensemble (IFS-ENS), are far too weak and their
  members far too similar to be used as probabilistic guidance. GECO and IECO (GEFS / IFS-ENS Ensemble, Consensus-Oriented) are two twin
  products that turn them into calibrated intensity ensembles, using only data that are public in real time and available within the
  early-cycle window. Each member is NHC's corrected consensus HCCA plus that member's normalised scenario from the physical ensemble, scaled by
  a storm-specific predicted error. GECO uses the 30 GEFS member tracks in NHC's public files; IECO uses the 50 IFS-ENS member tracks
  that ECMWF publishes on its open-data mirror. Both run at every cycle from runs at least 6 h old, and neither needs any gridded data.

  The fair yardstick is the best guidance already available, made probabilistic: "dressed HCCA". Over nine seasons (2018–2026) GECO beats
  it in every season, by #mag(e.geco_all_hccad) kt of CRPS on average (#e.geco_storms storms, every season significant); IECO beats it by
  #mag(e.ieco_all_hccad) kt over 2018–2026 (#e.ieco_storms storms, every season from 2020 significant). On 2025–26, with
  FNV3 (Google DeepMind) available, both are about 0.8 kt better than FNV3's members at the same early timing, borderline significant over
  both seasons and significant in 2026, with a mean bias of about −1 kt against FNV3's −6 kt. Their members evolve as smoothly as real storms.
  Rapid intensification remains the main weakness, as for every guidance system.
]

// ------------------------------------------------------------------ 1
= Motivation and design principles

AIFS-TC (Allen et al. 2026) showed that a cheap learned correction can lift an AI weather model's tropical-cyclone intensity forecasts
to the operational frontier. This project asks whether the physical-model ensembles that forecasters already use can be turned, in the
same spirit, into usable probabilistic intensity guidance. Their tracks are good, but at their resolution their intensities are 20–25 kt
too weak on average and their members disagree far too little: in 2025–26 the raw ensembles' 10–90% ranges contain only 20–45% of
outcomes.

Four principles shaped the design. Everything is built for the *early cycle*: an input may only be used if it is available within about
three hours of the synoptic time, as for NHC's own consensus aids and the official forecast. Only *public data* are used. The products
must run in *real time*. And every comparison uses NHC's verification conventions on *homogeneous* samples with storm-block bootstrap
confidence intervals (Appendix A). FNV3 is a benchmark only and never an input. The target is the best guidance already available, so
results are judged against HCCA and FNV3; GECO and IECO are two parts of one system, not competitors.

// ------------------------------------------------------------------ 2
= System

#figure(image("figures/fig_architecture.svg", width: 100%), caption: [GECO and IECO. Each member takes its scenario shape from the
physical ensemble (1), its centre from NHC's corrected consensus HCCA (2) and its spread from a storm-specific spread model (3).]) <fig-arch>

@fig-arch shows the construction and @tab-inputs the inputs. All are public in real time.

#figure(table(columns: (auto, 1fr), align: left,
  table.header[*Input*][*Source and use*],
  [GEFS member tracks (AP01–30)], [NHC's tracker output in its public a-decks (`atcf/aid_public`), every cycle. GECO's shapes.],
  [IFS-ENS member tracks (50)], [ECMWF's own tracker output, BUFR files on the AWS open-data mirror (`s3://ecmwf-forecasts`), all four runs
    from January 2023; NHC's archived copies (EE01–50, identical values) before. IECO's shapes.],
  [HCCA], [NHC's corrected consensus, in the public a-decks at every cycle. The centre. It draws on ECMWF guidance that NHC has
    internally.],
  [Storm state (CARQ), early aids], [NHC's analysis at $t_0$ and twelve interpolated aids (GFS, GEFS mean, UKMET, Canadian, Navy, HWRF/HMON/HAFS,
    COAMPS-TC, DSHP, LGEM). Inputs to the spread model.],
  [Consensus correction (TCCA)], [Our LightGBM correction of the early-aid mean (no ECMWF, no SHIPS). Its quantile width and predicted
    24-h change feed the spread model; it fills HCCA's gaps between and beyond its leads, and is GECO's fallback centre when HCCA is missing.],
), caption: [Inputs. For both ensembles the newest run at least 6 h old is used, with lead times shifted by its age.]) <tab-inputs>

The member runs follow the early-cycle rule: the newest run that is at least 6 h old at the cycle time. NHC's GEFS tracks are in its
files within that window. ECMWF posts its ensemble tracks 6.5–7.6 h after the start of the 00 and 12 UTC runs in 2025–26 (about 8.7 h in 2024) and
about 6 h after the 06 and 18 UTC runs (AWS timestamps), so the 6-h-old run is available before every advisory, 9 h after its
start, though with a thin margin in 2024. NHC's archive holds ECMWF member tracks
mostly for the 00 and 12 UTC runs, so before 2023 IECO uses a 6-h-old run at half the cycles and a 12- or 18-h-old run at the others;
from 2023 it is 6 h old at every cycle (#pct(e.ieco_6h_share) of IECO's cases overall).

// ------------------------------------------------------------------ 3
= Construction

*Shape from the physical ensemble.* For member $i$ at lead $tau$ (every 12 h) the departure from the ensemble mean is re-anchored so that
every member starts together at $t_0$, and smoothed along lead with a 6-h Gaussian kernel:
$ d_i (tau) = [v_i (tau) - overline(v)(tau)] - [v_i (0) - overline(v)(0)], quad tilde(d)_i = "smooth"(d_i), quad z_i (tau) = tilde(d)_i (tau) slash "sd"(tau), $
where $"sd"(tau)$ is the spread of the smoothed departures within that forecast, itself smoothed along lead (18 h). The $z_i$ keep only
the ensemble's scenario structure: which members intensify faster or slower, and when. The raw intensity level, about 20–25 kt too weak, is
discarded. The 6-h kernel makes the members' step-to-step behaviour match the best track's (section 4.4) at no cost in skill.

*Size from a spread model.* A LightGBM regression predicts $hat(sigma)$, the expected absolute error of a consensus forecast for this
storm, cycle and lead. Its features are the disagreement among the early aids, the spread of NHC's GEFS member tracks, the storm's
intensity, recent trend, position and distance to land, the lead time, and the consensus correction's quantile width and predicted
24-h change; all #e.spread_n_features inputs are listed in Appendix B. It is trained on the absolute out-of-fold errors of the consensus
correction from 2018–2024 (leave-one-season-out), smoothed along lead (18 h) and floored at 2 kt. On
2025–26 its prediction correlates with the actual error at #e.spread_corr (#e.spread_corr_within_lead within a fixed lead) and ranks forecasts
well (@tab-spreadbins).

#figure(csvtable("tables/spread_bins.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [2025–26 forecasts binned by predicted error $hat(sigma)$, against the consensus correction's actual mean absolute error.]) <tab-spreadbins>

*Centre.* HCCA as issued at its 12-hourly leads, without smoothing. Between HCCA's leads the HCCA-minus-TCCA offset is interpolated, and
past its last lead the last offset is carried forward. Forecasts without HCCA (0.3% in 2025–26) fall back to our own consensus for GECO and
are dropped for IECO.

*Combine.* $ "member"_i (tau) = "HCCA"(tau) + k dot hat(sigma)(tau) dot z_i (tau), $
clipped to 10–200 kt. Because the $z_i$ have unit spread, the members' standard deviation is $k hat(sigma)$; $k$ converts a predicted
_mean absolute_ error into a _standard deviation_. For normally distributed errors that factor is $sqrt(pi slash 2) approx 1.25$. $k$ is
fitted, as $k(tau) = c_0 + c_1 tau slash 100$, by minimising the fair continuous ranked probability score (CRPS) on out-of-fold 2018–2024
cases. The fit gives $c_1 = 0$ and $k = 1.2$ for both products, close to the normal-theory value, and 1.2–1.3 when refitted without any
single season. The minimum is shallow: any $k$ from 1.1 to 1.3 is within 0.02 kt of it (Appendix C). Because the $z_i$ average to zero, the ensemble mean is HCCA: the products add a distribution, not a new mean.

This design separates _where_ the ensemble is centred, _how wide_ it is and _how its members differ_, and takes each from the source that
does it best. It replaced earlier versions that centred the members on our own consensus correction (with or without gridded IFS-ENS
features), which were level with dressed HCCA over 2018–2024 but fell behind it in 2026, and an IECO built from our own tracker on the
gridded IFS-ENS fields, which could use only one run a day. Because the spread comes from the consensus error model and the raw ensembles
contribute only their shapes, model upgrades that change a raw ensemble's spread (the ECMWF ensemble's roughly doubled between 2024 and
2025) do not change the products' spread.

// ------------------------------------------------------------------ 4
= Results

== Yardsticks
Verification uses NHC's best track (2026: the working best track), leads 12–120 h, all four cycles and identical cases for every system in a
comparison. For 2018–2024 every fitted ingredient is out of sample for the season scored: the consensus correction, the spread model and $k$
are fitted without that season. 2025–26 use the production models, trained on 2018–2024. The primary yardstick is *dressed HCCA*: HCCA as a
normal distribution whose width at each lead is HCCA's root-mean-square error in the other 2018–2024 seasons. Scoring HCCA as a single value
would flatter any ensemble, whose CRPS is always below its own mean's absolute error. The second yardstick is *FNV3*, Google DeepMind's
50-member ensemble, taken from its newest run at least 6 h old, the same timing rule as ours; it is available for 2025–26.

== Nine seasons against dressed HCCA
#figure(csvtable("tables/seasons.csv", cols: (auto, auto, 1fr, 1fr, 1.9fr, auto, 1fr, 1fr, 1.9fr)),
  caption: [Fair CRPS (kt) by season. Each product is scored on its own sample (GECO: every cycle with NHC's GEFS tracks and HCCA; IECO:
  every cycle with ECMWF member tracks and HCCA), each against dressed HCCA on the same cases, with 95% storm-bootstrap intervals. NHC's
  archive has ECMWF member tracks for only one storm in each of 2018 and 2019.]) <tab-seasons>

GECO beats dressed HCCA in all nine seasons, each significantly, by #diff(e.geco_all_hccad) overall (@tab-seasons). The gain is steady across
the change from the 20-member GEFS to the 30-member GEFSv12 in September 2020 and the switch from HWRF to HAFS in 2023. IECO beats dressed HCCA
in every season from 2020, each significantly, by #diff(e.ieco_all_hccad) overall, and by the same amount whether its run is 6 h old or
older. Both are well calibrated: #pct(e.geco_all_cov) (GECO) and #pct(e.ieco_all_cov) (IECO) of outcomes fall inside the 10–90% range, against
80% for a perfect ensemble, and IECO's spread is #e.ieco_all_spread_err of its error. On identical cases the two are within
#f2(e.ieco_all_vs_geco.at(0)) kt of each other (#diff(e.ieco_all_vs_geco)). Since both means are HCCA, the gain comes entirely from the
members' scenario shapes and the storm-specific spread, and the GEFS and IFS-ENS shapes turn out to be about equally informative.

#figure(image("figures/fig_crps_by_lead.svg", width: 100%), caption: [Fair CRPS by lead time on identical cases. Left: 2018–2026
(#e.all_common_cases cases where both products exist). Right: 2025–26 with FNV3 members (#e.fnv3_cases cases).]) <fig-lead>

The gain over dressed HCCA holds at every lead from 12 to 120 h (@fig-lead).

== 2025–26 against FNV3
#figure(csvtable("tables/ensemble_summary.csv", cols: (1.5fr, auto, auto, auto, auto, 1.6fr, 1.6fr)),
  caption: [2025–26, all cycles, leads 12–120 h, identical cases for every row (#e.fnv3_storms storms, #e.fnv3_cases cases). Spread / error
  is the mean member spread divided by the root-mean-square error of the ensemble mean (1 for a well-dispersed ensemble). Differences in
  CRPS with 95% storm-bootstrap intervals.]) <tab-ens>

On the cases where all systems are available (@tab-ens), the CRPS difference from FNV3 is #diff(e.geco_2526_vs_fnv3) for GECO and
#diff(e.ieco_2526_vs_fnv3) for IECO: both intervals just reach zero over the two seasons. In 2026 both are significantly better
(#diff(e.geco_2026_vs_fnv3) and #diff(e.ieco_2026_vs_fnv3)); in 2025 the differences are smaller and uncertain. The interval for GECO
excludes zero on slightly different samples, for example cycles at which FNV3's late run also exists (`docs/fnv3_compare.md`). Both products
are better than FNV3 at every lead to 96 h, most clearly in the first two days, and slightly behind it at 108–120 h. FNV3's members are sharp but over-confident:
their spread is #f2(0.69) of their error, their mean is #str(calc.abs(e.bias_2526_fnv3_early)) kt too weak and only 62% of outcomes fall in
their 10–90% range. GECO and IECO are nearly unbiased (#e.bias_2526_geco and #e.bias_2526_ieco kt, as HCCA), and their spread follows their error
at every lead, a little below it (0.87 overall; @fig-spread). Their differences from dressed HCCA on these cases are
#diff(e.geco_2526_vs_hccad) (GECO) and #diff(e.ieco_2526_vs_hccad) (IECO). The raw ensembles are two to three times worse than either.

#figure(image("figures/fig_spread_vs_error.svg", width: 100%), caption: [Mean member spread (solid) and the error of the ensemble mean
(dashed) by lead time, 2025–26, same cases as @tab-ens. A well-calibrated ensemble has the two close together. The raw ensembles' errors
(20–40 kt) are off the scale and only their spread is shown.]) <fig-spread>

== Physical consistency
#figure(csvtable("tables/roughness.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [Step-to-step behaviour of intensity traces, 2025–26, leads 0–120 h. A zig-zag is two consecutive 12-h changes of opposite sign,
  each at least 10 kt.]) <tab-rough>

Scores alone do not show whether individual members look like plausible storms, so we also measured how traces evolve from one time step
to the next (@tab-rough). GECO and IECO members change by about as much per 12 h as the best track, exceed 30 kt in 12 h about as often, and
reverse direction less often. FNV3's members zig-zag about twice as often as the best track at 12-h spacing, and at their native 6-h spacing
they reverse direction #str(calc.round(e.zigzag_6h.at("FNV3 members") / e.zigzag_6h.at("Best track"))) times as often, which is the
short-term jitter visible in their traces.

== Case studies
#figure(image("figures/fig_case_6.svg", width: 100%), caption: [Hurricane Erin, 15 August 2025, 12 UTC cycle, about 18 h before its
extreme rapid intensification (65 kt to 140 kt by 16 August 18 UTC). GECO (NHC's 30 GEFS member tracks) and IECO (ECMWF's 50 member
tracks) from the 06 UTC runs, FNV3's 06 UTC members, the best track and the official forecast of the same cycle. In grey, the raw members,
starting at the models' own intensity.]) <fig-case-erin>

#figure(image("figures/fig_case_2.svg", width: 100%), caption: [Hurricane Melissa, 25 October 2025, 12 UTC cycle, at the start of its rapid
intensification to 165 kt.]) <fig-case-melissa>

The cases show what the numbers mean in practice. Every panel is for the same cycle, as the products would run in real time. For Erin
(@fig-case-erin) the spread model recognises a volatile situation: both ensembles fan out widely in the first day and their upper members
reach the 140-kt peak, whereas FNV3's members stay mostly below about 125 kt and the official forecast peaks near 120 kt. For Melissa
(@fig-case-melissa) both ensembles rise quickly from the observed intensity and fan out; their means, which are HCCA, peak well below the
165-kt best track, but the upper members reach 140–170 kt, so the extreme outcome lies within the ensemble. FNV3's mean comes closer to the
peak. Eight further cases are in Appendix D. A larger set of 39 cases, including eleven high-profile storms from 2018–2024 (Michael, Florence,
Dorian, Laura, Iota, Ida, Ian, Lee, Otis, Beryl, Milton) scored out of sample, is in `docs/case_plots.pdf`.

// ------------------------------------------------------------------ 5
= Real-time operation

At each cycle the products need NHC's public a-decks (CARQ, the early aids, HCCA and the GEFS member tracks) and ECMWF's newest ensemble
track file at least 6 h old. All are small text or BUFR files of a few MB at most. The computation (shapes, the spread model, combination) takes
seconds on one core, and no gridded data are read. The inputs are therefore all available in time, but two pieces are not yet in place: the
arrival times of HCCA and the GEFS member tracks relative to the forecast deadline have not been measured (a poller is ready for the next
active storm), and the trained models are not yet saved for a stand-alone real-time runner.

// ------------------------------------------------------------------ 6
= Limitations and next steps

The products are symmetric around HCCA, which, like every guidance system, under-predicts rapid intensification. In such situations the
truth still falls above the 90th percentile more often than it should, and some lower members describe weakening that does not happen.
Making the spread asymmetric did not help, because which borderline situations will intensify rapidly cannot be identified well enough in
advance. Both products inherit HCCA's errors in the mean. The spread model predicts our consensus correction's errors rather than
HCCA's, and $k$ absorbs the difference. The test seasons contain about 60 storms, so only large differences are significant, and the
2026 verification uses NHC's working best tracks. Before 2020, NHC's archive holds ECMWF member tracks for only two storms.

The next steps, in order:
- A real-time runner: persist the trained models, poll NHC's public files and ECMWF's track files, and write ATCF-format members.
- Measure the arrival times of HCCA, the GEFS member tracks and ECMWF's track files during the next active storms.
- Re-score 2026 once NHC finalises its best tracks.
- A convolutional network on multi-level gridded fields, combined with this construction, when compute allows; the present products use no
  gridded data at all.

// ------------------------------------------------------------------ appendices
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")
#show heading.where(level: 1): it => { set text(13pt); [Appendix #counter(heading).display("A") · #it.body]; v(0.2em) }

= Verification framework
All verification follows NHC's conventions. Truth is the final best track (b-deck; 2026: the working best track); a case counts only if
the system is a tropical or subtropical cyclone at both the initial and the valid time; and every comparison is homogeneous. Ensembles are
scored by the fair continuous ranked probability score (CRPS), the coverage of the 10–90% range and the ratio of spread to the error of
the mean. Confidence intervals resample whole storms (block bootstrap), as in the AIFS-TC paper, because cases from the same storm are
strongly correlated.

As a check, the harness recomputes NHC's published 2024 Atlantic verification. The official-forecast table is reproduced with identical
case counts and errors within 0.1 kt, and the homogeneous early-model table with identical case counts and #n.m1_t5a_n_within_0p2 of
#n.m1_t5a_n errors within 0.2 kt (@fig-m1). The AIFS-TC paper's 2025 values for OFCL and FNV3 are reproduced within their confidence intervals.

#figure(image("figures/fig_m1_reproduction.svg", width: 40%), caption: [Recomputed versus published errors for NHC's 2024 homogeneous
early-aid verification; each point is one aid at one lead time.]) <fig-m1>

= Spread-model inputs
The spread model is a LightGBM regression (500 trees of 15 leaves, learning rate 0.03, at least 200 cases per leaf, 70% of features and
80% of cases per tree, L2 penalty 1) whose target is the absolute error of the consensus correction, $|v_("obs") - "TCCA"|$, at each storm,
cycle and lead (12–168 h). @tab-feat lists every input. All are available by $t_0$ + 3 h; missing values (an aid not run, a lead beyond
an aid's range) are left missing and handled by the trees. Early aids are NHC's interpolated versions, with the 6-h and 12-h versions
merged. There are no ECMWF, SHIPS or gridded inputs. The spread model never sees the IFS-ENS members: IECO uses the same $hat(sigma)$ as GECO.

#[#show figure: set block(breakable: true)
#figure(csvtable("tables/spread_features.csv", cols: (auto, auto, 1fr), aligns: (left, left, left)),
  caption: [Every input of the spread model (#e.spread_n_features variables; each early aid contributes two), one row per storm, cycle $t_0$
  and lead time.]) <tab-feat>]

= Choice of the spread factor $k$
$k$ is chosen by minimising the out-of-fold CRPS. @tab-k shows the CRPS, the coverage of the 10–90% range and the spread / error ratio
of both products over 2018–2024 for constant $k$ (leads 12–120 h; #e.k_cases.GECO GECO and #e.k_cases.IECO IECO cases). Three points
explain the choice of 1.2:
- *It is the CRPS minimum*, for both products. A narrower ensemble (smaller $k$) misses too many outcomes; a wider one is penalised for
  spreading probability where outcomes rarely fall. The minimum is shallow between 1.1 and 1.3 (within 0.02 kt).
- *It matches normal theory.* The members' standard deviation is $k hat(sigma)$ and $hat(sigma)$ predicts a mean absolute error, so for
  normally distributed errors the calibrated factor would be $sqrt(pi slash 2) approx 1.25$.
- *It is slightly sharper than a spread that matches the error.* The errors are heavier-tailed than normal (their root-mean-square is
  #e.rms_over_mae times their mean absolute value, against 1.25 for a normal distribution), so spread / error = 1 needs $k approx 1.3$–1.4.
  The CRPS prefers a slightly narrower ensemble, because $hat(sigma)$ ranks forecasts only imperfectly: the extra width would be spent
  mostly on cases that do not need it. The cost is coverage a little below 80% (76–78% at $k = 1.2$).
The fit allowed $k$ to vary linearly with lead time ($c_1$ from −0.4 to +0.4 in steps of 0.1; $c_0$ from 0.6 to 2.0 in steps of 0.1);
the best slope was zero, so $k$ is constant. Refitting without each 2018–2024 season in turn gave 1.2 or 1.3 with slopes of 0 to −0.1.

#figure(csvtable("tables/k_curve.csv", cols: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)),
  caption: [Out-of-fold 2018–2024 fair CRPS (kt), 10–90% coverage (ideal 80%) and spread / error (1 for a well-dispersed ensemble) for
  constant $k$. GECO and IECO are on their own cases, so their CRPS values are not directly comparable.]) <tab-k>

= Additional case studies
All panels are for the 12 UTC cycle: GECO and IECO from the 06 UTC runs, FNV3 from its 06 UTC run.
#figure(image("figures/fig_case_1.svg", width: 100%), caption: [Hurricane Erin, 14 August 2025, two days before its rapid intensification.])
#figure(image("figures/fig_case_3.svg", width: 100%), caption: [Hurricane Kiko, 1 September 2025, East Pacific: steady intensification to a
category 4 hurricane.])
#figure(image("figures/fig_case_4.svg", width: 100%), caption: [Hurricane Erin, 19 August 2025: recurvature and extratropical transition.])
#figure(image("figures/fig_case_5.svg", width: 100%), caption: [Tropical Storm Henriette, 6 August 2025: a weak, long-lived storm.])
#figure(image("figures/fig_case_7.svg", width: 100%), caption: [Hurricane Humberto, 25 September 2025: rapid intensification to 140 kt
within three days, then rapid weakening.])
#figure(image("figures/fig_case_8.svg", width: 100%), caption: [Hurricane Gabrielle, 18 September 2025: weak for four days, then rapid
intensification to 120 kt.])
#figure(image("figures/fig_case_9.svg", width: 100%), caption: [Hurricane Melissa, 22 October 2025, three to four days before its rapid
intensification.])
#figure(image("figures/fig_case_10.svg", width: 100%), caption: [Hurricane Priscilla, 5 October 2025, East Pacific: rise to 100 kt and decay
over cooler water.])

= Glossary and reproduction
/ CARQ: NHC's operational analysis of position and intensity at $t_0$, with the preceding 24 h.
/ OFCL: the official NHC forecast. HCCA: NHC's corrected consensus.
/ Early / late: available / not available in time for the forecast cycle that starts at the synoptic time.
/ FNV3: Google DeepMind's cyclone model (Weather Lab archive, 50 members).
/ CRPS: continuous ranked probability score, the probabilistic analogue of the mean absolute error (lower is better).
/ Dressed HCCA: HCCA as a normal distribution with its out-of-season error by lead.
/ TCCA: our consensus correction of the early-aid mean (no ECMWF, no SHIPS).
/ GECO: GEFS Ensemble, Consensus-Oriented: NHC's 30 GEFS member tracks around HCCA, every cycle.
/ IECO: IFS-ENS Ensemble, Consensus-Oriented: ECMWF's 50 ensemble member tracks around HCCA, every cycle.

*Reproduce.* The whole chain, resumable: `scripts/rerun_chain.sh start`. Inputs: `python -m tcic.data.atcf` (NHC decks),
`scripts/fetch_ecmwf_tracks.py` and `python -m tcic.data.ecmwf_tf` (ECMWF tracks). Consensus correction and spread model:
`python -m tcic.consensus.features`, `scripts/m15_run.py`, `python -m tcic.ensemble.members`, `scripts/ens_run.py`, `scripts/ens_calib.py`.
Products: `src/tcic/ensemble/geco.py`; GECO `scripts/ens_geco.py` and `scripts/ens_multiseason.py`; IECO `scripts/ens_ieco.py`. Comparisons
and figures: `scripts/fnv3_compare.py`, `scripts/report_figures_ens.py`, `scripts/case_plots.py`. Detailed results: `docs/ensemble_multiseason.md`,
`docs/ensemble_ieco.md`, `docs/fnv3_compare.md`; history in `plan.md`.
