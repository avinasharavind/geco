// Project write-up: early-cycle TC intensity correction and calibrated intensity ensembles for physical-model guidance.
// Build:  .venv/bin/python scripts/report_figures.py && .venv/bin/python scripts/report_figures_ens.py && \
//         .venv/bin/python scripts/case_plots.py && \
//         .venv/bin/python -c "import typst; typst.compile('docs/report/report.typ', output='docs/report/report.pdf', font_paths=['/usr/share/fonts'])"

#let n = json("numbers.json")
#let e = json("numbers_ens.json")
#let f1(x) = {
  let r = str(calc.round(x, digits: 1))
  let r = if r.contains(".") { r } else { r + ".0" }
  r.replace("-", "−")
}
#let f2(x) = str(calc.round(x, digits: 2)).replace("-", "−")
#let pct(x) = str(calc.round(100 * x, digits: 0)) + "%"
#let diff(d) = [#f2(d.at(0)) kt [#f2(d.at(1)), #f2(d.at(2))]]

#let ink2 = rgb("#52514e")
#let rule = rgb("#d6d5d0")
#let accent = rgb("#2a78d6")

#set document(title: "Early-cycle intensity correction and calibrated intensity ensembles for physical-model TC guidance")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1",
  header: context { if counter(page).get().first() > 1 [#set text(8pt, fill: ink2); Early-cycle TC intensity correction · project write-up #h(1fr) 29 Oct 2026] })
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
    ..rows.slice(1).flatten().map(c => if c.starts-with("-") { "−" + c.slice(1) } else { c.replace(" -", " −") }),
  )
}
#let note(body) = block(fill: rgb("#f3f2ee"), inset: 10pt, radius: 3pt, width: 100%, body)

// ------------------------------------------------------------------ title
#align(center)[
  #text(17pt, weight: "bold")[Early-cycle intensity correction and calibrated\ intensity ensembles for physical-model TC guidance]
  #v(0.3em)
  #text(10.5pt, fill: ink2)[Project write-up · status as of 29 October 2026 · North Atlantic and East Pacific]
]
#v(0.6em)

#note[
  *Summary.* This project started from a simple question raised by AIFS-TC (Allen et al. 2026): if a cheap learned correction can lift
  an AI weather model's tropical-cyclone intensity forecasts to the operational frontier, can the same idea add value on top of the
  _physical_ models (GFS/GEFS, IFS/ECMWF ensemble, HAFS) that forecasters already use, using only public data and only information that
  is available in time for the forecast? The work has produced a three-part system. A *consensus correction* learns, with gradient-boosted
  trees, how far the average of NHC's early guidance typically is from the eventual best track, given the storm's current state and the
  disagreement between the guidance. An optional *gridded-feature* branch adds information extracted from the GEFS and IFS ensemble
  fields themselves. Finally, a *calibrated ensemble* layer turns each raw physical ensemble (31 GEFS or 51 ECMWF members), whose
  intensities are far too weak and far too similar, into a set of smooth, realistic intensity scenarios that are centred on the corrected
  consensus and spread according to how uncertain such forecasts have historically been.

  Everything is tested on the combined 2025 and 2026 seasons, held out from training. Testing on 2026, the first season run as it
  would be in real time, showed that NHC's public real-time files carry no ECMWF aids. The consensus correction is therefore trained
  without ECMWF inputs, which also made it more accurate in 2025. In 2025 it matches NHC's corrected consensus HCCA, and over 2018–2024 it
  beats HCCA and the official forecast on rapid intensification. In 2026, however, it falls about 2 kt behind HCCA (which uses ECMWF
  guidance internally) and is several knots too weak. Gridded fields from our own IFS-ENS extraction, the only real-time route to ECMWF
  information, now reduce the consensus error by 1–1.5 kt; GEFS fields add nothing, even with five seasons of training data.

  The result is two real-time ensemble products. *GECO* (GEFS Ensemble, Consensus-Oriented) reshapes NHC's public GEFS member tracks
  around HCCA at every cycle; *IECO* does the same for our IFS-ENS members at 12 UTC. Neither needs any gridded data in its centre or
  spread. Because GECO needs
  only NHC's files, it was tested on nine seasons (2018–2026, #e.ms_storms storms): it beats a probabilistic version of HCCA in every
  season, by #f2(-e.ms_all_hccad.at(0)) kt of CRPS on average, with about three quarters of outcomes inside its 10–90% range. Over
  2025–26 it beats FNV3's members by #f2(-e.at("ens_GEFS_vs_fnv3").at(0)) kt (significant) and is nearly unbiased. IECO, on the 12 UTC
  cycles, is #f2(-e.at("ens_IFS-ENS_vs_fnv3").at(0)) kt better than FNV3's members (not significant) and a little behind GECO. As for every system, rapidly intensifying storms remain the main weakness.
]

// ------------------------------------------------------------------ 1
= Motivation and design principles

AIFS-TC corrected the very weak intensities of ECMWF's AIFS-Single AI model with a gradient-boosted tree and a small convolutional
network trained on nine seasons of hindcasts, cutting the model's error by about a factor of three and bringing it level with FNV3 and the
NHC official forecast. The physical models used operationally are much less biased than AIFS at the storm scale, but their ensembles,
which are run at coarse resolution, still produce intensities that are far too weak and members that disagree far too little. The question
here is therefore not only whether a learned correction helps deterministic intensity forecasts from these models, but also whether it
can turn their ensembles into usable probabilistic intensity guidance.

Four principles shaped the design. First, everything is built for the *early cycle*: an input may only be used if it is available within
about three hours of the synoptic time, exactly as for NHC's own consensus aids and the official forecast. This makes the comparison with
OFCL fair (the AIFS-TC paper compared a late model) and means the system could run operationally. Second, only *public data* are used:
NHC's forecast and best-track files, IBTrACS, NCAR's archive of real-time SHIPS output, model archives from dynamical.org, ECMWF open
data and Google DeepMind's Weather Lab archive. Third, the work is *staged*, starting with the cheapest version that could possibly work
(track files only) and adding gridded data only to measure what it contributes. Fourth, all comparisons follow NHC's verification
conventions and use *homogeneous* samples with storm-block bootstrap confidence intervals, so that differences between systems are
measured on identical cases and judged against the storm-to-storm variability of a single season (details in Appendix A).

// ------------------------------------------------------------------ 2
= System architecture

#figure(image("figures/fig_architecture.svg", width: 100%), caption: [Overview of the system. All inputs are available within three
hours of the synoptic time $t_0$. The consensus correction (1) produces a corrected mean intensity, a predicted error and a probability
of rapid intensification; the member deviations (2) carry each physical ensemble's scenario structure; the corrected ensemble (3)
combines the two.]) <fig-arch>

@fig-arch shows how the pieces fit together. At each synoptic time $t_0$ the system gathers what NHC's forecasters would have in hand: the
operational analysis of the storm (CARQ), the interpolated ("early") versions of the dynamical and statistical-dynamical models that
NHC publishes in real time, and the member tracks of the GEFS ensemble from the most recent run that is old enough to have arrived.
ECMWF guidance enters only through our own extraction from ECMWF open data, because NHC's real-time files carry no ECMWF aids. The
*consensus correction* forms a simple reference forecast, the mean of whichever early aids are available, and uses a gradient-boosted
tree model to predict how far the best track will be from that reference at each lead time. Its output is a corrected mean intensity,
plus a 10–90% interval derived from quantile models and widened by conformal calibration so that it holds its nominal coverage.

The *ensemble layer* starts from a raw physical ensemble. Each member's departure from its own ensemble mean is treated as a scenario:
it is re-anchored so that all members start together at the observed intensity, lightly smoothed in time so that it evolves like a real
storm, and normalised so that only its shape (which members intensify faster or slower, and when) is kept. The size of the spread comes
instead from a *spread model*, trained on the consensus correction's past errors, that predicts how wrong this particular forecast is
likely to be. The *corrected ensemble* is a consensus centre plus the normalised scenarios scaled by that predicted error. For both
products, GECO (GEFS) and IECO (IFS-ENS), the centre is NHC's corrected consensus HCCA, which is in the public real-time files. A
companion *RI model* gives a calibrated probability of rapid intensification. A separate *gridded-feature* branch (section 5) can feed
additional predictors from the ensemble fields into the consensus correction.

All models are trained season-wise. The consensus correction uses 2018–2024 with leave-one-season-out cross-validation and is tested on
2025 and 2026 combined (about 60 named storms). The 2025 season is also the test year of the AIFS-TC paper, which makes those results
directly comparable. The 2026 season, verified against NHC's working best track up to 24 October, is the first season run exactly as
in real time, from NHC's public real-time files. The gridded models are cross-validated by season over what the archive allows: IFS-ENS
2024–2026 and GEFS 2020–2026.

// ------------------------------------------------------------------ 3
= Inputs and variables

== Storm state and early guidance
The storm state comes from NHC's operational analysis (CARQ) at $t_0$: maximum wind and minimum pressure, position, the intensity change
over the previous 6, 12 and 24 hours, the storm's motion over the previous 12 hours and the distance to land. The guidance consists of the
interpolated aids listed in @tab-aids. Interpolated aids are the previous cycle's late model runs shifted in time and adjusted to the
current intensity; following NHC's verification practice, the 6-hour and 12-hour versions of each aid are merged into one. For each aid
the model sees its intensity at the lead time in question and its change over the preceding 12 hours; missing aids are simply left
missing. The ECMWF aids are in NHC's archive files but absent from its real-time files during the season, so they are not used. A
version trained with them was 0.65 kt worse over 2025–26 (significant), because it learned to lean on inputs that are missing in real
time; even in 2025, when the archive has them, it was 0.4 kt worse (`docs/m15_realtime.md`).

#figure(table(columns: (auto, 1fr, auto), align: left,
  table.header[*Aid*][*Model*][*Type*],
  [AVNI, AEMI], [GFS deterministic; GEFS ensemble mean], [global, physical],
  [EGRI, CMCI, NVGI], [UK Met Office; Canadian; US Navy], [global, physical],
  [HWFI, HMNI, HFAI, HFBI, CTCI], [HWRF; HMON; HAFS-A; HAFS-B; COAMPS-TC], [regional hurricane models],
  [DSHP, LGEM], [SHIPS with decay over land; Logistic Growth Equation Model], [statistical-dynamical],
), caption: [The twelve early (interpolated) aids used by the consensus correction. ECMWF's EMXI and EMNI are not used, because they are
absent from NHC's real-time files.]) <tab-aids>

== Ensemble members
The GEFS (30 perturbed members) and ECMWF (50 perturbed members) ensembles enter in two ways. NHC's tracker output for the GEFS members is
in NHC's real-time files for every cycle; to respect the early-cycle rule we use the newest run that is at least 6 hours old, with lead
times shifted accordingly. NHC's ECMWF member tracks exist only in the archive, so they are used just for a 2025 comparison of ensembles,
not as inputs. For the gridded pilot and the
corrected ensembles we also ran our own tracker on the full ensembles (31 GEFS and 51 IFS-ENS members including the control), so that
the member intensities, structure and environment are all derived consistently from the model fields. In the consensus correction the
members are summarised by their mean, spread, 10th and 90th percentiles, the fraction of members that intensify by at least 30 kt in
24 hours, and their offset from the observed intensity at $t_0$.

== Optional predictors: SHIPS and gridded fields
The real-time SHIPS output archived by NCAR provides environmental predictors computed along the previous official forecast track: vertical
wind shear and its direction, sea-surface temperature and the corresponding potential intensity, 200 hPa temperature, mid-level relative
humidity, 850 hPa vorticity, 200 hPa divergence, ocean heat content, the model vortex strength and satellite infrared predictors. The
gridded branch reads surface fields for every ensemble member around the storm (sea-level pressure, 10 m wind and gusts, 2 m temperature and
humidity, low- and mid-level heights and temperatures for IFS-ENS, precipitable water for GEFS) and pressure-level fields from one member
for the environment (winds at 200, 500 and 850 hPa, temperatures at 200 and 500 hPa, humidity at 500–850 hPa, vorticity, divergence, skin
temperature and column water). From these we derive, for each member and lead time, the tracked centre, minimum pressure, maximum wind
and gust, radius of maximum wind, the depth of the pressure minimum and inner-core-minus-surroundings contrasts (warm core, moist core),
and for the ensemble as a whole SHIPS-style environmental quantities around the ensemble-mean centre (shear, steering flow, humidity,
vorticity, divergence, SST and an empirical maximum potential intensity). @tab-features summarises the feature groups.

#figure(csvtable("tables/features.csv", cols: (auto, auto, 1fr), aligns: (left, right, left)),
  caption: [Feature groups of the consensus correction (variant A uses all but SHIPS; variant B adds SHIPS). The gridded branch adds
  roughly 30–60 further features per model.]) <tab-features>

// ------------------------------------------------------------------ 4
= Consensus correction

== Formulation
For each storm, synoptic time $t_0$ and lead time $tau in {12, 24, dots, 168}$ h the model predicts the residual between the best-track
intensity and the reference consensus,
$ hat(v)(t_0 + tau) = overline(v)_("ref") (t_0, tau) + f(bold(x)), quad f "trained on" v_("obs") - overline(v)_("ref"), $
where $overline(v)_("ref")$ is the equal-weight mean of the available early aids and $bold(x)$ collects the features of section 3. A
single LightGBM model covers all lead times (the lead is a feature), cases of rapid intensification are given twice the weight in
training, as in AIFS-TC, and the hyperparameters are fixed rather than tuned (600 trees of 15 leaves, learning rate 0.03). Three
additional quantile models give the 10th, 50th and 90th percentiles. Because tree models fit the training storms' variability too
closely, their intervals are too narrow on new storms; a conformal adjustment, fitted per lead time on out-of-fold predictions, widens
them by about 1 kt at 12 h and about 11 kt at 168 h. The same machinery predicts minimum pressure, using as reference the operational
pressure at $t_0$ plus the GEFS ensemble's predicted pressure change, since the interpolated aids carry no pressure.

== What it achieves
#figure(image("figures/fig_m15_mae_by_lead.svg", width: 100%), caption: [Maximum-wind error by lead time for the consensus correction
(TCCA, without SHIPS), NHC's corrected consensus HCCA, the official forecast and the uncorrected aid mean. Left: out-of-fold 2018–2024;
right: the held-out 2025–26 seasons. Identical cases for all systems.]) <fig-lead>

Over the 2018–2024 cross-validation the corrected consensus is as accurate as NHC's best consensus aids and the official forecast
(@fig-lead): its average error differs from HCCA's by #diff(n.m15_cv_diff_TCCA_HCCA)\. The two test seasons behave differently. In 2025
the difference from HCCA is #diff(n.m15_test2025_diff_TCCA_HCCA), with almost no bias, and the correction is several knots better than
HCCA and OFCL at four to five days. In 2026 it is #diff(n.m15_test2026_diff_TCCA_HCCA) worse than HCCA, significantly, though level with
the simple consensus IVCN (#diff(n.m15_test2026_diff_TCCA_IVCN)), and it is about #f1(-n.m15_test2026_overall.TCCA.bias) kt too weak. HCCA
uses ECMWF guidance that NHC has internally but does not publish in real time, and its 2026 errors are unusually low, especially beyond
two days, so part of the gap is the missing ECMWF information and part is a 2026 season in which the early guidance was weaker than
usual. Over both seasons together the difference from HCCA is #diff(n.m15_test_diff_TCCA_HCCA)\. In every sample the model removes most
of the raw consensus's weakness: the uncorrected aid mean is about #f1(-n.m15_test_overall.TCRF.bias) kt too weak over 2025–26. The
correction is weakest at 12 h, where the official forecast and HCCA benefit from subjective and objective use of the very latest
observations.

#figure(image("figures/fig_m15_ri.svg", width: 100%), caption: [Error on cases of rapid intensification (an observed increase of at
least 30 kt in 24 h), with 95% storm-bootstrap intervals.]) <fig-ri>

The clearest gain is on rapid intensification (@fig-ri). Over 2018–2024 the corrected consensus beats HCCA on these cases by
#diff(n.m15_cv_ri_diff_TCCA_HCCA) and the official forecast by #diff(n.m15_cv_ri_diff_TCCA_OFCL)\. In 2025–26 it is level with both
(#diff(n.m15_test_ri_diff_TCCA_HCCA) against HCCA). Every system, including ours, remains strongly biased low during rapid
intensification, which is the main remaining weakness of intensity guidance in general. Adding the operational SHIPS predictors changes
the error by less than 0.2 kt, which is not surprising because DSHP and LGEM, themselves built on SHIPS, are already inputs. Against the
early version of FNV3 over 12–168 h the corrected consensus is about 0.9 kt worse on average (not significant) but less biased, and better
beyond about five days.

#figure(image("figures/fig_calibration.svg", width: 58%), caption: [Share of 2025–26 outcomes inside the 10–90% interval by lead time,
before and after conformal widening.]) <fig-cal>

// ------------------------------------------------------------------ 5
= Gridded features from the physical ensembles

The gridded branch asks whether the ensemble fields contain information that the track files do not. We extracted every 00 UTC run of
IFS-ENS from June 2024 to October 2026 and of GEFS from October 2020 to October 2026 with an active storm in either basin (347 IFS-ENS
and 839 GEFS runs), tracked every member with a
simple pressure-minimum tracker that agrees closely with NHC's own tracker (median position differences of
#f1(n.pilot_validation.ifs.pos_med_km) km for IFS-ENS and #f1(n.pilot_validation.gefs.pos_med_km) km for GEFS, wind correlations of 0.98–0.99),
and derived the per-member structure and ensemble-environment features described in section 3. Under the early-cycle rule the IFS-ENS
00 UTC run serves the 12 UTC forecast and the GEFS 00 UTC run the 06 UTC forecast.

#figure(image("figures/fig_melissa.svg", width: 100%), placement: auto, caption: [IFS-ENS run of 26 October 2025 for Hurricane Melissa: our member tracks
(left) and member intensities (right) against the best track. At 0.25° the ensemble cannot represent the storm's inner core; the
ensemble-mean intensity stays near 70–75 kt while Melissa reached 165 kt.]) <fig-melissa>

The raw ensembles illustrate the problem the correction has to solve (@fig-melissa): their tracks are good, but their intensities are
around 23 kt too weak on average and their members far too similar. Three experiments were run on identical cases with leave-one-season-out
cross-validation: correcting the ensemble's own mean intensity from gridded features alone (as AIFS-TC did for AIFS), the consensus
correction retrained on the same sample, and the consensus correction with the gridded features added. Gridded features alone are not
enough: they remove much of the ensemble's bias but remain 2–3 kt behind HCCA. Added to the consensus correction, the IFS-ENS features
reduce the error by about 1.0 kt over 2024–2026 and by about 1.5 kt over 2025–26, both significant, and by about 2 kt on rapid
intensification in 2025–26. The gain is larger than in earlier versions because the consensus no longer sees any ECMWF aids, so our
own IFS-ENS extraction is now its only source of ECMWF information. The GEFS features add nothing (−0.02 kt over 2020–2026, with a
confidence interval of ±0.2 kt), even after the GEFS archive was extended back to October 2020 to give five seasons of training data
instead of two. The likely reason is that NHC's tracker output for the GEFS members is already part of the consensus inputs.

// ------------------------------------------------------------------ 6
= Calibrated intensity ensembles

== Construction
The raw member intensities are weak and under-dispersive: in 2025–26 their 10–90% ranges contain only about 20–40% of outcomes and their
spread is a quarter to a third of their actual error. Correcting each member individually with a learned model turned out to collapse the
ensemble, because the model learns to send every member to nearly the same corrected value. The final construction instead separates
_where_ the ensemble is centred, _how wide_ it is, and _how its members differ_, and takes each from the source that does it best.

*Shape from the physical ensemble.* For member $i$ at lead $tau$ we form its departure from the ensemble mean, re-anchored so that every
member starts at zero at $t_0$, and smooth it in time with a Gaussian kernel of 6 h:
$ d_i (tau) = [v_i (tau) - overline(v)(tau)] - [v_i (0) - overline(v)(0)], quad tilde(d)_i = "smooth"(d_i), quad z_i (tau) = tilde(d)_i (tau) slash "sd"(tau), $
where $"sd"(tau)$ is the (lead-smoothed) standard deviation of the smoothed departures within that forecast, so that
$z_i$ carries only the ensemble's scenario structure. The 6-h kernel was chosen because it makes the members' step-to-step behaviour
match the best track's (section 6.4) while costing nothing in skill; wider kernels made the members unrealistically smooth.

*Size from a spread model.* A separate LightGBM model predicts, for each forecast and lead time, the expected absolute error
$hat(sigma)$ of the corrected consensus. Its inputs are the consensus-correction features (the disagreement between the early aids, the
spread of NHC's GEFS member tracks, the storm's intensity, recent trend and position, the lead time) together with the width of
the consensus correction's own quantile range and its predicted 24-hour intensity change. It is trained on the consensus correction's
out-of-fold errors from 2018–2024, so it learns from eight seasons of realistic forecast errors rather than from the two seasons of
gridded ensembles. On 2025–26 its prediction correlates with the actual error at #e.spread_corr_2025 (#e.spread_corr_within_lead_2025 within
a fixed lead time), and it ranks forecasts well (@tab-spreadbins).

#figure(csvtable("tables/spread_bins.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [2025–26 forecasts binned by the spread model's predicted error, against the corrected consensus's actual mean absolute error.]) <tab-spreadbins>

*Centre from a consensus.* The members are placed around a consensus mean $hat(v)_c$:
$ "member"_i (tau) = hat(v)_c (tau) + s dot k(tau) dot hat(sigma)(tau) dot z_i (tau), $
where $hat(sigma)$ is smoothed along lead time and $k(tau)$, a slowly varying factor of about 1.2–1.6 that converts a mean absolute
error into a member spread, is fitted by minimising the continuous ranked probability score (CRPS) on out-of-fold seasons.

For the GEFS product, *GECO*, the centre is NHC's corrected consensus HCCA exactly as issued, and $s = 1$ (the CRPS-fitted spread). HCCA
is in NHC's public real-time files at every cycle (0.3% of 2025–26 forecasts lacked it and fall back to our own consensus), and it draws
on the ECMWF guidance that NHC has internally but our public-data consensus cannot use. Centring on HCCA rather than on our consensus
correction improved the CRPS by #diff(e.ms_all_ccsr) over nine seasons and by #diff(e.ms_2526_ccsr) over 2025–26, mostly in 2026. HCCA
is used as issued; smoothing it along lead time, as our own consensus is, cost #f2(-e.ms_all_ccsh.at(0)) kt. Trimming the spread to 80%
of the fitted value cost about 0.1 kt and lowered the coverage, so GECO keeps the full spread. The fitted factor is $k = 1.2$ at all leads.

The IFS-ENS product, *IECO*, is built in exactly the same way from our 51 IFS-ENS members (00 UTC run, used at 12 UTC), with
$k = 1.5 - 0.3 tau slash 100$ fitted on 2024, the only out-of-fold IFS-ENS season. An earlier version centred IECO on the consensus
correction with IFS-ENS gridded features (section 5); on the same 2025–26 cases it was 0.5 kt worse than the HCCA-centred version and
no better than dressed HCCA (`docs/ensemble_ifs_hcca.md`). On identical 12 UTC cases IECO is 0.1 kt behind GECO (significant), probably
because its run is 12 h old against 6 h for NHC's GEFS tracks, and pooling the two ensembles around HCCA gained nothing.

Each step fixed a concrete problem seen along the way. Scaling raw member departures independently at every lead time produced traces
that zig-zagged by tens of knots; anchoring and smoothing removed that. A spread that depended only on lead time was too wide for calm
storms and too narrow for volatile ones; the spread model fixed that. The ECMWF ensemble's raw spread roughly doubled between 2024 and
2025, probably because of a model upgrade, which made any scaling learned on earlier seasons far too wide; because the spread now comes
from the consensus error model and the raw ensemble only contributes its shape, such changes no longer affect the spread. An attempt
to make the spread asymmetric (wider above the centre than below in rapid-intensification situations) made the ensembles worse and is
not used.

== Nine seasons of GECO
Because GECO needs only NHC's track files, it can be evaluated on every season since 2018 (#e.ms_storms storms, #e.ms_cases cases,
all four cycles, leads 12–120 h). For 2018–2024 every fitted ingredient is out of sample for the season being scored: the spread model
and $k(tau)$ are fitted without that season. The fair yardstick for a probabilistic forecast is a probabilistic version of the
deterministic consensus: "HCCA dressed" turns HCCA into a normal distribution whose width at each lead is HCCA's root-mean-square error
in the other seasons. (Scoring HCCA as a single value would flatter any ensemble, whose CRPS is always below its own mean's absolute
error.) GECO beats dressed HCCA in all nine seasons, each significantly, by #diff(e.ms_all_hccad) overall (@tab-ms). The gain is steady
across the change from the 20-member GEFS to the 30-member GEFSv12 in September 2020 and the switch from HWRF to HAFS in 2023, and the
refitted $k(tau)$ stayed at 1.2–1.3. GECO's ensemble mean is HCCA itself, so the gain comes entirely from the GEFS members' scenario shapes
and the storm-specific spread. Our own consensus as the centre, with the same members and spread, is level with dressed HCCA over
2018–2024 but falls behind it in 2026.

#figure(csvtable("tables/multiseason.csv", cols: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1.6fr, 1fr)),
  caption: [Fair CRPS (kt) of NHC's GEFS member tracks by season, identical cases. Raw GEFS: members offset to the observed intensity at
  $t_0$ (as NHC's interpolation does). Consensus centre: the same construction around our own consensus correction. HCCA / OFCL dressed:
  each as a normal distribution with its out-of-season error by lead. The difference column has 95% storm-bootstrap intervals; coverage is the
  share of outcomes inside GECO's 10–90% range (ideal 80%).]) <tab-ms>

== Probabilistic skill compared with FNV3
#figure(csvtable("tables/ensemble_summary.csv", cols: (auto, 1fr, 1fr, 1fr, 1fr)),
  caption: [2025–26, leads 12–120 h, identical cases within each block (GECO: #e.at("ens_GEFS_storms") storms, all cycles; IECO:
  #e.at("ens_IFS-ENS_storms") storms, 12 UTC cycles). FNV3 members are taken from the newest run at least 6 h old, the same timing rule as ours.
  Spread / error is the mean member spread divided by the root-mean-square error of the ensemble mean (1 for a well-dispersed ensemble).]) <tab-ens>

Over 2025–26 GECO beats FNV3's members: its CRPS is lower by #diff(e.at("ens_GEFS_vs_fnv3")), a significant difference, by
#diff(e.at("ens_GEFS_vs_fnv3_2025")) in 2025 and #diff(e.at("ens_GEFS_vs_fnv3_2026")) in 2026, with a mean bias of about −1 kt against
FNV3's −6 kt (@tab-ens). It is better than or level with FNV3 at every lead to 120 h, most clearly in the first two days, and is also
ahead of FNV3's late run, which arrives too late for the forecast cycle (−0.56 kt, not significant). IECO is also ahead of FNV3's members on the
12 UTC cycles: #diff(e.at("ens_IFS-ENS_vs_fnv3")), in 2025 (#diff(e.at("ens_IFS-ENS_vs_fnv3_2025"))) and in 2026
(#diff(e.at("ens_IFS-ENS_vs_fnv3_2026"))), though not significantly. The raw ensembles are more than twice as bad. FNV3's members are sharp but over-confident in
both seasons: their spread is about 60–70% of their error and their mean is 5–6 kt too weak. The spread of both GECO and IECO is close to their
error (@fig-spread); in both, the spread widens in situations the spread model recognises as volatile.

#figure(image("figures/fig_spread_vs_error.svg", width: 100%), caption: [Mean member spread (solid) and the error of the ensemble mean
(dashed) by lead time, 2025–26. A well-calibrated ensemble has the two curves close together. The raw ensembles' errors (25–40 kt) are off the
scale and only their spread is shown.]) <fig-spread>

== Probability of rapid intensification
Counting the share of members that intensify by 30 kt in 24 hours is a poor way to estimate the probability of rapid intensification
from these ensembles, because it depends on exactly how the members are smoothed and scaled. Instead a separate LightGBM classifier,
trained on 2018–2024 with the same early-cycle features plus the corrected consensus's own predicted 24-hour change, predicts the
probability that the best track will rise by at least 30 kt in the 24 hours ending at each lead time; an isotonic recalibration fitted on
out-of-fold predictions keeps its probabilities honest. On 2025–26 it is 19% better than climatology, a little behind FNV3's members
(23%), and ranks cases as well (@tab-ri). Both systems under-forecast: 2025–26 was rich in rapid intensification (#pct(e.ri_obs_rate) of
cases against #pct(e.ri_clim_rate) in 2018–2024), and at most probability levels the observed frequency is higher than forecast
(@tab-rirel). For the standard "next 24 hours" product the model's Brier score is 0.068 against 0.093 for climatology. Work on the RI
probability is paused, since it is a side product rather than a correction of the physical ensembles.

#figure(csvtable("tables/ri_summary.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [Probability of a ≥ 30 kt rise in the 24 h ending at $t_0 + tau$, $tau$ = 24–120 h, 2025–26 (#e.ri_cases cases, #e.ri_events with
  rapid intensification). Skill is the reduction in Brier score relative to climatology; AUC measures ranking only.]) <tab-ri>

#figure(csvtable("tables/ri_reliability.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [Reliability of the RI model's probabilities on 2025–26: forecasts in each probability band against the observed frequency.]) <tab-rirel>

== Physical consistency
#figure(csvtable("tables/roughness.csv", cols: (auto, 1fr, 1fr, 1fr)),
  caption: [Step-to-step behaviour of intensity traces, 2025–26. A zig-zag is two consecutive 12-h changes of opposite sign, each at least 10 kt.]) <tab-rough>

Statistics alone do not show whether individual members look like plausible storms, so we also measured how traces evolve from one
time step to the next (@tab-rough). The final corrected members change a little less from step to step than the best track does,
reverse direction less often, and rarely change by more than 30 kt in 12 hours. The first corrected version, by contrast,
zig-zagged two to three times as often as real storms. FNV3's members have a 12-hour zig-zag rate similar to the best track's, but at
their native 6-hour spacing they reverse direction about #str(calc.round(e.zigzag_6h.at("FNV3 members") / e.zigzag_6h.at("Best track")))
times as often as the best track (#e.zigzag_6h.at("FNV3 members") against #e.zigzag_6h.at("Best track")), which is the short-term jitter
visible in their traces.

== Case studies
#figure(image("figures/fig_case_6.svg", width: 100%), caption: [Hurricane Erin, 15 August 2025, about 18 h before its extreme rapid
intensification (65 kt to 140 kt by 16 August 18Z): GECO (NHC's 30 GEFS member tracks) for the 06Z cycle (09Z advisory, from the 00Z
run), IECO (our 51 IFS-ENS members) for the 12Z cycle (15Z advisory, from the 00Z run), FNV3's 00Z members, the best track and the
official forecast of the 12Z cycle. In grey, the raw members, starting at the models' own intensity.]) <fig-case-erin>

#figure(image("figures/fig_case_2.svg", width: 100%), caption: [Hurricane Melissa, 25 October 2025, at the start of its rapid
intensification to 165 kt.]) <fig-case-melissa>

The case studies show what the numbers mean in practice. For Erin on 15 August (@fig-case-erin) the spread model recognises a volatile
situation: both corrected ensembles fan out widely in the first day, and their upper members reach the 140-kt peak, whereas FNV3's members
stay mostly below about 125 kt and the official forecast peaks near 120 kt. For Melissa (@fig-case-melissa) the corrected ensembles again rise quickly from the observed
intensity and fan out; their means peak well below the 165-kt best track, as the consensus does, but the upper members reach 140–160 kt,
so the extreme outcome lies within the ensemble. FNV3's mean comes closer to the peak and its upper members also reach about 160 kt. Eight
further cases are in Appendix C, including a steady storm where the spread stays narrow, two forecasts made days before a rapid
intensification and an East Pacific storm that decays over cooler water. A larger set of 28 cases, with several start times for
Erin, Gabrielle, Humberto, Melissa and Kiko, is collected in a separate file (`docs/case_plots.pdf`). In these figures each system
starts at a different time, following the early-cycle rule. A second file (`docs/case_plots_12z.pdf`) shows the same cases with all three
for the same 12Z cycle (GECO from the 06Z GEFS run, as it would run in real time, IECO from the 00Z IFS-ENS run and FNV3 from its
06Z run) for a like-for-like visual comparison.

== Remaining limitations
The ensembles are symmetric around a consensus that, like every guidance system, under-predicts rapid intensification, so in such
situations the truth still falls above the 90th percentile more often than it should, and some lower members describe weakening that did
not happen. Making the spread asymmetric did not help, because which borderline situations will intensify rapidly cannot be identified
well enough in advance. Both products inherit HCCA's errors, which they cannot improve on in the mean. The RI probability is reliable in ranking but too low in
RI-rich seasons. Three follow-up attempts to fix the RI shortfall,
documented in `docs/ri_improvements.md` and `docs/ensemble_ritail.md`, had limited success. Adding SHIPS's environmental predictors to the
RI model did not help. Adapting its base rate to the RI frequency seen earlier in the same season did not help either, because a season's
early part says little about the rest. Stretching the ensemble's upper half in proportion to the RI probability helped only marginally.
Feeding NHC's own operational RI guidance (SHIPS-RII, DTOPS and related probabilities, already in the SHIPS files) into the RI model gave
a small but significant improvement. Because the RI probability is a side product rather than a correction of the physical ensembles,
further work on it is paused.

// ------------------------------------------------------------------ 7
= Conclusions and next steps

For physical-model guidance, a learned early-cycle correction built only from NHC's public real-time files reaches the operational
frontier over 2018–2025 and improves on it during rapid intensification. The 2026 season, the first run exactly as in real time,
shows its limit: without the ECMWF guidance that NHC uses internally it falls about 2 kt behind HCCA. The gridded IFS-ENS fields are
the real-time route to that information, and they now reduce the consensus error by 1–1.5 kt; the GEFS fields add nothing, even with
five seasons of training data. The calibrated ensemble combines the accuracy of the corrected consensus, a storm-specific estimate of
its uncertainty and the scenario structure of the physical ensembles. GECO, NHC's GEFS member tracks reshaped around HCCA, beats a
probabilistic HCCA in each of nine seasons and FNV3's members over 2025–26, at every cycle and from public real-time files only. IECO, the
same construction for the IFS-ENS, is a little behind GECO but also ahead of FNV3's members. Neither uses gridded data in its centre or
spread: further gains from the model fields would probably need a convolutional network on the full gridded fields, as in AIFS-TC.

A multi-model ensemble that pools the scenario shapes of NHC's GEFS and ECMWF member tracks around the same corrected centre was
slightly better than either model alone in 2025 (CRPS 7.98 against 7.99–8.06 kt, not significant) and missed the upper tail less often
in rapid-intensification runs. Doing the same with our own GEFS and IFS-ENS members at a common start time (GEFS used 12 h old) gave
no measurable gain over 2025–26 (−0.03 kt): once both ensembles share the corrected centre and the predicted spread, their scenario
shapes are close to interchangeable.

The aim of the project is the best possible correction of the physical ensembles, not the best forecast overall, so FNV3 remains a
benchmark and is not used as an input. The goal is a real-time system; using an older IFS-ENS run in the early window is acceptable.
The GEFS archive has been extended back to October 2020 (492 runs), which settled that GEFS fields add nothing, so the GEFSv12
reforecasts are no longer a priority. The next steps, in order, are:
- Extend IECO to the 00 UTC cycle from the 12 UTC IFS-ENS run.
- A convolutional network on multi-level gridded fields, combined with the consensus correction, when compute allows.
- Re-examine the 2026 shortfall once NHC finalises the 2026 best tracks, and look at which storms and lead times drive it.
- A real-time prototype triggered by NHC's public guidance files, after checking when each data source becomes available relative to the
  early-cycle deadline.

// ------------------------------------------------------------------ appendices
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")
#show heading.where(level: 1): it => { set text(13pt); [Appendix #counter(heading).display("A") · #it.body]; v(0.2em) }

= Verification framework
All verification follows NHC's conventions. Truth is the final best track (b-deck); a case counts only if the system is a tropical or
subtropical cyclone at both the initial and the valid time; and every comparison is homogeneous, i.e. all systems are scored on exactly
the same cases. Deterministic forecasts are scored by the mean absolute error and bias of maximum wind; ensembles by the fair continuous
ranked probability score (CRPS), a quantile score at the 10th, 50th and 90th percentiles, the coverage of the 10–90% range and the ratio
of spread to the error of the mean. Confidence intervals resample whole storms (block bootstrap), as in the AIFS-TC paper, because cases
from the same storm are strongly correlated. Rapid intensification is an observed increase of at least 30 kt over the 24 hours ending at
the valid time.

As a check, the harness recomputes NHC's published 2024 Atlantic verification. The official-forecast table is reproduced with identical
case counts and errors within 0.1 kt, and the homogeneous early-model table with identical case counts and #n.m1_t5a_n_within_0p2 of
#n.m1_t5a_n errors within 0.2 kt (@fig-m1). The AIFS-TC paper's 2025 values for OFCL and FNV3 are reproduced within their confidence intervals.

#figure(image("figures/fig_m1_reproduction.svg", width: 40%), caption: [Recomputed versus published errors for NHC's 2024 homogeneous
early-aid verification; each point is one aid at one lead time.]) <fig-m1>

= Detailed results
All values in kt; identical cases across the systems in each table; 95% storm-bootstrap intervals.

#grid(columns: (1fr, 1fr), gutter: 12pt,
  [*Consensus correction, cross-validation 2018–24* #csvtable("tables/overall_cv.csv")],
  [*Consensus correction, held-out 2025–26* #csvtable("tables/overall_test.csv")],
  [*Rapid intensification, 2018–24* (#n.m15_cv_ri_storms storms) #csvtable("tables/ri_cv.csv")],
  [*Rapid intensification, 2025–26* (#n.m15_test_ri_storms storms) #csvtable("tables/ri_test.csv")],
)
*Error by lead, held-out 2025–26* #csvtable("tables/bylead_test.csv")
*Error by lead against FNV3 early, 2025–26, 12–168 h* (#n.fnv3_storms storms) #csvtable("tables/bylead_fnv3.csv")

*Gridded features (leave-one-season-out, leads 12–120 h; from `docs/m2_results.md`).* On IFS-ENS cycles (2024–2026): consensus
correction retrained on the same rows 12.0 kt, with IFS-ENS gridded features 10.9 kt (difference −1.04 kt [−1.72, −0.40]), gridded
features alone 13.4 kt, HCCA 10.4 kt, OFCL 10.4 kt. On GEFS cycles (2020–2026): 10.41 kt without and 10.39 kt with GEFS gridded features
(−0.02 kt [−0.22, +0.18]), HCCA 9.6 kt.
*Pressure (2025–26, 12–120 h):* corrected 8.8 mb against 12.3 mb for the reference and 17.1 mb for persistence.

= Additional case studies
#figure(image("figures/fig_case_1.svg", width: 100%), caption: [Hurricane Erin, 14 August 2025, two days before its rapid intensification.])
#figure(image("figures/fig_case_3.svg", width: 100%), caption: [Hurricane Kiko, 1 September 2025, East Pacific: our ensembles bracket the
best track while FNV3's remain far too weak after the second day.])
#figure(image("figures/fig_case_4.svg", width: 100%), caption: [Hurricane Erin, 19 August 2025: recurvature and extratropical transition.])
#figure(image("figures/fig_case_5.svg", width: 100%), caption: [Tropical Storm Henriette, 6 August 2025: a weak, long-lived storm.])
#figure(image("figures/fig_case_7.svg", width: 100%), caption: [Hurricane Humberto, 25 September 2025: rapid intensification to 140 kt
within three days. All three ensembles are too slow; only our upper members reach the peak, a day late.])
#figure(image("figures/fig_case_8.svg", width: 100%), caption: [Hurricane Gabrielle, 18 September 2025: weak for four days, then rapid
intensification to 120 kt. Every system misses the timing and size of the burst; FNV3's mean comes closest at the peak.])
#figure(image("figures/fig_case_9.svg", width: 100%), caption: [Hurricane Melissa, 22 October 2025, three to four days before its rapid
intensification. Our ensembles already lean toward a strong hurricane and keep the best track inside their range; FNV3 stays weak until day four.])
#figure(image("figures/fig_case_10.svg", width: 100%), caption: [Hurricane Priscilla, 5 October 2025, East Pacific: rise to 100 kt and decay
over cooler water. All systems capture the shape; ours peak lower and decay later than the best track.])

#pagebreak(weak: true)
= Data coverage
#figure(image("figures/fig_aid_coverage.svg", width: 66%), caption: [Share of NHC forecast cycles with each early aid, after merging the 6-h and
12-h interpolated versions. HAFS (HFAI/HFBI) became operational in 2023; ECMWF aids appear in the public archive from 2020 but never in
the real-time files; FNV3 (GDMI) only from 2025.])

The operational SHIPS archive covers 70–96% of cases per season (lowest in 2020–21) and reaches 168 h only from 2020. GEFS members are
available for nearly all cases; ECMWF members for about three quarters of archive cases from 2020 and very few before, and none in
real time. Our own extraction covers every 00 UTC run with an active storm: IFS-ENS from June 2024 and GEFS from October 2020, both to
October 2026 (1,186 runs). FNV3 members for 2025–26 come from Google
DeepMind's Weather Lab archive (50 members, CC BY 4.0 for historical data); their ensemble mean matches the GDMN aid in NHC's files.

= Technical notes and limitations

*Data access.* Subsetting a cloud dataset does not always reduce what is downloaded: on dynamical.org's "virtual" stores each stored
chunk is a whole global field, so a storm-centred patch costs as much as the globe, whereas the "native" stores are tiled and make patches
cheap but hold only surface variables. The gridded extraction therefore reads per-member surface fields from the native stores and
pressure-level environment fields from one member only (the GEFS virtual store and ECMWF open data). Reading the zarr arrays directly,
rather than through xarray's lazy indexing, cut memory use by a factor of about four.

*Running on a shared machine.* The extraction ran on a shared DGX Spark, capped at four cores and 16 GB, at the lowest CPU and I/O
priority, with downloads limited to 85% of the network link minus other users' traffic and automatic pauses when the machine was busy.
Work was split into one unit per model run, written atomically, with a lock-file heartbeat so that a crash or reboot loses at most one unit
and the next start resumes; failed units are retried. The original pilot (about 0.75 TB downloaded, 6 GB stored) took about 20 hours
of network-bound work; the GEFS extension back to October 2020 (492 runs, about 210 GB) took 7.5 hours overnight. Our tracker's agreement with NHC's tracker over all units is shown in @fig-trackval.

#figure(image("figures/fig_tracker_validation.svg", width: 92%), caption: [Our tracker's maximum wind against NHC's tracker on the same
members and runs.]) <fig-trackval>

*Real-time availability.* ECMWF's track aids (EMX, EEMN and the members) are absent from NHC's public real-time files during the season
and appear in the archive only afterwards, so the consensus correction is trained without them; our own extraction from ECMWF open data
is the legitimate real-time route. From mid-May 2026 ECMWF's open data no longer include the control member, so 2026 environment fields come
from member 1.

*Model changes.* The ECMWF ensemble's raw intensity spread roughly doubled between 2024 and 2025, and HAFS replaced HWRF in 2023. The
ensemble spread is set by the consensus error model rather than by the raw ensemble, so the first no longer affects it; aid
availability changes are handled by leaving missing aids missing.

*Statistical caveats.* The two test seasons together contain about 60 storms, so confidence intervals are wide and only large
differences are significant. The 2026 best tracks are NHC's working best tracks; post-season analysis often revises peak intensities.
Hyperparameters were not tuned. The IFS-ENS gridded models have only three seasons of data. The
day of the year is among the most important features in some models, which may partly reflect differences between seasons rather than
physics.

= Glossary and reproduction
/ CARQ: NHC's operational analysis of position and intensity at $t_0$, with the preceding 24 h.
/ OFCL: the official NHC forecast. HCCA: NHC's corrected consensus. IVCN: NHC's simple intensity consensus.
/ Early / late: available / not available in time for the forecast cycle that starts at the synoptic time.
/ FNV3 (GDMN, GDMI): Google DeepMind's cyclone model; NHC files carry its ensemble mean (late) and an interpolated version (early).
/ CRPS: continuous ranked probability score, the probabilistic analogue of the mean absolute error (lower is better).
/ TCCA, TCCB: our consensus correction without and with SHIPS.
/ GECO: GEFS Ensemble, Consensus-Oriented: NHC's GEFS member tracks around HCCA, full spread, every cycle.
/ IECO: IFS-ENS Ensemble, Consensus-Oriented: our IFS-ENS members around HCCA, full spread, 12 UTC.

*Reproduce.* M1 checks: `scripts/m1_check.py`; consensus correction: `python -m tcic.consensus.features` and `scripts/m15_run.py`;
gridded extraction: `scripts/pilot.sh`; gridded models: `python -m tcic.gridded.features` and `scripts/m2_run.py`; ensembles:
`python -m tcic.ensemble.members`, `scripts/ens_run.py`, `scripts/ens_calib.py` (spread and RI models), `scripts/ens_smoothspread.py`, `scripts/ens_gridcentre.py` (gridded-centre variant), `scripts/ens_geco.py` (GECO and IECO), `scripts/ens_ifs_hcca.py`, `scripts/ens_multiseason.py` (nine seasons), `scripts/fnv3_compare.py`; the whole chain: `scripts/rerun_chain.sh`;
figures: `scripts/report_figures.py`, `scripts/report_figures_ens.py`, `scripts/case_plots.py`. Detailed notes are in `plan.md` and `docs/`.
