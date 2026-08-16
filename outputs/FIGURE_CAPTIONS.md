# Figure Captions

**Figure 1.** Bangladesh study region and seismicity, 1973–2024. All M ≥ 4.0
earthquake epicentres from the merged USGS ComCat + ISC Bulletin catalog
(N = 5,779 events) are plotted as circles coloured by hypocentral depth and
sized by magnitude. The Bangladesh national border (brown) and the GEM Global
Active Faults Database (GAFD) traces (grey) are shown for tectonic context.
A 220-km scale bar and a north arrow are included. The 1° × 1° study window
spans 20–28 °N and 88–96 °E and covers the Indo-Burman fold belt, the
Dauki–Dhubri fault system, and the southern Shillong Plateau.

**Figure 2.** Frequency–magnitude distribution (FMD) and Gutenberg–Richter
fit for the merged catalog. Blue bars: non-cumulative event count per 0.1-M
bin. Red circles: cumulative count N(≥M) on log axis. Green dashed line:
maximum-likelihood Gutenberg–Richter fit log₁₀ N = a − b·M with Mc = 4.13
(MAXC) and b = 0.808 (Aki–Utsu MLE). Orange dotted vertical line marks the
working completeness threshold Mc. The fit is well behaved across
M = 4.13–7.0.

**Figure 3.** Hypocentral depth distribution (0–300 km). Bars are coloured
by depth regime: shallow (< 25 km, orange), intermediate (25–70 km, green),
and deep (≥ 70 km, purple). The dashed red and dotted blue vertical lines
mark the mean and median depth, respectively. Shallow crustal and
intermediate-depth events dominate the catalog; a long deep tail extends
into the subducting Indian plate beneath the Indo-Burman Ranges.

**Figure 4.** Spatial seismicity rate per 1° × 1° cell (M ≥ 4.13). The
causal expanding-window estimator λ_cell = N_cell(<t) / T(<t) is used, with
T = 51.9 yr (catalog span 1973–2024). Cell values are annotated in events
per year. The Bangladesh border is overlaid in brown. Seismicity is strongly
heterogeneous (Gini ≈ 0.87): a small number of cells along the Indo-Burman
fold belt carry most of the activity.

** Figure 5.** Temporal seismicity of the Bangladesh study region,
1973–2024. Blue bars: annual count of M ≥ 4.0 events. Red line (right axis):
cumulative event count. Orange dashed line: mean annual rate. The catalog
is approximately stationary from the early 1990s onward, consistent with
the Stepp-style completeness analysis.

**Figure 6.** Omori-type clustering diagnostic. Non-parametric rate ratio
R(Δt) = post-mainshock rate / background rate, plotted on log–log axes
against lag time Δt since mainshock. Blue: M ≥ 5 mainshocks (N = 640); red:
M ≥ 6 mainshocks (N = 24). Vertical error bars are Poisson (1σ). The dashed
horizontal line marks R = 1 (background). Peak R ≈ 22× (M ≥ 5) and ≈ 377×
(M ≥ 6) at Δt ≈ 0.013 day (~20 min), confirming real short-lived clustering
that is nonetheless not captured by the standard ETAS MLE (which yields
K ≈ 0; see Figures 7 and 16).

**Figure 7.** Model comparison: Brier score across four forecast
configurations (M4.5/7d, M4.5/30d, M5.0/7d, M5.0/30d). Seven models are
compared on the untouched 2015–2023 evaluation period (9 yearly origins ×
64 cells): v1 Spatial Poisson (PRODUCTION), Uniform Poisson, ETAS (K ≈ 0),
ML Gradient Boost, v2 Bayesian Spatial, v3 Adaptive Spatial, and v4
Region-specific ETAS. Hatched bars indicate configurations for which the
model was not retrospectively scored. Lower Brier is better; v1, v2, and v3
are statistically indistinguishable, while ETAS, ML, and v4 under-perform.

**Figure 8.** Reliability (calibration) diagram for the production model
(Spatial Poisson v1). Mean predicted probability P(≥1 M ≥ 4.5 event in 7 d)
is plotted against observed event frequency for 7 equal-width probability
bins, over 9 yearly origins × 64 cells (576 cell-origin evaluations).
Vertical error bars are 95% binomial intervals; the dashed black diagonal
marks perfect calibration. v1 is well-calibrated in the low-probability
regime where virtually all cells lie.

**Figure 9.** Spatial holdout: 4-quadrant (NW/NE/SW/SE) Brier comparison.
Each model is fit on three quadrants and evaluated on the held-out one,
cycling through all four. v1 (blue), v2 (green), v3 (teal), and v4 (red)
are compared on the M ≥ 4.5 / 7-day configuration. v4 under-performs in 3
of 4 quadrants; the three other models are within 10⁻⁴ of one another,
confirming the absence of significant inter-model differences.

**Figure 10.** Sensitivity to the completeness threshold Mc. Left:
Gutenberg–Richter b-value (Aki–Utsu MLE) at Mc ∈ {3.8, 4.0, 4.13, 4.5}.
Right: regional seismicity rate λ(M ≥ Mc). The frozen working value
Mc = 4.13 (red dashed) is highlighted. Both b and λ move monotonically with
Mc, as expected; the chosen Mc is conservative.

**Figure 11.** Current 7-day forecast map. P(≥1 M ≥ 4.5 event in 7 days)
per 1° × 1° cell, computed with the v1 Spatial Poisson causal
expanding-window estimator using all events up to 2024-12-31. The
Bangladesh border is overlaid. Highest probabilities are concentrated along
the Indo-Burman fold belt in the east.

**Figure 12.** Large-event probability uncertainty. Left: number of catalog
events at or above each magnitude threshold M ∈ {4.5, 5.0, 5.5, 6.0, 6.5,
7.0} (log scale). Only a single M ≥ 7 event appears in the 51.9-year
catalog. Right: P(≥1 event, M ≥ threshold, in 7 days) point estimate with
95% Garwood confidence intervals. At M ≥ 7 the CI spans more than an order
of magnitude, indicating that any long-term M ≥ 7 hazard estimate from this
catalog is data-limited.

**Figure 13.** Grid sensitivity: Brier score at 0.5°, 1.0°, and 2.0° grid
resolutions. Left: Brier (log scale) for v1 Spatial Poisson (blue) and v3
Adaptive Spatial (teal), M ≥ 4.5 / 7-day, 2015–2023 evaluation period.
Right: Brier-score range (max − min) across grids for each model. v3
(adaptive Epanechnikov k-NN smoothing) is ~13× more stable across grid
resolutions than v1, but this does not translate into a statistically
significant Brier improvement.

**Figure 14.** Prospective monitoring timeline. Top: blue rectangles mark
the 9 yearly evaluation windows (2015–2023); red triangles mark v1 forecast
issuance events recorded in the live forecast ledger. Bottom: cumulative
mean Brier score across evaluation origins (M ≥ 4.5 / 7-day). The
INSUFFICIENT EVIDENCE threshold (Brier > 0.05, red dashed) is set by the
project's prospective monitoring protocol. v1 remains well below the
threshold throughout.

**Figure 15.** Final publication forecast with uncertainty band. (a) Point
estimate of P(≥1 M ≥ 4.5 event in 7 days) per 1° × 1° cell from the v1
Spatial Poisson causal expanding-window estimator (window to 2024-12-31).
(b) 95% uncertainty interval width (P_upper − P_lower) per cell, computed
by propagating the Jeffreys credible interval on each cell's Poisson rate
through 1 − exp(−λΔt). Cells with the highest rates also carry the widest
absolute uncertainty.

**Figure 16.** Model hierarchy and verdicts. Mean Brier score (4
configurations, 2015–2023) for the production model and three candidates.
v1 Spatial Poisson (blue) remains the production model: validated, frozen,
and used in the live prospective ledger. v2 Bayesian Spatial (green) is a
retained candidate that produces no statistically significant improvement.
v3 Adaptive Spatial (teal) and v4 Region-specific ETAS (red) were both
formally REJECTED (verdict D): neither produced a bootstrap CI that excludes
zero in any of the 4 (v3) or 16 (v4) tested configurations.
