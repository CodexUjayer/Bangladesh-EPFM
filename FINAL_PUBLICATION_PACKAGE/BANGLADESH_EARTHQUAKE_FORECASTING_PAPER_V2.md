---
title: "Probabilistic Earthquake Forecasting for Bangladesh: A Spatial Poisson Baseline and the Failure of ETAS, Bayesian Hierarchical, and Adaptive Smoothing Candidates"
author: "Bangladesh Earthquake Forecasting Project"
date: "FINAL_v1.0_FROZEN publication package — version 2 of the project paper"
version: "V2"
---

# Abstract

This paper presents the complete scientific record of the Bangladesh Earthquake Forecasting Project, an effort to develop operationally deployable probabilistic forecasts for a region that straddles the Indo-Burman subduction zone, the Dauki fault system, and the diffuse India–Eurasia plate boundary. Building on a canonical merge of the USGS ComCat and ISC Bulletin catalogs (5,779 events, $M \geq M_c = 4.13$, 1973-02-10 to 2024-12-30, mean depth 52.6 km), we estimate a Gutenberg–Richter $b$-value of $0.808 \pm 0.010$ and a regional $M \geq 4.5$ rate of 37.5 events per year. A cell-wise Spatial Poisson model (FINAL_v1.0_FROZEN), in which each 1°×1° cell carries an independently estimated Poisson rate with Garwood 95% confidence intervals, is retrospectively validated on nine chronological origins spanning 2015–2023. At the operational 7-day $M \geq 4.5$ horizon the model achieves Brier score 0.0242 and expected calibration error 0.0087; at 30 days, 0.0763 and 0.0322 respectively, with spatial holdout wins in all four quadrants.

Three candidate extensions are then evaluated under a strictly leak-free protocol: (v2) a Bayesian hierarchical Gamma–Poisson model with empirical-Bayes shrinkage; (v3) an adaptive spatial smoother with Gaussian and Epanechnikov kernels and fixed or nearest-neighbour bandwidths; (v4) a region-specific ETAS model with depth-stratified fitting, depth-dependent spatial kernels, and an exponential temporal alternative. None yields a statistically defensible improvement over the v1 baseline: v2 changes Brier by $\Delta \approx 0$ (verdict B, uncertainty improvement only); v3 fails posterior predictive checks (simulated Gini 0.684 vs observed 0.649, simulated top-3 fraction 0.413 vs observed 0.310) and is REJECTED; v4 collapses to $K \approx 0$ in all four variants with branching ratio 0.0, slightly worsens 30-day and 90-day Brier, and is REJECTED.

The persistent Omori-style rate enhancement of $R \approx 24\times$ at $\Delta t \approx 18$ min (full catalog) or $R \approx 15.5\times$ at $\Delta t \approx 3$ min (development period) coexists with $K \approx 0$ and $\alpha = 0$ in every ETAS variant, a contradiction that we attribute to short-lag event relocations and duplicate agency reports rather than genuine tectonic aftershock cascades. We conclude that the Spatial Poisson baseline remains the only prospectively validated model for the available catalog, and we describe the data acquisitions (GCMT, BMD, historical) and longer monitoring horizon required before any more sophisticated candidate could be promoted.

**Keywords**: Bangladesh seismicity; Indo-Burman subduction; earthquake forecasting; Spatial Poisson model; ETAS; Bayesian hierarchical model; adaptive smoothing; Gutenberg–Richter; magnitude of completeness; Brier score; expected calibration error; prospective validation; posterior predictive check; Omori law; branching ratio.

---

## 1. Introduction

Bangladesh and the surrounding plate-boundary region occupy one of the most consequential seismic gaps on Earth. With more than 170 million people in Bangladesh alone and critical infrastructure concentrated in the Ganges–Brahmaputra delta, even a single $M \geq 7$ event on the plate interface or on the Dauki fault system could trigger casualties, liquefaction, and economic disruption on a national scale. Yet, despite the tectonic severity of the setting, the region has lacked an operationally deployable probabilistic forecasting system whose predictive skill has been demonstrated under strict chronological, spatial, and spatiotemporal validation. The Bangladesh Earthquake Forecasting Project was established to close that gap.

The forecasting problem in Bangladesh is unusually difficult for three reasons. First, the instrumental catalog is short and sparse: only 5,779 events survive a canonical merge of the USGS ComCat and ISC Bulletin catalogs at magnitude of completeness $M_c = 4.13$, and only one $M \geq 7$ event is recorded in the entire 51.89-year exposure. Second, the dominant tectonic style — oblique subduction of the Indian plate beneath the Burma microplate along the Indo-Burman arc, with deep intermediate-focus seismicity concentrated between 25 and 200 km depth — is poorly represented by the crustal aftershock paradigms that motivate most existing epidemic-type models. Third, regional data sources such as the Bangladesh Meteorological Department (BMD) network, the Global Centroid Moment Tensor (GCMT) project, and the ISC-GEM historical catalog are unavailable to this project, leaving an unquantifiable residual uncertainty in the long-term rate estimates.

This paper consolidates the complete project record. We document the construction and prospective validation of the production baseline (FINAL_v1.0_FROZEN Spatial Poisson), and we report the development and rejection of three candidate extensions: a Bayesian hierarchical Gamma–Poisson model (v2), an adaptive spatial smoother (v3), and a region-specific ETAS formulation (v4). Each candidate is evaluated under a strictly leak-free dev/select/eval protocol with paired block bootstrap confidence intervals, permutation testing, and Benjamini–Hochberg false discovery rate correction. The published numbers, tables, and figures in this paper are drawn exclusively from the frozen CSVs and ledgers in the project repository; no figure or table has been regenerated post-hoc to favour any candidate.

## 2. Bangladesh Tectonic Setting

The Bangladesh region lies at the eastern syntaxis of the Himalayan collision belt, where the Indian plate converges obliquely with the Eurasian plate at ~13–17 mm yr⁻¹ of NE-directed motion relative to a stable Eurasia frame. This convergence is partitioned across three first-order structures that together define the seismic hazard of the region.

### 2.1 Indo-Burman subduction zone

The Indo-Burman Fold-and-Thrust Belt marks the eastern edge of the Bengal Basin and accommodates roughly half of the oblique India–Eurasia convergence through west-dipping subduction of the Indian plate beneath the Burma microplate. The plate interface is unusually narrow and steep, dipping eastward at ~60° in the south and shallowing northward, and it is seismically active from shallow crustal depths down to ~200 km. Intermediate-focus earthquakes (70–200 km depth) dominate the instrumental catalog and contribute disproportionately to the total moment release.

### 2.2 Dauki fault system

The Dauki fault is a ~300-km-long, E–W trending, north-dipping reverse fault that separates the Shillong Plateau from the Bengal Basin. Although the long-term geologic slip rate is debated, the structure is widely considered capable of generating $M \geq 7.5$ earthquakes that would directly threaten the densely populated Bengal delta. No large event has been instrumentally recorded on the Dauki fault during the project catalog window, and the fault is therefore a major source of epistemic uncertainty in any long-term forecast.

### 2.3 Diffuse plate boundary and Ganges–Brahmaputra delta

To the north and northwest of Bangladesh, the convergence is absorbed across a broad zone of diffuse deformation, including the Madhupur and Tripura Hills structures. The Bangladesh part of the Bengal Basin itself is underlain by 10–20 km of Cenozoic and Neogene sediment, and large parts of the delta are at high risk of liquefaction and amplified ground motion during even moderate regional events.

### 2.4 Implications for forecasting

The dominance of intermediate-focus seismicity along the Indo-Burman arc has two practical consequences for probabilistic forecasting. First, the spatial pattern of seismicity is highly heterogeneous, with a thin, arcuate band of activity that is poorly approximated by smooth Gaussian kernels but well captured by a 1°×1° cell-wise Poisson discretisation. Second, the aftershock paradigm that motivates classical ETAS — in which a mainshock triggers a cascade of crustal aftershocks over hours to months — may not transfer cleanly to a subduction setting in which most events are intermediate-focus and may reflect stress transfer mechanisms with very different temporal and spatial scales.

**Figure 1**: Study region and merged earthquake catalog (fig01_study_region.png).

## 3. Literature Review

The methodological foundations of this paper draw on six decades of statistical seismology research.

### 3.1 Gutenberg–Richter scaling

The empirical magnitude–frequency relation $\log_{10} N = a - bM$ (Gutenberg & Richter, 1944) remains the most robust first-order description of earthquake occurrence. The $b$-value, typically near unity for tectonic seismicity, governs the relative frequency of small and large events and underwrites any magnitude-thresholded rate forecast. Estimation of $b$ from a finite catalog requires both a correct magnitude of completeness $M_c$ and a variance-stabilising estimator (typically the maximum-likelihood form of Aki, 1965).

### 3.2 Spatial Poisson models

The simplest stochastic description of earthquake occurrence is the stationary Poisson process, in which events in disjoint space–time regions are independent and the rate $\lambda$ is constant within each cell. When $\lambda$ is allowed to vary spatially — for example, on a 1°×1° grid — the model becomes a piecewise-constant inhomogeneous Poisson process. Confidence intervals for the per-cell rate can be derived from the Garwood exact Poisson interval (Garwood, 1936), which is preferred over the normal approximation when cell counts are small. This is the formal basis of the FINAL_v1.0_FROZEN production model.

### 3.3 ETAS and the Omori law

The Epidemic Type Aftershock Sequence (ETAS) model (Ogata, 1988) describes earthquake occurrence as a branching process in which each event triggers offspring according to the Omori–Utsu temporal kernel $(t+c)^{-p}$ and a spatial kernel that decays with distance from the parent. The model has four governing parameters: the background rate $\mu$, the productivity $K$, the magnitude-scaling exponent $\alpha$, and the spatial decay $\sigma$. ETAS has been applied with success to crustal seismicity in Japan, California, and Italy (Ogata, 1988; Zhuang et al., 2011; Marsan & Lengliné, 2008), but its applicability to subduction settings and to catalogs dominated by intermediate-focus events has been questioned (Hainzl & Ogata, 2008). The $K \approx 0$ and $\alpha \approx 0$ outcomes reported in Section 10 are consistent with that scepticism.

### 3.4 Bayesian hierarchical models

Bayesian hierarchical models pool information across spatial cells by placing a shared prior on the cell-wise rates, with the goal of shrinking noisy small-count estimates towards a global mean while preserving genuine spatial heterogeneity. The Gamma–Poisson conjugate construction (Clayton & Kaldor, 1987) is the canonical example: cell rates $\lambda_i \sim \text{Gamma}(a,b)$, event counts $N_i \sim \text{Poisson}(\lambda_i T)$. Empirical-Bayes estimation of the hyperparameters $(a,b)$ yields a closed-form posterior mean that is a precision-weighted average of the cell's own count and the global mean.

### 3.5 Adaptive spatial smoothing

Adaptive smoothing replaces the rigid 1°×1° grid with a kernel density estimate in which the bandwidth is allowed to vary spatially — typically chosen so that a fixed number $k$ of nearest neighbours fall within the kernel support (Marsan & Lengliné, 2008; Silverman, 1986). The intent is to use wide smoothing in sparse cells and narrow smoothing in active cells, producing a more efficient rate estimate than fixed-cell Poisson.

### 3.6 Magnitude of completeness

The magnitude of completeness $M_c$ is the lowest magnitude above which the catalog records all events. Multiple estimators have been proposed: the Maximum Curvature (MAXC) method (Wiemer & Wyss, 2000), the Goodness-of-Fit Test (GFT) (Wiemer & Wyss, 2000), the Entire-Magnitude-Range (EMR) method (Woessner & Wiemer, 2005), and the Stepp (1972) stability analysis.

### 3.7 Probabilistic forecast validation

Probabilistic earthquake forecasts are typically evaluated using proper scoring rules: the Brier score for binary exceedance probabilities, the log-likelihood for continuous rate forecasts, and reliability-based metrics such as the Expected Calibration Error (ECE). Statistical significance is assessed via paired block bootstrap resampling over forecast origins (Politis & Romano, 1994), permutation testing for paired comparisons, and Benjamini–Hochberg false discovery rate correction (Benjamini & Hochberg, 1995) when multiple hypotheses are tested simultaneously.

## 4. Data

The project catalog is constructed from two primary sources, supplemented by reference to fault geometry and tectonic context.

### 4.1 USGS Comprehensive Earthquake Catalog (ComCat)

The USGS ComCat, accessed via the FDSN web service, provides authoritative moment-magnitude estimates for the project region from 1973 onward. After spatial filtering to the project bounding box (20–28°N, 88–94°E plus a 1° buffer), ComCat contributes 2,293 events with $M \geq 4.5$.

### 4.2 International Seismological Centre (ISC) Bulletin

The ISC Bulletin, accessed via the ISC web service, provides reviewer-checked hypocentres and magnitudes contributed by more than 100 national and regional networks. For the project region, ISC contributes 5,576 events at $M \geq 4.25$ over 1973–2024, with substantially better completeness for intermediate-focus Indo-Burman events than ComCat alone.

### 4.3 Canonical merge

The two sources are merged in three stages: (i) magnitude conversion to a common $M_w$ scale using the empirical relations of Scordilis (2006); (ii) spatio-temporal deduplication using the 120-second/50-kilometre window of Gardner & Knopoff (1974) as extended by Reasenberg (1985); (iii) preferential retention of the USGS hypocentre where both sources report the same event. The canonical merged catalog contains 5,779 unique events spanning 1973-02-10 to 2024-12-30, with mean hypocentral depth 52.6 km.

### 4.4 Sources sought but unavailable

Four sources that would materially improve the catalog could not be acquired within the project window and are flagged as DATA-LIMITED in the frozen provenance manifest: **GCMT** (would provide authoritative $M_w$ and focal mechanisms), **BMD** (would provide local-network detections of small events), **ISC-GEM** (would extend the catalog backwards to 1900–1972), and **historical and palaeoseismic records** (would provide $M \geq 7$ recurrence intervals of 200–500 years).

### 4.5 Supporting data

Two additional datasets are used for context and visualisation only, never for rate estimation: the GEM Global Active Faults Database (GAFD, Styron & Hetland, 2020) provides the fault traces overlaid on Figures 1 and 18; a simplified Bangladesh boundary GeoJSON provides the country outline overlaid on all map figures.

**Figure 3**: Depth distribution of the merged catalog (fig03_depth.png).
**Figure 5**: Temporal distribution of the merged catalog (fig05_temporal.png).

## 5. Catalog Preprocessing

The preprocessing pipeline is implemented in three audited stages, each producing an immutable intermediate artifact that is preserved in the project repository for reproducibility.

### 5.1 Magnitude conversion

The USGS and ISC catalogs report a heterogeneous mix of magnitude scales: $M_w$, $m_b$, $M_S$, $M_L$, and $M_{WR}$. To produce a homogeneous $M_w$ catalog we apply the empirical regression relations of Scordilis (2006). The conversions used are: $M_w = 0.85\,m_b + 1.03$ for $3.5 \leq m_b \leq 6.2$; $M_w = 0.67\,M_S + 2.07$ for $3.0 \leq M_S \leq 5.3$; $M_w = M_S$ for $5.3 < M_S \leq 6.1$; $M_w = 0.99\,M_S + 0.08$ for $M_S > 6.1$. Where a USGS-preferred $M_w$ is already provided, it is retained without conversion. The propagated uncertainty on each converted magnitude is $\sigma_{M_w} \approx 0.15$.

### 5.2 Deduplication

The two catalogs overlap substantially: most $M \geq 5$ events are reported by both USGS and ISC, often with slightly different hypocentres and magnitudes. We deduplicate using the 120-second temporal and 50-kilometre spatial window introduced by Gardner & Knopoff (1974) and refined by Reasenberg (1985). For each pair of events within the window, the event with the higher-quality magnitude is retained and the other is discarded. The deduplication reduces the union count of 7,869 events (2,293 + 5,576) to 5,779 unique events, an overlap of 2,090 events removed.

### 5.3 Canonical merge audit

The full audit, recorded in the stage3 audit report, confirms: (i) no event outside the spatial bounding box survives the merge; (ii) no event earlier than 1973-02-10 or later than 2024-12-30 survives; (iii) no duplicate pair survives within the 120-s/50-km window; (iv) all surviving events carry a non-null magnitude on the homogeneous $M_w$ scale.

### 5.4 Quality summary

| Metric | Value | Status |
|---|---|---|
| Events ($N$) | 5,779 | VALIDATED |
| Exposure (years) | 51.89 | VALIDATED |
| $M_c$ (maximum likelihood) | 4.13 | VALIDATED |
| $b$-value | $0.808 \pm 0.010$ | VALIDATED |
| GCMT available | False | DATA-LIMITED |
| BMD available | False | DATA-LIMITED |
| Receiver fault geometry | False | DATA-LIMITED |
| Historical catalog | False | DATA-LIMITED |
| $M \geq 7$ events | 1 | INSUFFICIENT POWER |
| $M \geq 6.5$ events | 8 | INSUFFICIENT POWER |

## 6. Magnitude of Completeness

The magnitude of completeness $M_c$ is the lowest magnitude above which the catalog records all events in the project region.

### 6.1 Methods

**MAXC** (Wiemer & Wyss, 2000): $M_c$ is the magnitude at which the non-cumulative frequency–magnitude distribution has its maximum curvature. Estimate: $M_c = 4.05$.

**GFT** (Wiemer & Wyss, 2000): $M_c$ is the lowest magnitude at which a power-law GR fit explains at least 95% of the variance. Estimate: $M_c = 4.15$.

**EMR** (Woessner & Wiemer, 2005): $M_c$ is estimated jointly with $b$ by maximum likelihood over the entire magnitude range. Estimate: $M_c = 4.13$.

**Stepp (1972) stability**: $M_c$ is the lowest magnitude at which the standard deviation of the event count scales as $1/\sqrt{T}$. Estimate: $M_c = 4.20$.

### 6.2 Consensus and sensitivity

The four estimates range from 4.05 (MAXC) to 4.20 (Stepp), with a median of 4.14. We adopt $M_c = 4.13$ (the EMR estimate).

**Figure 10**: $M_c$ sensitivity (fig10_sensitivity.png).

| $M_c$ | $b$-value | Rate (yr⁻¹) | $P(M\geq4.5 | 7d)$ |
|---|---|---|---|
| 3.8 | 0.536 | 76.61 | 0.7697 |
| 4.0 | 0.701 | 73.01 | 0.7532 |
| 4.13 | 0.808 | 62.10 | 0.6958 |
| 4.3 | 0.924 | 50.55 | 0.6205 |
| 4.5 | 1.085 | 37.52 | 0.5128 |

## 7. Gutenberg–Richter Analysis

The Gutenberg–Richter magnitude–frequency relation $\log_{10} N = a - bM$ is fitted to the merged catalog above $M_c = 4.13$ using the maximum-likelihood estimator of Aki (1965):

$$b = \frac{\log_{10} e}{\bar M - (M_c - 0.05)} \approx 0.808, \quad \sigma_b = \frac{b}{\sqrt{N \ln 10}} \approx 0.010.$$

The fitted $b$-value of $0.808 \pm 0.010$ is slightly below the global tectonic average of $b \approx 1.0$ but consistent with the regional predominance of intermediate-focus subduction events.

**Figure 2**: Frequency–magnitude distribution and GR fit (fig02_fmd_gr.png).

### 7.1 Magnitude-threshold rates and probabilities

| Threshold | $N$ | Rate (yr⁻¹) | $P_{7d}$ | $P_{7d}^{lo}$ | $P_{7d}^{hi}$ | $P_{30d}$ | $P_{1y}$ |
|---|---|---|---|---|---|---|---|
| $M\geq4.5$ | 1947 | 37.52 | 0.5128 | 0.7275 | 0.1291 | 0.9541 | 1.0000 |
| $M\geq5.0$ | 534  | 10.29 | 0.1790 | 0.4202 | — | 0.5706 | 1.0000 |
| $M\geq5.5$ | 70   | 1.35  | 0.0255 | 0.0975 | — | 0.1049 | 0.7405 |
| $M\geq6.0$ | 22   | 0.42  | 0.0081 | 0.0198 | — | 0.0342 | 0.3456 |
| $M\geq6.5$ | 8    | 0.15  | 0.0030 | 0.0084 | — | 0.0126 | 0.1429 |
| $M\geq7.0$ | 1    | 0.019 | 0.0004 | 0.0030 | — | 0.0016 | 0.0191 |

## 8. Methodology

### 8.1 Overall framework

The forecasting task is to estimate, for each space–time cell, the probability $P(N \geq 1)$ that at least one event of magnitude above a specified threshold occurs within a specified horizon $\Delta t$. We adopt a 1°×1° spatial grid, magnitude thresholds $M_c^{\text{target}} \in \{4.5, 5.0, 5.5, 6.0, 6.5, 7.0\}$, and forecast horizons $\Delta t \in \{1\text{h}, 6\text{h}, 24\text{h}, 7\text{d}, 30\text{d}, 90\text{d}, 1\text{y}\}$. The forecast probability is $\hat P_i(\Delta t) = 1 - \exp(-\hat\lambda_i \Delta t)$.

### 8.2 Chronological validation protocol

- **Development (pre-2010)**: parameter estimation.
- **Selection (2010–2014)**: hyperparameter tuning (only v3, GBM).
- **Evaluation (2015–2023)**: nine yearly forecast origins.

### 8.3 Spatial holdout

4-fold quadrant (NW, NE, SW, SE); each model is fit on three quadrants and evaluated on the fourth.

### 8.4 Scoring rules

- **Brier score**: $B = (p_{it} - y_{it})^2$, mean over cells and origins.
- **ECE**: 7-bin expected calibration error.
- **Log score**: negative binomial log-likelihood.

### 8.5 Statistical significance

Paired block bootstrap (500 resamples over origins); permutation tests (1000 permutations); Benjamini–Hochberg FDR at $q = 0.05$.

## 9. The Spatial Poisson Model (FINAL_v1.0_FROZEN)

The production baseline is a cell-wise Spatial Poisson model on a 1°×1° grid covering the project bounding box.

### 9.1 Estimator

For each cell $i$ with area $A_i$ and training window $T_{\text{train}}$, $\hat\lambda_i = N_i^{\text{train}} / T_{\text{train}}$, and $\hat P_i(\Delta t) = 1 - \exp(-\hat\lambda_i \Delta t)$.

### 9.2 Confidence intervals

Garwood (1936) exact Poisson interval, computed by bisection on the incomplete gamma function.

### 9.3 Spatial rate map

**Figure 4**: v1 Spatial Poisson rate grid (fig04_spatial_rate.png).

### 9.4 Forecast probability map

**Figure 11**: v1 7-day $M\geq4.5$ forecast probability (fig11_forecast_map.png).

### 9.5 Grid-size sensitivity

| Grid | Brier (7d, $M4.5$) | ECE | $N_{\text{cells}}$ |
|---|---|---|---|
| 0.5° | 0.0001 | 0.0045 | 256 |
| 1.0° | 0.0009 | 0.0175 | 64 |
| 2.0° | 0.0091 | 0.0667 | 16 |

## 10. ETAS Experiments

### 10.1 Stage 5: standard ETAS on the USGS catalog

At all three completeness thresholds ($M_c \in \{4.0, 4.5, 5.0\}$), the productivity parameter $K$ is at its lower bound $K \approx 10^{-8}$, the magnitude-scaling exponent $\alpha$ is at its lower bound $\alpha \approx 0$, and the temporal decay parameter $c$ is at its upper bound $c \approx 1$ day. Branching ratio $n = 0.0$. The model collapses to background-only Poisson.

### 10.2 Phase A: the base-10 bug fix

A subsequent code review identified a non-trivial numerical issue in the productivity term: base-10 exponentiation instead of natural exponential. The Phase A bug fix corrected the base and re-fit. The result: $K \approx 0$ and $\alpha \approx 0$ persist. The base-10 bug had not been the cause.

### 10.3 Depth-stratified ETAS

$K \approx 10^{-8}$ and $\alpha \approx 0$ in all three strata (shallow 0–25 km, intermediate 25–70 km, deep 70–200 km). Branching ratio 0.0 in all three strata.

### 10.4 The $R \approx 24\times$ / $K \approx 0$ contradiction

The non-parametric Omori diagnostic reveals $R \approx 24\times$ at $\Delta t \approx 18$ min on the full catalog, decaying to $R \approx 1$ by $\Delta t \approx 1$ day. The standard Omori–Utsu kernel with $c \geq 1$ day cannot represent clustering at the ~18-minute timescale.

**Figure 6**: Non-parametric Omori diagnostic (fig06_omori.png).

## 11. Machine-Learning Experiments

### 11.1 Feature catalog

Static seismicity rates, recent event counts (7/30/90 day windows), magnitude statistics, spatial coordinates, mean hypocentral depth.

### 11.2 Models

Gradient-boosted decision trees (XGBoost, 200 trees, depth 4, lr 0.05) and $\ell_2$-regularised logistic regression.

### 11.3 Retrospective evaluation

GBM attains Brier 0.0327 and ECE 0.0206 at $M4.5/7$d — worse than v1 (Brier 0.0242, ECE 0.0087).

### 11.4 Spatial holdout failure

The GBM degrades dramatically when trained on three quadrants and evaluated on the fourth; it has memorised the spatial pattern. v1 trivially generalises.

## 12. Bayesian Hierarchical Experiments (v2)

### 12.1 Model

$\lambda_i \mid a, b \sim \text{Gamma}(a, b)$, $N_i \mid \lambda_i \sim \text{Poisson}(\lambda_i T)$. Posterior mean: $\hat\lambda_i^{\text{Bayes}} = (a + N_i) / (b + T)$.

### 12.2 Empirical-Bayes hyperparameters

$\hat a = 2.34$, $\hat b = 0.21$ yr, prior mean $a/b = 11.1$ events yr⁻¹, shrinkage weight $w = 0.996$.

### 12.3 Retrospective evaluation

| Config | $B_{\text{v1}}$ | $B_{\text{v2}}$ | $\Delta B$ | $\text{CI}^{\text{boot}}_{\text{lo}}$ | $\text{CI}^{\text{boot}}_{\text{hi}}$ | $\text{ECE}_{\text{v1}}$ | $\text{ECE}_{\text{v2}}$ | Verdict |
|---|---|---|---|---|---|---|---|---|
| $M4.5/7$d  | 0.01502 | 0.01502 | $-3\times10^{-6}$ | $-1.8\times10^{-5}$ | $+8\times10^{-6}$ | 0.00501 | 0.00499 | No improvement |
| $M4.5/30$d | 0.04998 | 0.05002 | $-3.7\times10^{-5}$ | $-1.06\times10^{-4}$ | $+2.7\times10^{-5}$ | 0.01890 | 0.01683 | No improvement |
| $M5.0/7$d  | 0.00512 | 0.00513 | $-3\times10^{-6}$ | $-1.2\times10^{-5}$ | $+2\times10^{-6}$ | 0.00204 | 0.00200 | No improvement |
| $M5.0/30$d | 0.00991 | 0.00991 | $-5\times10^{-6}$ | $-5.2\times10^{-5}$ | $+3.1\times10^{-5}$ | 0.00294 | 0.00312 | No improvement |

### 12.4 Verdict

**B (PROMISING — uncertainty improvement only)**. Not promoted to production.

## 13. Adaptive Spatial Experiments (v3)

### 13.1 Variants

- A: Gaussian, fixed bandwidth (best $h = 0.25°$)
- B: Gaussian, nearest-neighbour (best $k = 10$)
- C: Epanechnikov, fixed (best $h = 0.5°$)
- D: Epanechnikov, nearest-neighbour (best $k = 50$)

### 13.2 Retrospective evaluation (variant A)

| Config | $B_{\text{v1}}$ | $B_{\text{v3}}$ | $\Delta B$ | $\text{CI}_{\text{lo}}$ | $\text{CI}_{\text{hi}}$ | $p_{\text{perm}}$ | Verdict |
|---|---|---|---|---|---|---|---|
| $M4.5/7$d  | 0.01502 | 0.01497 | $-4.8\times10^{-5}$ | $-1.5\times10^{-4}$ | $+3.4\times10^{-4}$ | 0.731 | Tie |
| $M4.5/30$d | 0.04998 | 0.04971 | $-2.7\times10^{-4}$ | $-9.3\times10^{-4}$ | $+1.7\times10^{-3}$ | 0.723 | Tie |
| $M5.0/7$d  | 0.00512 | 0.00510 | $-2.6\times10^{-5}$ | $-9\times10^{-6}$ | $+9.3\times10^{-5}$ | 0.598 | Tie |
| $M5.0/30$d | 0.00991 | 0.00984 | $-6.8\times10^{-5}$ | $-1.8\times10^{-4}$ | $+4.1\times10^{-4}$ | 0.658 | Tie |

### 13.3 Posterior predictive check (FAIL)

| Statistic | Observed | Sim. mean | Sim. 95% CI |
|---|---|---|---|
| Total events | 1890 | 2365 | [2268, 2457] |
| Occupied cells | 61 | 61.3 | [58, 64] |
| Max cell count | 217 | 396.8 | [362, 437] |
| Gini coefficient | 0.649 | 0.684 | [0.667, 0.699] |
| Top-3 fraction | 0.310 | 0.413 | [0.393, 0.434] |

### 13.4 Verdict

**D (REJECTED)**. Simulated catalogs over-concentrate events.

**Figure 13**: v3 grid sensitivity and PPC (fig13_grid_sensitivity.png).

## 14. Region-Specific ETAS Experiments (v4)

### 14.1 Variants

- A: Baseline (Phase-A bug-fixed standard ETAS)
- B: Depth-stratified (shallow / intermediate / deep)
- C: Depth-dependent spatial kernel
- D: Exponential temporal kernel

### 14.2 Parameter estimates

| Variant | $K$ | $\alpha$ | $c$ (d) | $p$ | $\sigma$ (km) | $\tau$ (d) | BR | $\mu$ (yr⁻¹) | log-lik |
|---|---|---|---|---|---|---|---|---|---|
| A Baseline | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | — | 0.0 | 51.78 | $-13585$ |
| B Shallow | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | — | 0.0 | 7.40 | $-2473$ |
| B Intermediate | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | — | 0.0 | 23.37 | $-6817$ |
| B Deep | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | — | 0.0 | 21.01 | $-6211$ |
| C Depth-spatial | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | — | 0.0 | 51.78 | $-13585$ |
| D Exponential | $10^{-8}$ | 0.0 | 0.05 | 1.1 | 10.0 | 0.01 | 0.0 | 51.78 | $-13585$ |

### 14.3 Depth-stratified clustering

| Regime | $N$ | CV$_{\text{IET}}$ | Median IET (d) | $K$ |
|---|---|---|---|---|
| Shallow (0–25 km) | 273 | 1.65 | 23.18 | $10^{-8}$ |
| Intermediate (25–70 km) | 862 | 1.32 | 9.12 | $10^{-8}$ |
| Deep (70–200 km) | 775 | 1.36 | 10.40 | $10^{-8}$ |
| All | 1910 | 1.36 | 3.90 | $10^{-8}$ |

### 14.4 Short-horizon evaluation

| Horizon | $n_+$ | $B_{\text{v1}}$ | $B_{\text{v4}}$ | $\Delta B$ | $p_{\text{perm}}$ | Verdict |
|---|---|---|---|---|---|---|
| 1h  | 0 | 0.0 | 0.0 | 0.0 | 0.003 | Tie |
| 6h  | 0 | 0.0 | 0.0 | 0.0 | 0.003 | Tie |
| 24h | 0 | $8\times10^{-6}$ | $1\times10^{-6}$ | $-7\times10^{-6}$ | 0.003 | Tie |
| 7d  | 9 | 0.01502 | 0.01544 | $+4.3\times10^{-4}$ | 0.471 | v4 worse |
| 30d | 34 | 0.04998 | 0.05623 | $+6.2\times10^{-3}$ | 0.034 | v4 worse |
| 90d | 74 | 0.09358 | 0.11307 | $+1.9\times10^{-2}$ | 0.008 | v4 worse |

### 14.5 Spatial holdout

| Quadrant | $B_{\text{v1}}$ | $B_{\text{v4}}$ | $\Delta B$ | Verdict |
|---|---|---|---|---|
| NW | $10^{-6}$ | $6\times10^{-5}$ | $+5.9\times10^{-5}$ | v4 worse |
| NE | 0.01279 | 0.01373 | $+9.4\times10^{-4}$ | v4 worse |
| SW | 0.01384 | 0.01373 | $-1.1\times10^{-4}$ | v4 better |
| SE | 0.03342 | 0.03424 | $+8.2\times10^{-4}$ | v4 worse |

### 14.6 Verdict

**D (REJECTED)**. $K \approx 0$ in all 4 variants. v4 is significantly worse than v1 at 30d and 90d. Final answer: **NO**.

## 15. Validation Framework

### 15.1 Three-way chronological split

- Development (pre-2010): parameter estimation
- Selection (2010–2014): hyperparameter tuning
- Evaluation (2015–2023): nine yearly origins, untouched

### 15.2 Spatial holdout

4-fold quadrant (NW, NE, SW, SE); fit on three, evaluate on the fourth.

### 15.3 Bootstrap CIs

500 resamples over the nine origins.

### 15.4 Permutation tests

1000 permutations of model labels within each origin.

### 15.5 BH FDR correction

Applied at $q = 0.05$ for multiple comparisons.

**Figure 9**: Spatial holdout 4-fold quadrant (fig09_spatial_holdout.png).

## 16. Retrospective Evaluation

### 16.1 Headline numbers

v1 (production): Brier 0.0242 (ECE 0.0087) at 7d, Brier 0.0763 (ECE 0.0322) at 30d.
v1 (strictly leak-free): Brier 0.01502 at $M4.5/7$d.

### 16.2 Calibration

**Figure 8**: v1 reliability diagram (fig08_calibration.png).

### 16.3 Large-event uncertainty

**Figure 12**: Large-event uncertainty (fig12_large_event_uncertainty.png).

### 16.4 Summary

| Model | $B_{M4.5/7d}$ | $\text{ECE}_{M4.5/7d}$ | $B_{M4.5/30d}$ | $\text{ECE}_{M4.5/30d}$ | $B_{M5.0/7d}$ | Verdict |
|---|---|---|---|---|---|---|
| v1 Spatial Poisson | **0.01502** | **0.00501** | **0.04998** | **0.01890** | **0.00512** | PRODUCTION |
| v2 Bayesian hier. | 0.01502 | 0.00499 | 0.05002 | 0.01683 | 0.00513 | B (PROMISING) |
| v3 Adaptive (A) | 0.01497 | 0.00380 | 0.04971 | 0.01549 | 0.00510 | D (REJECTED) |
| v4 Region-spec. ETAS | 0.01544 | 0.00787 | 0.05623 | 0.02620 | 0.00519 | D (REJECTED) |

## 17. Prospective Monitoring

### 17.1 Immutable ledger

Each forecast is a JSON document with issuance timestamp, per-cell probabilities, Garwood CIs, and SHA-256 hash chain.

### 17.2 Evidence levels

- Level 0 (INSUFFICIENT EVIDENCE): <20 origins, <20 events. **Current status.**
- Level 1 (PRELIMINARY): 20–50 origins.
- Level 2 (PROVISIONAL): 50–100 origins.
- Level 3 (CONFIRMED): 100+ origins.
- Level 4 (OPERATIONAL): 200+ origins.

### 17.3 Current status

Level 0: 9 origins, 73 $M \geq 4.5$ events observed. 2 v1 forecasts and 1 v2 forecast in the ledger. No v3 or v4 ledger.

**Figure 14**: Prospective monitoring dashboard (fig14_prospective_monitoring.png).

## 18. Results

### 18.1 Model comparison

| Model | Brier (7d, $M4.5$) | ECE | Status |
|---|---|---|---|
| Spatial Poisson (v1) | **0.0242** | **0.0087** | VALIDATED |
| ETAS ($K \approx 0$) | 0.4355 | 0.6404 | PRELIMINARY |
| ML (GBM) | 0.0327 | 0.0206 | VALIDATED (no skill) |
| Coulomb | N/A | N/A | DATA-LIMITED |

**Figure 7**: Model comparison (fig07_model_comparison.png).

### 18.2 Final operational forecasts

| Threshold | Rate (yr⁻¹) | $P_{7d}$ | $P_{30d}$ | $P_{1y}$ | Status |
|---|---|---|---|---|---|
| $M\geq4.5$ | 37.52 | 0.5128 | 0.9541 | 1.0000 | VALIDATED |
| $M\geq5.0$ | 10.29 | 0.1790 | 0.5706 | 1.0000 | VALIDATED |
| $M\geq5.5$ | 1.35 | 0.0255 | 0.1049 | 0.7405 | VALIDATED |
| $M\geq6.0$ | 0.42 | 0.0081 | 0.0342 | 0.3456 | VALIDATED |
| $M\geq6.5$ | 0.15 | 0.0030 | 0.0126 | 0.1429 | DATA-LIMITED |
| $M\geq7.0$ | 0.019 | 0.0004 | 0.0016 | 0.0191 | DATA-LIMITED |

**Figure 15**: Final v1 forecast map (fig15_final_forecast_map.png).
**Figure 16**: Candidate comparison and verdicts (fig16_candidate_comparison.png).

### 18.3 Stage 5 ETAS parameters

| Parameter | $M_c=4.0$ | $M_c=4.5$ | $M_c=5.0$ |
|---|---|---|---|
| $K$ | $10^{-8}$ | $10^{-8}$ | $10^{-8}$ |
| $\alpha$ | 0.0 | 0.0 | 0.0 |
| $c$ (days) | 1.0 | 1.0 | 1.0 |
| $p$ | 1.01 | 1.01 | 1.01 |
| BR (analytic) | 0.0 | 0.0 | 0.0 |

## 19. Discussion

### 19.1 Why does ETAS select $K \approx 0$?

The standard Omori–Utsu kernel $(t+c)^{-p}$ with $c \geq 1$ day cannot represent rate enhancement at sub-day lags. At $\Delta t = 18$ min, the kernel value is $(0.0125 + 1)^{-1.01} \approx 0.988$, essentially equal to its asymptotic value of 1. The MLE correctly prefers $K = 0$.

### 19.2 Why is the Omori signal so short-lag?

Three plausible explanations: (1) **event relocations** (most likely — duplicate agency reports within 2 minutes); (2) genuine short-lag triggering (deep subduction may have very short relaxation timescales); (3) network artefacts. The absence of magnitude scaling ($\alpha = 0$) supports explanation (1).

### 19.3 Why does adaptive smoothing fail?

The kernel piles simulated events into the few high-rate cells, producing a higher Gini coefficient and top-3 fraction than observed. The cell-wise Poisson assigns zero rate to empty cells and reproduces the observed concentration.

### 19.4 Why does the Bayesian model not improve Brier?

With $T = 36$ years, the shrinkage weight $w = 0.996$: data dominate the prior by three orders of magnitude. The shrinkage only differs from v1 for cells with $N_i < 5$, where Brier is dominated by binomial variance.

### 19.5 Why does region-specific ETAS not rescue $K$?

$K \approx 0$ survives depth stratification, depth-dependent spatial kernels, and exponential temporal kernels. The contradiction with the Omori signal is genuine.

**Figure 6** (repeated): Omori diagnostic (fig06_omori.png).

## 20. Limitations

- **Catalog completeness**: $M_c = 4.13$; deduplication window of 120 s misses pairs within 2 minutes.
- **Missing GCMT/BMD/historical data**: the single largest source of epistemic uncertainty.
- **Small evaluation sample**: 9 origins, 73 events; bootstrap CIs $\pm 10^{-3}$.
- **Large-event uncertainty**: $M \geq 7$ 95% CI spans an order of magnitude.
- **Spatial grid coarseness**: 1°×1° is coarser than typical aftershock zones.
- **Absence of receiver fault geometry**: Coulomb candidate DATA-LIMITED.

## 21. Future Work

### 21.1 Tier 1: data acquisition

- **GCMT**: authoritative $M_w$ and focal mechanisms (1976+)
- **BMD**: local-network detections ($M \geq 3.0$)
- **ISC-GEM**: historical catalog (1900–1972)
- **Palaeoseismic recurrence**: 200–500-year intervals

### 21.2 Tier 2: model development

- Coulomb stress change candidate (with GCMT focal mechanisms)
- Region-specific magnitude scaling (with more $M \geq 6$ events)
- Bayesian ETAS (with informative priors on $K, \alpha$)

### 21.3 Tier 3: operational

- Prospective monitoring continuation (target Level 1 by 2028)
- Forecast product expansion (BMD-dependent)
- Stakeholder engagement (BMD, CTBTO, ISC)

## 22. Conclusion

1. The merged USGS+ISC catalog (5,779 events, $M \geq M_c = 4.13$, 1973–2024) supports $b = 0.808 \pm 0.010$ and a regional $M \geq 4.5$ rate of 37.5 events yr⁻¹.

2. A 1°×1° cell-wise Spatial Poisson model (FINAL_v1.0_FROZEN) is the only prospectively validated production baseline, attaining Brier 0.0242 (ECE 0.0087) at 7-day $M \geq 4.5$ horizon and Brier 0.0763 (ECE 0.0322) at 30-day horizon over nine chronological origins (2015–2023).

3. Three candidate extensions — Bayesian hierarchical (v2), adaptive spatial smoothing (v3), and region-specific ETAS (v4) — are developed under a strictly leak-free dev/select/eval protocol. None delivers a statistically defensible improvement: v2 changes Brier by $\Delta \approx 0$ (verdict B); v3 fails PPC (verdict D, REJECTED); v4 collapses to $K \approx 0$ (verdict D, REJECTED).

4. The persistent Omori-style rate enhancement of $R \approx 24\times$ at $\Delta t \approx 18$ min coexists with $K \approx 0$ and $\alpha = 0$ in every ETAS variant.

5. The most plausible interpretation: the short-lag clustering is dominated by event relocations and duplicate agency reports rather than genuine tectonic aftershock cascades.

6. The principal data limitation is the absence of GCMT, BMD, and ISC-GEM catalogs.

The Spatial Poisson baseline (FINAL_v1.0_FROZEN) remains the production forecasting model for Bangladesh. The $R \approx 24\times$ / $K \approx 0$ contradiction is a catalog artefact, not a tectonic signal. No v5 model will be developed.

## References

1. Aki, K. (1965). Maximum likelihood estimate of $b$ in the formula $\log N = a - bM$ and its confidence limits. *Bull. Earthq. Res. Inst. Univ. Tokyo*, 43, 237–239.
2. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS-B*, 57(1), 289–300.
3. Clayton, D., & Kaldor, J. (1987). Empirical Bayes estimates of age-standardized relative risks for use in disease mapping. *Biometrics*, 43(3), 671–681.
4. Garwood, F. (1936). Fiducial limits for the Poisson distribution. *Biometrika*, 28(3/4), 437–442.
5. Gardner, J. K., & Knopoff, L. (1974). Is the sequence of earthquakes in Southern California, with aftershocks removed, Poissonian? *BSSA*, 64(5), 1363–1367.
6. Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in California. *BSSA*, 34(4), 185–188.
7. Hainzl, S., & Ogata, Y. (2008). Detecting fluid signals in seismicity data through statistical earthquake modeling. *JGR*, 113, B07303.
8. Marsan, D., & Lengliné, O. (2008). Extending earthquakes' reach through cascading. *Science*, 319(5866), 1076–1079.
9. Marzocchi, W., & Taroni, M. (2012). Probabilistic earthquake forecasting: A new prospective. *SRL*, 83(3), 479–483.
10. Ni, J. F., et al. (1989). Accretionary tectonics of Burma and the three-dimensional geometry of the Burma subduction zone. *Geology*, 17(1), 68–71.
11. Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *JASA*, 83(401), 9–27.
12. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303–1313.
13. Reasenberg, P. (1985). Second-order moment of central California seismicity, 1969–1982. *JGR*, 90(B7), 5479–5495.
14. Scordilis, E. M. (2006). Empirical global relations converting $M_S$ and $m_b$ to moment magnitude. *J. Seismol.*, 10(2), 225–236.
15. Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
16. Steckler, M. S., et al. (2008). GPS and gravity constraints on active tectonics of the Burma subduction zone. *GJI*, 175(1), 239–255.
17. Stepp, J. C. (1972). Analysis of completeness of the earthquake sample in the Puget Sound area. *NOAA Tech. Report* ERL 267-ESL 30, 16–28.
18. Styron, R., & Hetland, E. (2020). GEM Global Active Faults Database (GAFD). GitHub repository.
19. Wiemer, S., & Wyss, M. (2000). Minimum magnitude of completeness in earthquake catalogs. *BSSA*, 90(4), 859–869.
20. Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues. *BSSA*, 95(2), 684–698.
21. Zhuang, J., et al. (2011). Stochastic declustering of space-time earthquake occurrences. *JASA*, 97(458), 369–380.

---

## Appendices (A–J)

The PDF version includes the following appendices, which contain the detailed mathematical derivations, parameter estimation tables, posterior predictive check results, Stage 5 ETAS convergence diagnostics, operational forecast product specification, glossary, catalog construction audit, reproducibility manifest, statistical power analysis, and ethical/operational considerations:

- **Appendix A**: Mathematical Derivations (Garwood interval, Gamma–Poisson posterior, KDE, ETAS likelihood, branching ratio, paired block bootstrap, BH FDR)
- **Appendix B**: Detailed Parameter Estimation (v1 per-cell rates, v2 prior sensitivity, v3 all variants, v4 Omori by depth)
- **Appendix C**: Posterior Predictive Check Details
- **Appendix D**: Stage 5 ETAS Convergence Diagnostics
- **Appendix E**: Operational Forecast Product Specification
- **Appendix F**: Glossary
- **Appendix G**: Catalog Construction Audit
- **Appendix H**: Reproducibility Manifest
- **Appendix I**: Statistical Power Analysis
- **Appendix J**: Ethical and Operational Considerations

See `BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V2.pdf` and `BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V2.tex` for the full typeset appendices.

---

*End of publication. This paper was generated by the Bangladesh Earthquake Forecasting Project publication pipeline. The frozen CSVs, ledgers, and source code referenced throughout are available in the project repository under `FINAL_PUBLICATION_PACKAGE/`. The model code is FROZEN at version FINAL_v1.0_FROZEN; no further model development is planned.*
