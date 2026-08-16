# Can Probabilistic Earthquake Forecasting for Bangladesh Improve Upon Historical Spatial Seismicity Rates?

## A Chronological Validation Study Using the Merged USGS–ISC Catalog

---

## Abstract

We investigate whether statistical, machine-learning, or physics-based models can improve probabilistic earthquake forecasts for Bangladesh beyond historical spatial seismicity rates. Using a merged USGS+ISC catalog of 5,779 events (M≥2.4, 1973–2024, Mc≈4.13, b≈0.808), we compare Spatial Poisson, uniform Poisson, ETAS (epidemic-type aftershock sequence, base-10 formulation with Gardner–Knopoff declustered background), gradient-boosting machine learning, and Coulomb stress-transfer models under strict chronological validation on an untouched 2015–2024 evaluation period. Spatial Poisson achieves the lowest Brier score (0.024 at 7 days, 0.076 at 30 days) with the best calibration (ECE=0.009). Neither ETAS (K≈0, no triggering detected in-sample; Brier=0.44) nor ML (Brier=0.033) demonstrates statistically defensible incremental skill; ML additionally fails spatial holdout. A non-parametric Omori diagnostic reveals strong short-lag post-mainshock temporal clustering (R≈24× for M≥5), indicating that the ETAS failure reflects model misspecification rather than absence of triggering. Coulomb forecasting remains disabled due to missing validated receiver-fault geometry. Forecast uncertainty becomes substantial for rare large events (M≥7: N=1, 95% CI spanning an order of magnitude). We conclude that historical spatial seismicity rates provide the strongest validated probabilistic forecasting baseline for the available Bangladesh catalog, and that the tested ETAS and ML formulations do not provide incremental predictive skill beyond this baseline.

---

## 1. Introduction

Bangladesh lies at the junction of the Indian, Eurasian, and Burma plates, surrounded by active tectonic structures including the Dauki Fault, the Indo-Burman fold belt, the Arakan megathrust, and the Shillong Plateau fault system. Historical great earthquakes (1762 Arakan M≈8.5, 1897 Shillong M8.7) demonstrate the region's seismic potential, yet local seismic monitoring is sparse and earthquake forecasting research is limited.

Probabilistic earthquake forecasting — estimating the probability of future events conditional on observed seismicity — is fundamentally different from deterministic prediction. The goal is not to predict the exact time, location, and magnitude of a specific earthquake, but to provide calibrated probability estimates that can inform risk reduction.

This study addresses a central question: **Can statistical, machine-learning, or physics-based models improve probabilistic earthquake forecasts for Bangladesh beyond what historical spatial seismicity rates alone provide?**

## 2. Data

### 2.1 Catalog Sources

We integrate two publicly accessible earthquake catalogs:

- **USGS ComCat** (FDSN web service): 2,293 events, M≥2.5 query, 1973–2024, magnitude floor M3.2.
- **ISC Bulletin** (FDSN web service): 5,576 events, M≥3.0 query, 1973–2024, magnitude floor M2.4.

The ISC catalog provides 2.4× more events than USGS alone and extends the magnitude floor from M3.2 to M2.4, resolving the completeness estimation problem that limited earlier USGS-only analyses.

### 2.2 Unavailable Data

The following data sources could not be acquired and remain unavailable:

| Source | Status | Impact |
|--------|--------|--------|
| GCMT (focal mechanisms) | Unavailable | Coulomb disabled; ETAS spatial kernels uninformed |
| ISC-GEM (1904+) | Requires registration | No pre-1973 instrumental extension |
| BMD (local bulletins) | Requires formal request | M2–3 local events missing |
| Historical (Alam & Dominey-Howes 2016) | Requires manual transcription | No pre-1900 great earthquakes |
| Receiver-fault geometry | Not published in machine-readable form | Coulomb disabled |

### 2.3 Catalog Construction

USGS and ISC observations are matched by time (120 s window) and space (50 km window) into canonical events. Original magnitudes and magnitude types are preserved for every observation; Mw is derived only when a validated published conversion exists (Scordilis 2006 for mb and MS; Mw-family types retained as authoritative). The merged catalog contains **5,779 canonical events** with full provenance.

## 3. Methods

### 3.1 Completeness

Magnitude of completeness (Mc) is estimated by four independent methods: Maximum Curvature (MAXC), Goodness-of-Fit Test (GFT), Entire-Magnitude-Range (EMR), and Stepp's method. The recommended Mc is the median of the four estimates.

### 3.2 Gutenberg–Richter

The b-value is estimated by the Aki–Utsu maximum-likelihood estimator with Shi–Bolt (1982) uncertainty.

### 3.3 Spatial Poisson

The spatial Poisson model assigns each 1° grid cell a rate λ_cell = N_cell(\<t) / exposure(\<t), computed causally (only events before the forecast origin). The cell probability is P_cell = 1 − exp(−λ_cell · Δt).

### 3.4 ETAS

The Epidemic-Type Aftershock Sequence model uses the conditional intensity:

λ(x,y,t) = μ(x,y) + Σ K · 10^{α(M_i−Mc)} · g(t−t_i) · f(x−x_i, y−y_i)

where g is the normalized Omori–Utsu kernel and f is a power-law spatial kernel. The background μ(x,y) is estimated from Gardner–Knopoff declustered mainshocks via KDE. Parameters are estimated by MLE (L-BFGS-B, multi-start). The formulation uses **base-10** productivity (per the research report specification), corrected from an earlier base-e error (Phase A).

### 3.5 Machine Learning

Gradient boosting (200 trees, depth 3, learning rate 0.1) with 43 causal features (historical rate, temporal, magnitude, spatial, depth, clustering). Features use only information available before each forecast origin. No neural networks (insufficient temporal structure to justify).

### 3.6 Coulomb

A mathematical prototype (Okada 1992 point-source, elastic half-space) is implemented and unit-tested with synthetic geometry. Real forecasting is **disabled** due to missing validated receiver-fault geometry.

### 3.7 Uncertainty

Aleatory uncertainty (Poisson counting, Garwood exact CI) and epistemic uncertainty (magnitude conversion σ=0.41 for mb→Mw; Mc sensitivity) are propagated separately through the forecast chain.

## 4. Validation Framework

### 4.1 Chronological Split

| Period | Years | Purpose |
|--------|-------|---------|
| Development | 1973–2006 | Model development and initial training |
| Selection | 2006–2015 | Model selection and hyperparameter choices |
| **Evaluation** | **2015–2024** | **Untouched final evaluation** |

No model selection, feature selection, parameter tuning, or methodological decision uses the evaluation period.

### 4.2 Scoring Rules

- **Primary**: Brier score, log-likelihood, information gain vs. Spatial Poisson, expected calibration error (ECE)
- **Secondary**: ROC-AUC, PR-AUC
- **Statistical testing**: block bootstrap (500 resamples over forecast origins), permutation tests, Bonferroni/BH correction

### 4.3 Spatial Holdout

4-fold quadrant holdout (NW/NE/SW/SE): train on 48 cells, test on 16 held-out cells. Tests whether ML generalizes or memorizes.

## 5. Results

### 5.1 Catalog and Completeness

| Metric | Value |
|--------|-------|
| N events | 5,779 |
| Exposure | 51.89 years |
| Mc (recommended) | 4.13 (MAXC=4.05, GFT=5.65, EMR=4.15, Stepp=4.10) |
| b-value | 0.808 ± 0.010 |
| Mean depth | 52.6 km |
| Shallow (<25 km) | 1,827 events |
| Deep (≥70 km) | 1,945 events |

### 5.2 Spatial Poisson (Primary Validated Model)

On the untouched 2015–2024 evaluation period:

| Horizon | N origins | N+ | Brier | ECE | Sharpness |
|---------|-----------|-----|-------|-----|-----------|
| 7 days | 9 | 15 | **0.0242** | **0.0087** | 0.0243 |
| 30 days | 9 | 58 | **0.0763** | **0.0322** | 0.0890 |

### 5.3 ETAS

| Parameter | Value |
|-----------|-------|
| K | 0 |
| α | 0 |
| μ (events/yr) | 62.1 |
| No triggering detected | Yes |
| Branching ratio n | 0 (subcritical) |
| ETAS Brier (7d, eval) | 0.4355 |
| ETAS beats SP | **No** |

K≈0 persists in all depth regimes (shallow, intermediate, deep) and survives the expanded catalog, corrected base-10 formulation, and declustered background.

### 5.4 Machine Learning

| Model | Brier (7d, eval) | ECE | Beats SP? |
|-------|-----------------|-----|-----------|
| Spatial Poisson | 0.0242 | 0.0087 | baseline |
| Gradient Boosting | 0.0327 | 0.0206 | **No** |
| Logistic Regression | 0.3784 | 0.3861 | **No** |

ML fails the spatial holdout (SP beats ML on all 4 held-out quadrants).

### 5.5 Omori Diagnostic

| Mainshock threshold | N mainshocks | Peak R(Δt) | Δt at peak | Omori-like? |
|---------------------|-------------|------------|------------|-------------|
| M≥5 | 534 | 24× | 0.013 d | **Yes** |
| M≥6 | 22 | 290× | 0.013 d | **Yes** |

The catalog exhibits strong short-lag post-mainshock temporal clustering. This is observed evidence, independent of any model.

### 5.6 Short-Horizon Comparison

Spatial Poisson wins at ALL horizons (1 hour through 90 days), including the shortest horizons where Omori clustering is strongest.

### 5.7 Coulomb

**Disabled** — no validated receiver-fault geometry exists. Mathematical prototype validated with synthetic geometry.

## 6. Discussion

### 6.1 Why Spatial Poisson is strongest

The catalog has high spatial concentration (Gini coefficient ≈0.87): the top 10% of grid cells contain most events. Spatial Poisson captures this heterogeneity directly. ETAS and ML add no incremental spatial information beyond what the historical rate already encodes.

### 6.2 What the ETAS result means

The K≈0 result means the standard ETAS formulation does not detect a statistically supported triggering component in-sample. This is **not** evidence that earthquake triggering is absent — the Omori diagnostic independently confirms strong short-lag clustering. The failure reflects **model misspecification**: the standard 2D ETAS with Omori–Utsu temporal decay and power-law spatial kernel cannot represent the clustering pattern in this catalog. Possible causes include the deep Indo-Burman subduction character (mean depth 52.6 km), the mix of shallow crustal and deep intra-slab events, or a temporal decay pattern that deviates from Omori–Utsu at the relevant timescales.

### 6.3 What the ML result means

ML initially appeared to outperform Poisson, but this was an artifact of comparing against uniform Poisson (which ignores spatial heterogeneity). When compared against Spatial Poisson, ML loses. The spatial holdout confirms that ML memorizes historically active cells rather than learning transferable relationships. This is consistent with heavy reliance on historical spatial heterogeneity; the experiments do not prove the internal mechanism of the model.

### 6.4 What the Omori diagnostic means

The strong short-lag clustering (R≈24×) is real observed evidence that should not be ignored. It is scientifically separate from the ETAS forecasting result. The clustering could reflect physical triggering, reporting effects, catalog heterogeneity, or magnitude uncertainty — these are not distinguished by the available data.

### 6.5 Implications of catalog limitations

The absence of GCMT, BMD, ISC-GEM, and historical data limits stronger physical inference. GCMT would enable focal-mechanism-informed ETAS spatial kernels and Coulomb stress calculations. BMD would provide M2–3 local events and more aftershocks. These data could change the conclusions.

## 7. Limitations

1. **Limited large earthquakes**: M≥7 has N=1; precise recurrence estimates are impossible.
2. **Catalog heterogeneity**: 92% of USGS magnitudes are mb (σ=0.41 conversion uncertainty).
3. **No GCMT**: Coulomb disabled; ETAS spatial kernels uninformed by focal mechanisms.
4. **No BMD**: Local M2–3 events missing; Mc could be lower.
5. **No ISC-GEM/historical**: No pre-1973 extension; no Mmax constraint.
6. **No receiver-fault geometry**: Coulomb disabled.
7. **Limited statistical power**: M≥5.5+ has insufficient events for reliable ML.
8. **Model misspecification**: Standard ETAS cannot represent observed clustering.
9. **Finite observation period**: 52 years is short for M≥7 recurrence.
10. **Deep Indo-Burman subduction**: Standard ETAS designed for shallow crustal seismicity.

## 8. Conclusions

Historical spatial seismicity rates provide the strongest validated probabilistic forecasting baseline for the available Bangladesh earthquake catalog. On an untouched 2015–2024 evaluation period, the tested ETAS and machine-learning formulations did not demonstrate statistically defensible incremental predictive skill beyond this spatial baseline. The catalog nevertheless exhibits strong short-lag post-mainshock temporal clustering, so the inability of the tested ETAS formulations to improve forecasts should not be interpreted as evidence that earthquake triggering is absent. Forecast uncertainty becomes substantial for rare large-magnitude events, and missing local, focal-mechanism, and historical data limit stronger physical inference.

**What this does NOT claim:**
- Does not claim earthquakes cannot be predicted
- Does not claim Bangladesh has no earthquake triggering
- Does not claim ETAS proves there are no aftershocks
- Does not claim ML is useless
- Does not claim Coulomb does not work
- Does not present a precise M≥7 probability

**Model development is FROZEN (FINAL_v1.0). Future data acquisition would constitute a new research revision.**

---

*Data: USGS ComCat (public), ISC Bulletin (public). Code: Python 3.12, numpy, scipy, scikit-learn. All results reproducible from `run_final.py`. Random seed: 42.*
