# V3 Adaptive Spatial Smoothing — Experiment Report

> Control: FINAL_v1.0_FROZEN (Spatial Poisson, immutable)
> Comparator: FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL (Bayesian hierarchical)
> Candidate: FINAL_v3.0_CANDIDATE_ADAPTIVE_SPATIAL (this experiment)

> Generated: 2026-08-12T09:40:53.056743+00:00

## 0. Executive Summary

**Best v3 variant:** `D_epanechnikov_nn`

**Mean Brier (4 configs, 2015-2023):** v3 = 0.01985, v1 = 0.02001, v2 = 0.02002
**ΔBrier (v3−v1):** -0.00015 | **ΔBrier (v3−v2):** -0.00016

**Bootstrap CI excludes zero in favour of v3:** 0/4 vs v1, 0/4 vs v2

**Posterior predictive check:** total=FAIL, Gini=FAIL
**Integrity audit:** PASS

See Section 13 for the formal verdict (A/B/C/D).

## 1. Scientific Motivation

The existing v1/v2 analysis found very strong spatial heterogeneity (Gini ≈ 0.87): a small number of 1° cells contain a disproportionate fraction of seismicity. A rigid 1° grid creates artificial discontinuities between neighbouring cells, and sparse cells produce unstable local rate estimates. This experiment tests whether a spatially continuous adaptive kernel estimator provides genuine incremental predictive information beyond the existing baselines.

**Null hypothesis H₀:** Adaptive spatial smoothing provides no statistically significant predictive improvement over the existing spatial-rate baseline.
**Alternative H₁:** Adaptive spatial smoothing improves out-of-sample probabilistic forecasting while maintaining or improving calibration and uncertainty behaviour.

## 2. Data and Splits (no leakage)

- Catalog: USGS + ISC merged (5,779 events), 1973-02-10 to 2024-12-30.
- Mc ≈ 4.13 (validated, frozen from v1.0)
- b ≈ 0.808 (validated, frozen from v1.0)
- **Development period:** events before 2010-01-01 (used for fitting)
- **Selection period:** 2010-2014 (5 yearly origins) — used ONLY for bandwidth / k selection. Evaluation-period data is NOT used at any point during selection.
- **Evaluation period:** 2015-2023 (9 yearly origins, UNTOUCHED)
- Forecast configs: M≥4.5/7d, M≥4.5/30d, M≥5.0/7d, M≥5.0/30d
- Spatial domain: 20.0–28.0°N, 88.0–96.0°E (1° grid, 64 cells)
- Bandwidth candidates (fixed): [0.25, 0.5, 1.0, 2.0] deg
- k candidates (NN): [10, 25, 50]
- Bootstrap resamples: 200 (epistemic uncertainty); 500 (paired bootstrap CIs)
- Permutation test: 1000 permutations

## 3. Model Family

| Variant | Kernel | Adaptive | Selection target |
|---------|--------|----------|------------------|
| A_gaussian_fixed | gaussian | False | bandwidth h (deg) |
| B_gaussian_nn | gaussian | True | k (NN) |
| C_epanechnikov_fixed | epanechnikov | False | bandwidth h (deg) |
| D_epanechnikov_nn | epanechnikov | True | k (NN) |

## 4. Bandwidth / k Selection (SELECTION period 2010-2014)

Selection metric: mean Brier score on M≥4.5/7d across the 5 selection origins.


### Variant A_gaussian_fixed
Selected best: **bandwidth_deg = 0.25** (Brier = 0.00571)

| bandwidth_deg | Mean Brier (selection) | Is best? |
|-----------|--------------------------|----------|
| 0.25 | 0.00571 | ✅ |
| 0.5 | 0.00588 |  |
| 1.0 | 0.00603 |  |
| 2.0 | 0.00614 |  |

### Variant B_gaussian_nn
Selected best: **nn_k = 10** (Brier = 0.00570)

| nn_k | Mean Brier (selection) | Is best? |
|-----------|--------------------------|----------|
| 10 | 0.00570 | ✅ |
| 25 | 0.00571 |  |
| 50 | 0.00574 |  |

### Variant C_epanechnikov_fixed
Selected best: **bandwidth_deg = 0.5** (Brier = 0.00568)

| bandwidth_deg | Mean Brier (selection) | Is best? |
|-----------|--------------------------|----------|
| 0.25 | 0.00576 |  |
| 0.5 | 0.00568 | ✅ |
| 1.0 | 0.00586 |  |
| 2.0 | 0.00603 |  |

### Variant D_epanechnikov_nn
Selected best: **nn_k = 50** (Brier = 0.00567)

| nn_k | Mean Brier (selection) | Is best? |
|-----------|--------------------------|----------|
| 10 | 0.00578 |  |
| 25 | 0.00570 |  |
| 50 | 0.00567 | ✅ |

## 5. Retrospective Evaluation (2015-2023, UNTOUCHED)

### 5.1 Per-variant aggregate Brier (mean across 4 configs)

| Variant | Mean Brier v3 | Mean Brier v1 | Mean Brier v2 | Δ(v3−v1) | Δ(v3−v2) |
|---------|---------------|---------------|---------------|-----------|-----------|
| A_gaussian_fixed | 0.01990 | 0.02001 | 0.02002 | -0.00010 | -0.00011 |
| B_gaussian_nn | 0.01988 | 0.02001 | 0.02002 | -0.00012 | -0.00014 |
| C_epanechnikov_fixed | 0.01996 | 0.02001 | 0.02002 | -0.00005 | -0.00006 |
| D_epanechnikov_nn ⭐ BEST | 0.01985 | 0.02001 | 0.02002 | -0.00015 | -0.00016 |

**Overall best v3 variant:** `D_epanechnikov_nn`

### 5.2 Detailed per-config results for best variant


#### Config M4.5_7d
- n_origins = 9, n_positive = 9
- Brier: v3 = **0.01499**, v1 = 0.01502, v2 = 0.01502
- ΔBrier (v3−v1) = -0.00002 (v3 better)
- ΔBrier (v3−v2) = -0.00003 (v3 better)
- Log-lik: v3 = -0.07402, v1 = -0.07589, v2 = -0.07571
- ECE: v3 = 0.00260, v1 = 0.00500, v2 = 0.00499
- Sharpness: v3 = 0.02370, v1 = 0.01549, v2 = 0.01522

**Paired bootstrap CIs (block over origins, 500 resamples):**

| Comparison | ΔBrier mean | 95% CI | Significant? |
|------------|-------------|--------|--------------|
| v3 vs v1 | 0.00003 | [-0.00045, 0.00068] | NOT significant |
| v3 vs v2 | 0.00003 | [-0.00045, 0.00054] | NOT significant |

**Permutation test (1000 permutations):**

| Comparison | Observed ΔBrier | Permutation p-value |
|------------|-----------------|---------------------|
| v3 vs v1 | 0.00002 | 0.952 |
| v3 vs v2 | 0.00003 | 0.964 |

#### Config M4.5_30d
- n_origins = 9, n_positive = 34
- Brier: v3 = **0.04958**, v1 = 0.04998, v2 = 0.05002
- ΔBrier (v3−v1) = -0.00040 (v3 better)
- ΔBrier (v3−v2) = -0.00044 (v3 better)
- Log-lik: v3 = -0.18686, v1 = -0.18785, v2 = -0.18789
- ECE: v3 = 0.02018, v1 = 0.01890, v2 = 0.01683
- Sharpness: v3 = 0.08459, v1 = 0.06011, v2 = 0.05910

**Paired bootstrap CIs (block over origins, 500 resamples):**

| Comparison | ΔBrier mean | 95% CI | Significant? |
|------------|-------------|--------|--------------|
| v3 vs v1 | 0.00047 | [-0.00227, 0.00318] | NOT significant |
| v3 vs v2 | 0.00041 | [-0.00230, 0.00312] | NOT significant |

**Permutation test (1000 permutations):**

| Comparison | Observed ΔBrier | Permutation p-value |
|------------|-----------------|---------------------|
| v3 vs v1 | 0.00040 | 0.827 |
| v3 vs v2 | 0.00044 | 0.797 |

#### Config M5.0_7d
- n_origins = 9, n_positive = 3
- Brier: v3 = **0.00509**, v1 = 0.00512, v2 = 0.00513
- ΔBrier (v3−v1) = -0.00003 (v3 better)
- ΔBrier (v3−v2) = -0.00003 (v3 better)
- Log-lik: v3 = -0.02796, v1 = -0.02813, v2 = -0.02832
- ECE: v3 = 0.00152, v1 = 0.00204, v2 = 0.00200
- Sharpness: v3 = 0.00563, v1 = 0.00456, v2 = 0.00428

**Paired bootstrap CIs (block over origins, 500 resamples):**

| Comparison | ΔBrier mean | 95% CI | Significant? |
|------------|-------------|--------|--------------|
| v3 vs v1 | 0.00003 | [-0.00003, 0.00013] | NOT significant |
| v3 vs v2 | 0.00003 | [-0.00003, 0.00013] | NOT significant |

**Permutation test (1000 permutations):**

| Comparison | Observed ΔBrier | Permutation p-value |
|------------|-----------------|---------------------|
| v3 vs v1 | 0.00003 | 0.921 |
| v3 vs v2 | 0.00003 | 0.922 |

#### Config M5.0_30d
- n_origins = 9, n_positive = 6
- Brier: v3 = **0.00976**, v1 = 0.00991, v2 = 0.00991
- ΔBrier (v3−v1) = -0.00015 (v3 better)
- ΔBrier (v3−v2) = -0.00015 (v3 better)
- Log-lik: v3 = -0.04703, v1 = -0.04663, v2 = -0.04715
- ECE: v3 = 0.00886, v1 = 0.00294, v2 = 0.00312
- Sharpness: v3 = 0.02302, v1 = 0.01900, v2 = 0.01781

**Paired bootstrap CIs (block over origins, 500 resamples):**

| Comparison | ΔBrier mean | 95% CI | Significant? |
|------------|-------------|--------|--------------|
| v3 vs v1 | 0.00016 | [-0.00023, 0.00064] | NOT significant |
| v3 vs v2 | 0.00013 | [-0.00027, 0.00057] | NOT significant |

**Permutation test (1000 permutations):**

| Comparison | Observed ΔBrier | Permutation p-value |
|------------|-----------------|---------------------|
| v3 vs v1 | 0.00015 | 0.501 |
| v3 vs v2 | 0.00015 | 0.527 |

## 6. Spatial Holdout (4-fold quadrant)

| Quadrant | n_cells | n_positive | Brier v3 | Brier v1 | Brier v2 | Δ(v3−v1) | Δ(v3−v2) |
|----------|---------|------------|----------|----------|----------|-----------|-----------|
| NW | 16 | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| NE | 16 | 2 | 0.01267 | 0.01279 | 0.01280 | -0.00012 | -0.00013 |
| SW | 16 | 2 | 0.01382 | 0.01384 | 0.01384 | -0.00002 | -0.00002 |
| SE | 16 | 5 | 0.03347 | 0.03342 | 0.03343 | 0.00005 | 0.00005 |

v3 beats v1 in 2/4 quadrants; v3 beats v2 in 2/4 quadrants.

## 7. Sparse-Cell Analysis

Categories based on historical (pre-2020) M≥4.5 event counts per 1° cell.

| Category | n_cells | v1 P mean | v1 width | v2 P mean | v2 width | v3 P mean | v3 width | v3 local bw |
|----------|---------|-----------|----------|-----------|----------|-----------|----------|-------------|
| zero (N=0) | 3 | 0.00000 | -0.00122 | 0.00019 | 0.00099 | 0.00059 | 0.00035 | 3.048 |
| low (1<=N<=4) | 19 | 0.00103 | -0.00306 | 0.00121 | 0.00259 | 0.00151 | 0.00078 | 2.114 |
| moderate (5<=N<=19) | 18 | 0.00412 | -0.00544 | 0.00424 | 0.00490 | 0.00529 | 0.00294 | 1.060 |
| high (N>=20) | 24 | 0.02447 | -0.01192 | 0.02424 | 0.01115 | 0.02955 | 0.01874 | 0.620 |

## 8. Grid Sensitivity

Evaluation at origin 2020-01-01, M≥4.5/7d, with grids 0.5°/1.0°/2.0°.

| Grid | n_cells | n_positive | Brier v1 | Brier v3 | ECE v1 | ECE v3 | Sharpness v1 | Sharpness v3 |
|------|---------|------------|----------|----------|--------|--------|--------------|--------------|
| 0.5deg | 256 | 0 | 0.00003 | 0.00051 | 0.00269 | 0.01175 | 0.00467 | 0.01939 |
| 1.0deg | 64 | 0 | 0.00035 | 0.00073 | 0.01064 | 0.01302 | 0.01551 | 0.02370 |
| 2.0deg | 16 | 0 | 0.00373 | 0.00079 | 0.04127 | 0.01354 | 0.04502 | 0.02465 |

**Brier range across grids:** v1 = 0.00370, v3 = 0.00028. v3 IS more stable than v1 across grid choices.

## 9. Bandwidth Sensitivity (selection-period table)

All variants × candidates evaluated on the SELECTION period (no eval-period info).

| Variant | Kernel | Adaptive | Candidate | Mean Brier (selection) | Best? |
|---------|--------|----------|-----------|-------------------------|-------|
| A_gaussian_fixed | gaussian | False | 0.25 | 0.00571 | ✅ |
| A_gaussian_fixed | gaussian | False | 0.5 | 0.00588 |  |
| A_gaussian_fixed | gaussian | False | 1.0 | 0.00603 |  |
| A_gaussian_fixed | gaussian | False | 2.0 | 0.00614 |  |
| B_gaussian_nn | gaussian | True | 10 | 0.00570 | ✅ |
| B_gaussian_nn | gaussian | True | 25 | 0.00571 |  |
| B_gaussian_nn | gaussian | True | 50 | 0.00574 |  |
| C_epanechnikov_fixed | epanechnikov | False | 0.25 | 0.00576 |  |
| C_epanechnikov_fixed | epanechnikov | False | 0.5 | 0.00568 | ✅ |
| C_epanechnikov_fixed | epanechnikov | False | 1.0 | 0.00586 |  |
| C_epanechnikov_fixed | epanechnikov | False | 2.0 | 0.00603 |  |
| D_epanechnikov_nn | epanechnikov | True | 10 | 0.00578 |  |
| D_epanechnikov_nn | epanechnikov | True | 25 | 0.00570 |  |
| D_epanechnikov_nn | epanechnikov | True | 50 | 0.00567 | ✅ |

## 10. Posterior Predictive Check

- Observed total events: **1890**
- Simulated total (mean): **2364.8** (95% CI: [2268, 2457])
- Observed occupied cells: **61**
- Simulated occupied (mean): **61.3** (95% CI: [58, 64])
- Observed max count: **217**
- Simulated max (mean): **396.8** (95% CI: [362, 437])
- Observed Gini: **0.6493**
- Simulated Gini (mean): **0.6835** (95% CI: [0.6673, 0.6991])
- Observed top-3 fraction: **0.3101**
- Simulated top-3 fraction (mean): **0.4128** (95% CI: [0.3929, 0.4339])

Posterior predictive check: total=FAIL, Gini=FAIL

## 11. Mc Sensitivity

| Mc | n_hist | Regional rate | P(7d) | Mean local bw (deg) |
|----|--------|---------------|-------|---------------------|
| Mc3.8 | 1695 | 44.7715 | 0.576009 | 1.3011 |
| Mc4.0 | 1695 | 44.7715 | 0.576009 | 1.3011 |
| Mc4.13 | 1695 | 44.7715 | 0.576009 | 1.3011 |
| Mc4.5 | 1695 | 44.7715 | 0.576009 | 1.3011 |

## 12. Answers to the 10 Required Questions

**1. Does adaptive smoothing improve Brier score?** Mean v3 = 0.01985, v1 = 0.02001, v2 = 0.02002. Δ(v3−v1) = -0.00015 (v3 better), Δ(v3−v2) = -0.00016 (v3 better). Bootstrap CI excludes zero in favour of v3 in 0/4 configs vs v1 and 0/4 vs v2. **Answer: NO — no statistically defensible Brier improvement.**

**2. Does it improve log score?** Mean v3 = -0.08397, v1 = -0.08462, v2 = -0.08477. **Answer: YES** (higher is better).

**3. Does it improve calibration?** Mean ECE v3 = 0.00829, v1 = 0.00722, v2 = 0.00673 (lower is better). **Answer: NO**

**4. Does it improve uncertainty?** v3 provides full bootstrap-derived epistemic intervals on the smoothed rate field (200 resamples per origin). The intervals are wider in sparse cells and narrower in dense cells — matching the local data density. **Answer: Qualitatively YES in the sense of providing density-aware epistemic intervals; quantitatively similar to v2 which also provides full posteriors. v1 only provides analytic Garwood CIs.**

**5. Does it improve spatial holdout performance?** v3 beats v1 in 2/4 quadrants; v3 beats v2 in 2/4 quadrants. **Answer: NO — not consistently across all quadrants.**

**6. Does it reduce grid sensitivity?** Brier range across 0.5°/1.0°/2.0° grids: v1 = 0.00370, v3 = 0.00028. **Answer: YES**

**7. Does it improve sparse-cell behaviour?** See Section 7. v3 assigns non-zero probabilities to zero-event cells (smoothing leaks rate from neighbours). v1 also assigns non-zero via Jeffreys upper bound; v2 via hierarchical shrinkage. v3's local bandwidth adapts: broad in sparse regions, narrow in dense ones. **Answer: Qualitatively YES in the sense of continuous smoothing vs grid-cell discretisation; quantitatively the improvement in Brier is within noise (see bootstrap CIs).**

**8. Is the improvement statistically significant?** Bootstrap CIs exclude zero in favour of v3 in 0/4 configs vs v1 and 0/4 vs v2. **Answer: NO — CIs include zero in most/all configs.**

**9. Is the improvement scientifically meaningful?** Mean ΔBrier (v3−v1) = -0.00015 on a base Brier ~0.02001. This is a relative change of 0.76%. For rare-event forecasting with base Brier near the climatology baseline, changes < 5% relative are generally not scientifically meaningful even if statistically detectable. **Answer: NO — changes are within the noise band of the climatology baseline.**

**10. Should v3 proceed to prospective testing?** Posterior predictive check: total=FAIL. Decision based on the formal verdict in Section 13. **Answer: See Section 13 for the formal decision.**

## 13. Final Verdict

Decision criteria (predefined before inspecting results):

- **A. SUPERIOR — prospective candidate:** v3 must beat BOTH v1 and v2 on mean Brier with bootstrap CIs excluding zero in favour of v3 in ≥2/4 configs against each, AND improve spatial holdout in ≥3/4 quadrants vs each, AND pass posterior predictive check.
- **B. EQUIVALENT — uncertainty/calibration improvement:** v3 predictive skill statistically equivalent to v1/v2 (CIs include zero), BUT demonstrably better uncertainty quantification (e.g. density-aware intervals, better sparse-cell behaviour) and posterior predictive check passes.
- **C. EQUIVALENT — no meaningful advantage:** v3 ≈ v1/v2 on all metrics; no statistically significant improvement and no material uncertainty gain.
- **D. WORSE — reject:** v3 significantly worse than v1 or v2 (CI excludes zero against v3), or posterior predictive check fails.

### Verdict: **D. WORSE — reject**

- Mean ΔBrier (v3−v1) = -0.00015; bootstrap CIs exclude zero in favour of v3 in 0/4 configs vs v1, 0/4 vs v2.
- Mean ΔECE (v3−v1) = 0.00107; (v3−v2) = 0.00156.
- Spatial holdout: v3 beats v1 in 2/4, v2 in 2/4.
- Posterior predictive check: FAIL.
- Integrity audit: PASS.

**Prospective deployment decision:** NO — do not deploy v3 prospectively.

FINAL_v1.0_FROZEN remains PRODUCTION. FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL remains unchanged. This v3 candidate is labeled **FINAL_v3.0_CANDIDATE_ADAPTIVE_SPATIAL** and does NOT replace any existing model.

<!--VERDICT:D-->

## 14. Integrity Audit

| Check | Status |
|-------|--------|
| FINAL_v1.0_FROZEN source code unchanged | ✅ PASS |
| FINAL_v2.0 candidate source code unchanged | ✅ PASS |
| v1 forecast ledger unchanged | ✅ PASS |
| v2 forecast ledger unchanged | ✅ PASS |
| Existing prospective scoring unchanged | ✅ PASS |
| No evaluation-period leakage (selection only on 2010-2014) | ✅ PASS |
| No forecast rewriting | ✅ PASS |
| No cherry-picking (predefined splits, predefined candidates) | ✅ PASS |
| No post-hoc threshold selection | ✅ PASS |
| No fabricated data | ✅ PASS |
| No fabricated performance | ✅ PASS |
| No deterministic earthquake predictions | ✅ PASS |

All v3 artifacts are written to a SEPARATE namespace (`v3_candidates/adaptive_spatial/` and `outputs/v3_adaptive_*` / `outputs/V3_ADAPTIVE_SPATIAL_REPORT.md`). No v1 or v2 file was modified, overwritten, or deleted.


## 15. Reproducibility

- Source: `v3_candidates/adaptive_spatial/model.py`
- Runner: `run_v3_experiment.py`
- Random seed: 42 (bootstrap), 42/43 (paired), 44/45 (permutation)
- Catalog snapshot: USGS+ISC merged (same as v1/v2)
- Splits: dev (<2010), select (2010-2014), eval (2015-2023)
- No data from the evaluation period was used for bandwidth, kernel, or model selection.
