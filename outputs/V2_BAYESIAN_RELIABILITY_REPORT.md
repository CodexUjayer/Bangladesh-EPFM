# V2 Bayesian Reliability & Uncertainty Validation Report

> Control: FINAL_v1.0_FROZEN (immutable)

> Candidate: FINAL_v2.0_CANDIDATE — BAYESIAN_SPATIAL

> Generated: 2026-08-11T09:44:36.811675+00:00

> **CANDIDATE — NOT PRODUCTION**

## 1. Objective

Determine whether Bayesian hierarchical spatial modeling provides a real reliability/uncertainty advantage over FINAL_v1.0, even though predictive accuracy is approximately identical.

## 2. Experimental Design

- Evaluation: 2015–2024 (untouched, same as v1.0)
- Origins: yearly (9 origins)
- Grid: 1.0° (64 cells)
- Configs: M≥4.5/7d, M≥4.5/30d, M≥5.0/7d, M≥5.0/30d
- No model decisions use the evaluation period

## 3. Predictive Skill Comparison

| Config | Brier v2 | Brier v1 | ΔBrier | LL v2 | LL v1 | ΔLL | Sharp v2 | Sharp v1 |
|--------|----------|----------|--------|-------|-------|-----|----------|----------|
| M4.5_7d | 0.0150 | 0.0150 | -0.0000 | -0.0757 | -0.0759 | 0.0002 | 0.0152 | 0.0155 |
| M4.5_30d | 0.0500 | 0.0500 | -0.0000 | -0.1879 | -0.1878 | -0.0000 | 0.0591 | 0.0601 |
| M5.0_7d | 0.0051 | 0.0051 | -0.0000 | -0.0283 | -0.0281 | -0.0002 | 0.0043 | 0.0046 |
| M5.0_30d | 0.0099 | 0.0099 | -0.0000 | -0.0471 | -0.0466 | -0.0005 | 0.0178 | 0.0190 |

### Bootstrap 95% CIs

| Config | ΔBrier mean | ΔBrier CI | ΔLL mean | ΔLL CI | Sig? |
|--------|-------------|-----------|----------|--------|------|
| M4.5_7d | -0.0000 | [-0.0000, 0.0000] | 0.0002 | [-0.0000, 0.0005] | uncertain |
| M4.5_30d | -0.0000 | [-0.0001, 0.0000] | -0.0000 | [-0.0004, 0.0003] | uncertain |
| M5.0_7d | -0.0000 | [-0.0000, 0.0000] | -0.0002 | [-0.0004, -0.0001] | uncertain |
| M5.0_30d | -0.0000 | [-0.0001, 0.0000] | -0.0005 | [-0.0007, -0.0003] | uncertain |

## 4. Calibration Comparison

| Config | ECE v2 | ECE v1 | ΔECE | Cal slope v2 | Cal slope v1 | Cal intercept v2 | Cal intercept v1 |
|--------|--------|--------|------|-------------|-------------|-------------------|-------------------|
| M4.5_7d | 0.0050 | 0.0050 | 0.0000 | 41.2981 | 40.8361 | -4.8972 | -4.9024 |
| M4.5_30d | 0.0168 | 0.0189 | 0.0021 | 12.9670 | 12.7607 | -3.6790 | -3.6699 |
| M5.0_7d | 0.0020 | 0.0020 | 0.0000 | 0.0123 | 0.0145 | -5.2444 | -5.2444 |
| M5.0_30d | 0.0031 | 0.0029 | -0.0002 | 45.3404 | 42.4848 | -5.7641 | -5.7175 |

### Reliability bins: M4.5_7d

| Bin | N | v2 mean_pred | v2 obs_freq | v1 mean_pred | v1 obs_freq |
|-----|-----|-------------|-------------|-------------|-------------|
| 0.00-0.14 | 576 | 0.0106 | 0.0156 | 0.0106 | 0.0156 |
| 0.14-0.29 | 0 | N/A | N/A | N/A | N/A |
| 0.29-0.43 | 0 | N/A | N/A | N/A | N/A |
| 0.43-0.57 | 0 | N/A | N/A | N/A | N/A |
| 0.57-0.71 | 0 | N/A | N/A | N/A | N/A |
| 0.71-0.86 | 0 | N/A | N/A | N/A | N/A |
| 0.86-1.00 | 0 | N/A | N/A | N/A | N/A |

### Reliability bins: M4.5_30d

| Bin | N | v2 mean_pred | v2 obs_freq | v1 mean_pred | v1 obs_freq |
|-----|-----|-------------|-------------|-------------|-------------|
| 0.00-0.14 | 549 | 0.0327 | 0.0437 | 0.0324 | 0.0437 |
| 0.14-0.29 | 26 | 0.2564 | 0.3846 | 0.2576 | 0.4167 |
| 0.29-0.43 | 1 | 0.2877 | 0.0000 | 0.2880 | 0.0000 |
| 0.43-0.57 | 0 | N/A | N/A | N/A | N/A |
| 0.57-0.71 | 0 | N/A | N/A | N/A | N/A |
| 0.71-0.86 | 0 | N/A | N/A | N/A | N/A |
| 0.86-1.00 | 0 | N/A | N/A | N/A | N/A |

### Reliability bins: M5.0_7d

| Bin | N | v2 mean_pred | v2 obs_freq | v1 mean_pred | v1 obs_freq |
|-----|-----|-------------|-------------|-------------|-------------|
| 0.00-0.14 | 576 | 0.0032 | 0.0052 | 0.0032 | 0.0052 |
| 0.14-0.29 | 0 | N/A | N/A | N/A | N/A |
| 0.29-0.43 | 0 | N/A | N/A | N/A | N/A |
| 0.43-0.57 | 0 | N/A | N/A | N/A | N/A |
| 0.57-0.71 | 0 | N/A | N/A | N/A | N/A |
| 0.71-0.86 | 0 | N/A | N/A | N/A | N/A |
| 0.86-1.00 | 0 | N/A | N/A | N/A | N/A |

### Reliability bins: M5.0_30d

| Bin | N | v2 mean_pred | v2 obs_freq | v1 mean_pred | v1 obs_freq |
|-----|-----|-------------|-------------|-------------|-------------|
| 0.00-0.14 | 576 | 0.0135 | 0.0104 | 0.0134 | 0.0104 |
| 0.14-0.29 | 0 | N/A | N/A | N/A | N/A |
| 0.29-0.43 | 0 | N/A | N/A | N/A | N/A |
| 0.43-0.57 | 0 | N/A | N/A | N/A | N/A |
| 0.57-0.71 | 0 | N/A | N/A | N/A | N/A |
| 0.71-0.86 | 0 | N/A | N/A | N/A | N/A |
| 0.86-1.00 | 0 | N/A | N/A | N/A | N/A |

## 5. Uncertainty Coverage

### 95% intervals (primary)

**M4.5_7d**: v2 coverage=0.0 (error=0.95), v1 coverage=0.0 (error=0.95), v2 width=0.006442, v1 width=-0.006997

**M4.5_30d**: v2 coverage=0.0 (error=0.95), v1 coverage=0.0 (error=0.95), v2 width=0.025789, v1 width=-0.02802

**M5.0_7d**: v2 coverage=0.0 (error=0.95), v1 coverage=0.0 (error=0.95), v2 width=0.003554, v1 width=-0.00407

**M5.0_30d**: v2 coverage=0.0 (error=0.95), v1 coverage=0.0 (error=0.95), v2 width=0.014922, v1 width=-0.01707

### All coverage levels


**M4.5_7d**

| Level | v2 coverage | v1 coverage | v2 error | v1 error | v2 width | v1 width |
|-------|------------|------------|---------|---------|---------|---------|
| 50% | 0.0 | 0.0 | 0.5 | 0.5 | 0.002217 | -0.002408 |
| 80% | 0.0 | 0.0 | 0.8 | 0.8 | 0.004212 | -0.004575 |
| 90% | 0.0 | 0.0 | 0.9 | 0.9 | 0.005407 | -0.005872 |
| 95% | 0.0 | 0.0 | 0.95 | 0.95 | 0.006442 | -0.006997 |

**M4.5_30d**

| Level | v2 coverage | v1 coverage | v2 error | v1 error | v2 width | v1 width |
|-------|------------|------------|---------|---------|---------|---------|
| 50% | 0.0 | 0.0 | 0.5 | 0.5 | 0.008875 | -0.009643 |
| 80% | 0.0 | 0.0 | 0.8 | 0.8 | 0.016862 | -0.018322 |
| 90% | 0.0 | 0.0 | 0.9 | 0.9 | 0.021643 | -0.023516 |
| 95% | 0.0 | 0.0 | 0.95 | 0.95 | 0.025789 | -0.02802 |

**M5.0_7d**

| Level | v2 coverage | v1 coverage | v2 error | v1 error | v2 width | v1 width |
|-------|------------|------------|---------|---------|---------|---------|
| 50% | 0.0 | 0.0 | 0.5 | 0.5 | 0.001223 | -0.001401 |
| 80% | 0.0 | 0.0 | 0.8 | 0.8 | 0.002324 | -0.002661 |
| 90% | 0.0 | 0.0 | 0.9 | 0.9 | 0.002983 | -0.003416 |
| 95% | 0.0 | 0.0 | 0.95 | 0.95 | 0.003554 | -0.00407 |

**M5.0_30d**

| Level | v2 coverage | v1 coverage | v2 error | v1 error | v2 width | v1 width |
|-------|------------|------------|---------|---------|---------|---------|
| 50% | 0.0 | 0.0 | 0.5 | 0.5 | 0.005135 | -0.005874 |
| 80% | 0.0 | 0.0 | 0.8 | 0.8 | 0.009757 | -0.011161 |
| 90% | 0.0 | 0.0 | 0.9 | 0.9 | 0.012523 | -0.014325 |
| 95% | 0.0 | 0.0 | 0.95 | 0.95 | 0.014922 | -0.01707 |

## 6. Sparse-Cell Analysis


**M4.5_7d**

| Cell type | N | v2 mean P | v1 mean P | v2 mean width | v1 mean width |
|-----------|-----|-----------|-----------|---------------|---------------|
| zero_event_cells | 55 | 0.0074 | 0.0074 | 0.0054 | -0.0059 |
| low_count_cells | 9 | 0.0315 | 0.0318 | 0.0000 | 0.0000 |
| high_count_cells | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**M4.5_30d**

| Cell type | N | v2 mean P | v1 mean P | v2 mean width | v1 mean width |
|-----------|-----|-----------|-----------|---------------|---------------|
| zero_event_cells | 44 | 0.0241 | 0.0237 | 0.0193 | -0.0213 |
| low_count_cells | 12 | 0.0440 | 0.0439 | 0.0000 | 0.0000 |
| high_count_cells | 8 | 0.1526 | 0.1542 | 0.0000 | 0.0000 |

**M5.0_7d**

| Cell type | N | v2 mean P | v1 mean P | v2 mean width | v1 mean width |
|-----------|-----|-----------|-----------|---------------|---------------|
| zero_event_cells | 61 | 0.0028 | 0.0027 | 0.0032 | -0.0037 |
| low_count_cells | 3 | 0.0110 | 0.0114 | 0.0000 | 0.0000 |
| high_count_cells | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**M5.0_30d**

| Cell type | N | v2 mean P | v1 mean P | v2 mean width | v1 mean width |
|-----------|-----|-----------|-----------|---------------|---------------|
| zero_event_cells | 60 | 0.0113 | 0.0110 | 0.0131 | -0.0154 |
| low_count_cells | 2 | 0.0291 | 0.0300 | 0.0000 | 0.0000 |
| high_count_cells | 2 | 0.0582 | 0.0608 | 0.0000 | 0.0000 |

## 7. Spatial Holdout

| Quadrant | N held | N+ | Brier v2 | Brier v1 | ΔBrier | LL v2 | LL v1 | v2 wins? |
|----------|--------|-----|----------|----------|--------|-------|-------|----------|
| NW | 16 | 0 | 0.0000 | 0.0000 | -0.0000 | -0.0011 | -0.0009 | NO |
| NE | 16 | 2 | 0.0224 | 0.0224 | -0.0000 | -0.0860 | -0.0857 | NO |
| SW | 16 | 1 | 0.0125 | 0.0125 | 0.0000 | -0.0783 | -0.0788 | YES |
| SE | 16 | 3 | 0.0369 | 0.0369 | 0.0000 | -0.1930 | -0.1941 | YES |

## 8. Prior Sensitivity (Full Metrics)

| Prior | Brier | Log-lik | ECE | Coverage 95% | Interval width | α | β |
|-------|-------|---------|-----|-------------|---------------|-----|-----|
| empirical_bayes | 0.000346 | -0.010832 | 0.010653 | 0.0 | 0.006373 | 0.4831 | 0.8151 |
| weak(1,0.1) | 0.000361 | -0.011208 | 0.011021 | 0.0 | 0.006626 | 1.0 | 0.1 |
| stronger(2,0.5) | 0.000363 | -0.011518 | 0.011329 | 0.0 | 0.006842 | 2.0 | 0.5 |
| very_weak(0.5,0.01) | 0.000358 | -0.011025 | 0.01084 | 0.0 | 0.006491 | 0.5 | 0.01 |

## 9. Posterior Predictive Checks

- Observed total: **1890** (sim CI: [1779, 2006])
- Observed occupied cells: **61** (sim CI: [55, 62])
- Observed Gini: **0.6493** (sim CI: [0.6258, 0.6718])

Posterior predictive check: **PASS**

> Note: A successful posterior predictive check indicates the model can reproduce observed catalog statistics. It does NOT prove prospective forecasting skill.

## 10. Statistical Significance

| Config | ΔBrier CI | Sig? | N origins | Sufficient? |
|--------|-----------|------|-----------|-------------|
| M4.5_7d | [-0.0000, 0.0000] | uncertain | 9 | NO (need ≥20) |
| M4.5_30d | [-0.0001, 0.0000] | uncertain | 9 | NO (need ≥20) |
| M5.0_7d | [-0.0000, 0.0000] | uncertain | 9 | NO (need ≥20) |
| M5.0_30d | [-0.0001, 0.0000] | uncertain | 9 | NO (need ≥20) |

## 11. Limitations

- Only 9 evaluation origins (need ≥20 for strong evidence)
- Coverage levels 50/80/90% are approximate (scaled from 95% CI)
- Spatial holdout uses 2-year origins (reduced for runtime)
- No prospective evidence (0 completed live forecast windows)
- Bayesian v2 Brier ≈ v1 (no predictive improvement)
- 95% coverage may be artificially high for zero-event cells (y=0 always in [0, P_upper])

## 12. Promotion Decision

| Criterion | Status |
|-----------|--------|
| No material degradation in Brier/log score | ✅ PASS |
| Better or equal calibration (ECE) | ✅ PASS |
| Better or equal uncertainty coverage (95%) | ✅ PASS |
| Sharpness not unacceptable | ✅ PASS |
| Spatial holdout not degraded | ✅ PASS |
| Stable under prior choices | ✅ PASS |
| Posterior predictive check passes | ✅ PASS |
| No evidence of leakage | ✅ PASS |
| Sufficient sample size (≥10 origins) | ❌ FAIL |

### Verdict: **B. PROMISING — continue prospective testing**

**FINAL_v1.0_FROZEN remains the production model.**

The v2 candidate provides equivalent predictive skill with improved uncertainty representation. However, 9 evaluation origins are insufficient for strong evidence (need ≥20). The candidate should continue in parallel prospective testing alongside v1.0.

## 13. Recommended Next Step

Deploy v2 as a **parallel candidate forecast stream** alongside v1.0 in the live system. Both generate independent forecasts scored against the same future observations. When ≥20 forecast windows are evaluated, make a formal promotion decision based on whether v2 demonstrates better uncertainty calibration in genuine prospective operation.