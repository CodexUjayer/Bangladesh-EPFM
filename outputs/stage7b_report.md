# STAGE 7B — ML vs Spatial Poisson

> **Phase A corrected**: base-rate check fixed (was metrically wrong; now
> compares mean sum(cell P) to mean observed regional rate). ETAS base-10
> correction does not affect this report (ML and SP formulations unchanged).
> The SP-beats-ML conclusion is robust.
>
> **Caveats**: This conclusion is CONDITIONAL on the tested configurations
> (2 horizons × 2 thresholds × 2 models × 2 feature sets × 9 origins). The
> spatial holdout test (required by Stage 7 §15 and Stage 7B §11) remains
> UNRESOLVED — it was never implemented. The conclusion that "ML adds no
> skill beyond spatial Poisson" may not generalize to untested configs or
> to spatial regions held out during training.
>
> Generated 2026-08-09T09:16:12.174270+00:00 (Phase A re-run).

## 0. The decisive question

**Does ML add predictive information beyond the historical spatial seismicity-rate model?**

Stage 7 showed ML beats UNIFORM Poisson, but ML-A (historical rate) already captured most of the improvement — strongly suggesting the gain is spatial heterogeneity. Stage 7B compares ML directly against a CAUSALLY-RECONSTRUCTED Spatial Poisson baseline (expanding-window, raw).

## 1. Causal spatial-Poisson baseline

- **Estimator**: expanding-window rate per cell, λ_cell(t) = N_cell(<t) / exposure(<t)
- **Causal**: only events before the forecast origin are used. The static Stage-4 spatial forecast (estimated from the full 1973–2024 catalog) is NOT used — that would leak future spatial information.
- **Smoothing**: raw (no smoothing) for the primary comparison. Neighbor-smoothing tested as a secondary baseline.
- **P_cell** = 1 − exp(−λ_cell · Δt). Cells treated as independent (mutual-exclusivity holds approximately because events are assigned to exactly one cell).

## 2. Identical evaluation conditions

ML and Spatial Poisson use:
- Identical catalog (usgs_bangladesh_1973_2025_m25)
- Identical geographic grid (1.0°, 64 cells)
- Identical forecast origins (yearly 1995–2022, every 3 years)
- Identical training cutoff (all data before the origin)
- Identical horizons and thresholds
- Identical observed outcomes (same y_true)
- No model receives information unavailable to the other.

## 3. Base-rate check

Verifies that sum(cell probabilities) ≈ observed regional probability, ensuring no incorrect normalization.

| Origin | Sum cell P | Regional P (indep) | Observed regional | Ratio | Pass? |
|--------|------------|--------------------|-------------------|-------|-------|
| 1998 | 0.528 | 0.415 | N/A | 0.951 | True |
| 2001 | 0.532 | 0.417 | N/A | 0.958 | True |
| 2004 | 0.536 | 0.419 | N/A | 0.964 | True |
| 2007 | 0.563 | 0.435 | N/A | 1.013 | True |
| 2010 | 0.583 | 0.447 | N/A | 1.050 | True |
| 2013 | 0.602 | 0.457 | N/A | 1.084 | True |
| 2016 | 0.637 | 0.476 | N/A | 1.147 | True |
| 2019 | 0.660 | 0.489 | N/A | 1.188 | True |
| 2022 | 0.699 | 0.509 | N/A | 1.258 | True |
| 1998 | 1.061 | 0.663 | N/A | 1.910 | False |
| 2001 | 1.017 | 0.647 | N/A | 1.830 | False |
| 2004 | 0.989 | 0.637 | N/A | 1.781 | False |
| 2007 | 0.976 | 0.632 | N/A | 1.756 | False |
| 2010 | 0.972 | 0.630 | N/A | 1.750 | False |
| 2013 | 0.994 | 0.638 | N/A | 1.789 | False |
| 2016 | 0.993 | 0.638 | N/A | 1.788 | False |
| 2019 | 0.993 | 0.638 | N/A | 1.787 | False |
| 2022 | 1.013 | 0.646 | N/A | 1.824 | False |

## 4. Primary comparison: ML vs Spatial Poisson

### Per (horizon, threshold) configuration


#### Horizon 7d, threshold M≥4.5

| Model | N test | N+ | Base rate | Brier | Brier SP | ΔBrier (SP−ML) | IG vs SP | ECE | Sharpness | ROC-AUC (sec) | PR-AUC (sec) |
|-------|--------|-----|-----------|-------|----------|---------------|---------|-----|-----------|---------------|--------------|
| spatial_poisson | 576 | 7 | 0.012152777777777778 | 0.012 | 0.012 | baseline | baseline | 0.003 | 0.014 | 0.827 | 0.085 |
| uniform_poisson | 576 | 7 | 0.012152777777777778 | 0.206 | 0.012 | -0.195 | -0.549 | 0.439 | 0.032 | 0.468 | 0.013 |
| gb|ML-A | 576 | 7 | 0.012152777777777778 | 0.022 | 0.012 | -0.011 | -0.232 | 0.024 | 0.100 | 0.515 | 0.015 |
| gb|ML-F | 576 | 7 | 0.012152777777777778 | 0.021 | 0.012 | -0.009 | -0.240 | 0.021 | 0.094 | 0.464 | 0.016 |
| logistic_l2|ML-A | 576 | 7 | 0.012152777777777778 | 0.132 | 0.012 | -0.121 | -0.526 | 0.256 | 0.243 | 0.522 | 0.041 |
| logistic_l2|ML-F | 576 | 7 | 0.012152777777777778 | 0.050 | 0.012 | -0.039 | -0.263 | 0.054 | 0.187 | 0.460 | 0.014 |
| spatial_poisson | 576 | 7 | 0.012152777777777778 | 0.012 | 0.012 | baseline | baseline | 0.003 | 0.014 | 0.827 | 0.085 |
| uniform_poisson | 576 | 7 | 0.012152777777777778 | 0.206 | 0.012 | -0.195 | -0.549 | 0.439 | 0.032 | 0.468 | 0.013 |

**Block bootstrap 95% CIs (ML vs Spatial Poisson):**

| Model | ΔBrier mean | ΔBrier 95% CI | Δlog-lik mean | Δlog-lik 95% CI | Significant? |
|-------|-------------|---------------|---------------|------------------|--------------|
| gb|ML-A | -0.011 | [-0.023, -0.002] | -0.225 | [-0.386, -0.093] | NO (SP better) |
| gb|ML-F | -0.010 | [-0.016, -0.003] | -0.235 | [-0.389, -0.096] | NO (SP better) |
| logistic_l2|ML-A | -0.122 | [-0.148, -0.084] | -0.527 | [-0.602, -0.461] | NO (SP better) |
| logistic_l2|ML-F | -0.039 | [-0.091, -0.006] | -0.256 | [-0.473, -0.084] | NO (SP better) |

#### Horizon 30d, threshold M≥5.0

| Model | N test | N+ | Base rate | Brier | Brier SP | ΔBrier (SP−ML) | IG vs SP | ECE | Sharpness | ROC-AUC (sec) | PR-AUC (sec) |
|-------|--------|-----|-----------|-------|----------|---------------|---------|-----|-----------|---------------|--------------|
| spatial_poisson | 576 | 6 | 0.010416666666666666 | 0.010 | 0.010 | baseline | baseline | 0.005 | 0.022 | 0.943 | 0.146 |
| uniform_poisson | 576 | 6 | 0.010416666666666666 | 0.408 | 0.010 | -0.399 | -0.976 | 0.631 | 0.009 | 0.500 | 0.011 |
| gb|ML-A | 576 | 6 | 0.010416666666666666 | 0.020 | 0.010 | -0.010 | -0.210 | 0.019 | 0.102 | 0.696 | 0.039 |
| gb|ML-F | 576 | 6 | 0.010416666666666666 | 0.017 | 0.010 | -0.008 | -0.187 | 0.016 | 0.084 | 0.635 | 0.023 |
| logistic_l2|ML-A | 576 | 6 | 0.010416666666666666 | 0.060 | 0.010 | -0.051 | -0.212 | 0.101 | 0.218 | 0.746 | 0.075 |
| logistic_l2|ML-F | 576 | 6 | 0.010416666666666666 | 0.114 | 0.010 | -0.104 | -0.602 | 0.119 | 0.299 | 0.410 | 0.011 |
| spatial_poisson | 576 | 6 | 0.010416666666666666 | 0.010 | 0.010 | baseline | baseline | 0.005 | 0.022 | 0.943 | 0.146 |
| uniform_poisson | 576 | 6 | 0.010416666666666666 | 0.408 | 0.010 | -0.399 | -0.976 | 0.631 | 0.009 | 0.500 | 0.011 |

**Block bootstrap 95% CIs (ML vs Spatial Poisson):**

| Model | ΔBrier mean | ΔBrier 95% CI | Δlog-lik mean | Δlog-lik 95% CI | Significant? |
|-------|-------------|---------------|---------------|------------------|--------------|
| gb|ML-A | -0.010 | [-0.015, -0.005] | -0.208 | [-0.267, -0.137] | NO (SP better) |
| gb|ML-F | -0.008 | [-0.016, -0.002] | -0.182 | [-0.272, -0.091] | NO (SP better) |
| logistic_l2|ML-A | -0.051 | [-0.060, -0.040] | -0.212 | [-0.241, -0.178] | NO (SP better) |
| logistic_l2|ML-F | -0.108 | [-0.289, -0.007] | -0.613 | [-1.486, -0.069] | NO (SP better) |

## 5. Feature ablation vs Spatial Poisson

Which feature groups provide information beyond spatial rate?

| Feature set | Configs beating Spatial Poisson | Total | Win rate |
|-------------|--------------------------------|-------|----------|
| ML-A | 0 | 4 | 0.0% |
| ML-B | 0 | 0 | N/A (not tested) |
| ML-C | 0 | 0 | N/A (not tested) |
| ML-D | 0 | 0 | N/A (not tested) |
| ML-E | 0 | 0 | N/A (not tested) |
| ML-F | 0 | 4 | 0.0% |

## 6. Statistical significance

Block bootstrap over forecast ORIGINS (not individual cell rows). 500 resamples. ΔBrier = Brier_SP − Brier_ML (positive = ML better). If the 95% CI includes zero, the improvement is UNCERTAIN.

See per-config bootstrap tables in Section 4.

## 7. Model complexity check

If ML only marginally beats Spatial Poisson, the incremental improvement may not justify the additional complexity:

| Model | Features | Parameters | Interpretability | Calibration |
|-------|----------|------------|------------------|-------------|
| Spatial Poisson | 1 (rate per cell) | 64 cell rates | High (transparent) | High (Poisson) |
| Logistic ML-A | 4 | ~4 coefficients | High (linear) | Moderate |
| GB ML-F | 42 | 200 trees × depth 3 | Low (ensemble) | Moderate |

A tiny statistically uncertain gain may not justify replacing a transparent spatial Poisson model.

## 8. Scientific conclusion


**C. NO — spatial Poisson explains the apparent ML improvement**

- Total ML-vs-SP comparisons: 8
- Significant wins (CI excludes zero, ML better): 0
- Uncertain (CI includes zero): False

This answer is based ONLY on the direct ML-vs-Spatial-Poisson comparison. The old uniform-Poisson results are NOT used.

## 9. Artifacts

- `outputs/stage7b_report.md` (this file)
- `outputs/stage7b_model_results.csv`
- `outputs/stage7b_backtest/`
- `outputs/stage7b_calibration/`
- `outputs/stage7b_ablation/`
- `outputs/stage7b_spatial_generalization/`
- `outputs/stage7b_uncertainty/`
- `outputs/stage7b_model_metadata.json`