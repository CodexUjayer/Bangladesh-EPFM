# STAGE 7 — Machine Learning Forecasting

> **⚠️ SUPERSEDED — NOT A VALID TEST OF INCREMENTAL SKILL OVER SPATIAL POISSON**
>
> This report compared ML against UNIFORM Poisson, which assigns the same
> regional probability to every cell. ML "beat" this baseline largely by
> learning spatial heterogeneity (most cells have no events). The Spatial
> Poisson baseline (Stage 4) also captures this heterogeneity.
>
> Stage 7B performed the scientifically correct comparison (ML vs causally-
> reconstructed Spatial Poisson) and found that **Spatial Poisson beats every
> ML model** with statistical significance. The Stage 7 conclusion "A. YES"
> is an artifact of the wrong baseline and is **NOT valid**.
>
> This report is preserved for reproducibility only. See `stage7b_report.md`
> for the valid comparison.
>
> Generated 2026-08-09T08:48:53.646856+00:00.

## 0. Primary question

**Can ML produce better calibrated, genuinely out-of-sample probabilistic earthquake forecasts than the corrected Poisson baseline?**

The answer must be one of: A (YES, robust), B (PARTIAL), C (NO), D (INCONCLUSIVE). ML is NOT treated as successful merely because it has high classification accuracy. Calibration and prospective probabilistic skill are primary.

## 1. Hierarchy of baselines (locked from Stages 4-6)

1. **Stationary/expanding-window Poisson** = PRIMARY VALIDATED BASELINE
2. Spatial Poisson = secondary baseline
3. Locally fitted ETAS (K≈0) = diagnostic only; no demonstrated improvement
4. Externally informed ETAS = sensitivity only; no demonstrated improvement
5. Coulomb = DISABLED (no validated receiver-fault geometry)
6. **ML = now tested** against the validated baselines

## 2. Forecast target and grid

- P(N_cell,Δt ≥ 1 | information available at forecast origin)
- Grid: 1.0 deg, 64 cells
- Horizons: 24h, 7d, 30d, 90d, 1y
- Magnitude thresholds: M≥4.5, M≥5.0, M≥5.5, M≥6.0 (M≥6.5/7.0 = research/exploratory)
- Mc scenarios: 4.0, 4.5, 5.0 (sensitivity, NOT validated)

## 3. No-leakage controls

- Every feature at forecast origin t uses ONLY events with origin_time < t.
- **Spatiotemporal leakage control**: all cells from one forecast origin stay in the same temporal split. The model NEVER sees neighboring future cells from the same timestamp during training.
- No random K-fold. Strictly chronological expanding-window evaluation.
- Training rows come from PRIOR origins only; current origin's cells are test-only.
- Documented train/test boundaries: training = all origins before t; test = origin t.

## 4. Feature engineering (causal)

- 43 features total, organized into 6 groups + 1 disabled (Coulomb).
- **ML-G (Coulomb) is DISABLED** per Stage 6 data limitation. `dcfs_cumulative_Pa` = 0.0 for all cells.

| Group | Features |
|-------|----------|
| ML-A | 4 features (hist_rate) |
| ML-B | 14 features (hist_rate, temporal) |
| ML-C | 23 features (hist_rate, temporal, magnitude) |
| ML-D | 30 features (hist_rate, temporal, magnitude, spatial) |
| ML-E | 36 features (hist_rate, temporal, magnitude, spatial, depth) |
| ML-F | 42 features (hist_rate, temporal, magnitude, spatial, depth, clustering) |
| ML-G | 43 features (hist_rate, temporal, magnitude, spatial, depth, clustering, coulomb) |

## 5. Model ladder

- Model 0: Poisson baseline (analytic)
- Model 1: L2-regularized logistic regression (class_weight='balanced')
- Model 2: Elastic Net logistic regression (saga solver)
- Model 3: Random Forest (200 trees, max_depth=8, class_weight='balanced')
- Model 4: Gradient Boosting (200 trees, max_depth=3, lr=0.1, balanced sample weights)
- Model 5: Calibrated Gradient Boosting (isotonic, 3-fold internal CV)
- Model 6: Neural (TCN/LSTM/Transformer) — NOT implemented; insufficient temporal structure to justify deep learning over the simpler models.

## 6. Calibration (PRIMARY)

Every model is evaluated on: Brier, log-likelihood, information gain vs Poisson, reliability curve, expected calibration error (ECE), sharpness. ROC-AUC and PR-AUC are SECONDARY. Accuracy is NOT the primary metric.

## 7. Results — model comparison

### Per (horizon, threshold) configuration


#### Horizon 7d, threshold M≥4.5

| Model | Feature set | N test | N+ | Base rate | Brier | Brier Poisson | ΔBrier | IG vs Poisson | ECE | Sharpness | ROC-AUC (sec) | PR-AUC (sec) | Verdict |
|-------|-------------|--------|-----|-----------|-------|---------------|--------|---------------|-----|-----------|---------------|--------------|---------|
| gb | ML-A | 576 | 7 | 0.012152777777777778 | 0.022 | 0.206 | 0.184 | 0.318 | 0.024 | 0.100 | 0.515 | 0.015 | BEATS |
| gb | ML-F | 576 | 7 | 0.012152777777777778 | 0.021 | 0.206 | 0.185 | 0.309 | 0.021 | 0.094 | 0.464 | 0.016 | BEATS |
| logistic_l2 | ML-A | 576 | 7 | 0.012152777777777778 | 0.132 | 0.206 | 0.074 | 0.024 | 0.256 | 0.243 | 0.522 | 0.041 | BEATS |
| logistic_l2 | ML-F | 576 | 7 | 0.012152777777777778 | 0.050 | 0.206 | 0.156 | 0.286 | 0.054 | 0.187 | 0.460 | 0.014 | BEATS |
| poisson | none | 576 | 7 | 0.012152777777777778 | 0.206 | 0.206 | 0.000 | 0.000 | 0.439 | 0.032 | 0.468 | 0.013 | baseline |

#### Horizon 30d, threshold M≥5.0

| Model | Feature set | N test | N+ | Base rate | Brier | Brier Poisson | ΔBrier | IG vs Poisson | ECE | Sharpness | ROC-AUC (sec) | PR-AUC (sec) | Verdict |
|-------|-------------|--------|-----|-----------|-------|---------------|--------|---------------|-----|-----------|---------------|--------------|---------|
| gb | ML-A | 576 | 6 | 0.010416666666666666 | 0.020 | 0.408 | 0.389 | 0.766 | 0.019 | 0.102 | 0.696 | 0.039 | BEATS |
| gb | ML-F | 576 | 6 | 0.010416666666666666 | 0.017 | 0.408 | 0.391 | 0.789 | 0.016 | 0.084 | 0.635 | 0.023 | BEATS |
| logistic_l2 | ML-A | 576 | 6 | 0.010416666666666666 | 0.060 | 0.408 | 0.348 | 0.764 | 0.101 | 0.218 | 0.746 | 0.075 | BEATS |
| logistic_l2 | ML-F | 576 | 6 | 0.010416666666666666 | 0.114 | 0.408 | 0.295 | 0.374 | 0.119 | 0.299 | 0.410 | 0.011 | BEATS |
| poisson | none | 576 | 6 | 0.010416666666666666 | 0.408 | 0.408 | 0.000 | 0.000 | 0.631 | 0.009 | 0.500 | 0.011 | baseline |

## 8. Ablation study

Sequential feature groups (ML-A through ML-F). The purpose: determine which information actually contributes predictive skill.

| Feature set | Configs beating Poisson | Total configs | Win rate |
|-------------|-------------------------|---------------|----------|
| ML-A | 4 | 4 | 100.0% |
| ML-B | 0 | 0 | N/A |
| ML-C | 0 | 0 | N/A |
| ML-D | 0 | 0 | N/A |
| ML-E | 0 | 0 | N/A |
| ML-F | 4 | 4 | 100.0% |
| ML-G | 0 | 0 | N/A |

## 9. Multiple-comparison control

- Total model × horizon × threshold × feature-set configurations tested: **10**
- Configurations beating Poisson: **8** (80.0%)
- With many configurations tested, the family-wise error rate is inflated. A single lucky configuration is NOT sufficient for success. We report the full matrix and the win rate.

## 10. Small-sample warning

- M≥6.5 and M≥7.0: too few historical events for reliable high-dimensional ML. Labeled 'research / exploratory' and NOT included as primary ML classification targets.
- M≥7.0: report the number of positive test cases explicitly (see per-config N+).

## 11. Scientific-conclusion questions

1. **Improvement over Poisson on proper probabilistic scoring?** 8/10 configurations beat Poisson on Brier.
2. **Improvement on genuinely unseen chronological data?** YES if #1 holds (chronological evaluation, no leakage).
3. **Reasonable calibration?** See ECE column in Section 7.
4. **Robustness across forecast origins?** See win rate across (horizon × threshold) configs.
5. **Stability across Mc scenarios?** See Mc sensitivity in metadata.
6. **No evidence of leakage?** YES — spatiotemporal leakage control enforced.
7. **Interpretable feature contributions?** See feature importance artifacts.
8. **Improvement not limited to one lucky combination?** See win rate — must be broad, not a single config.

## 12. Final Stage-7 answer

**CRITICAL CAVEAT:** The Poisson baseline compared here is the UNIFORM (temporal) Poisson, which assigns the same regional probability to every cell. The ML models beat this baseline largely by learning SPATIAL heterogeneity (most cells have no events; a few cells have most events). The Spatial Poisson baseline (Stage 4) also captures this heterogeneity. A fair ML-vs-Spatial-Poisson comparison is needed to determine whether ML adds skill BEYOND spatial rate estimation. The current result (ML beats uniform Poisson) is expected and does NOT by itself demonstrate that ML adds skill beyond the spatial baseline.

> ML beats the uniform Poisson baseline (8/8 ML configurations), largely by learning spatial heterogeneity. Whether ML adds skill BEYOND the Spatial Poisson baseline (Stage 4) requires a direct ML-vs-Spatial-Poisson comparison, which is the recommended next step. The current result is PARTIAL: ML clearly beats uniform Poisson, but the scientifically meaningful comparison (vs Spatial Poisson) is not yet performed in this run.

**B. PARTIAL — improvement over uniform Poisson, but spatial-Poisson comparison needed**

ML did NOT tune until it succeeded. The full experiment matrix is reported. If ML did not beat Poisson, that is reported as a valid scientific result, not a failure to tune.

## 13. Artifacts

- `outputs/stage7_report.md` (this file)
- `outputs/stage7_feature_catalog.csv` (feature names + groups)
- `outputs/stage7_model_results.csv` (per-config metrics)
- `outputs/stage7_backtest/` (per-origin predictions)
- `outputs/stage7_calibration/` (reliability curves)
- `outputs/stage7_feature_importance/` (permutation importance)
- `outputs/stage7_ablation/` (ablation summary)
- `outputs/stage7_spatial_generalization/` (region holdout)
- `outputs/stage7_depth_analysis/` (per-depth performance)
- `outputs/stage7_model_metadata.json` (experiment manifest)