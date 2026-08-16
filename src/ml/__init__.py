"""Stage 7 ML forecasting: feature engineering, model ladder, chronological evaluation.

PRIMARY QUESTION: Can ML produce better calibrated, genuinely out-of-sample
probabilistic earthquake forecasts than the corrected Poisson baseline?

NO-LEAKAGE RULE (absolute): Every feature at forecast origin t uses only
information available at or before t. No future earthquakes, magnitudes,
aftershock labels, completeness, declustering, spatial rates, model outputs,
or future-based Coulomb calculations enter the feature matrix.

SPATIOTEMPORAL LEAKAGE CONTROL: Grid-cell observations are NOT independent.
All cells from one forecast origin stay in the same temporal split. The model
never sees neighboring future cells from the same timestamp during training.

CALIBRATION IS PRIMARY: Brier, log-likelihood, information gain vs Poisson,
reliability curves, ECE, sharpness. ROC-AUC and PR-AUC are SECONDARY.
Accuracy is NOT the primary metric.

Coulomb features are DISABLED (Stage 6 data-limited). ML-G (ML + Coulomb)
branch is reserved for future validated data.
"""
