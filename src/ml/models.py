"""Model ladder: Poisson, Logistic (L1/L2/ElasticNet), RandomForest, GradientBoosting, CalibratedGB.

Implements Models 0-5 from the Stage 7 spec. Model 6 (neural) is reserved for
only if temporal structure justifies it; not implemented by default.

All models expose a common interface: fit(X_train, y_train, sample_weight)
and predict_proba(X) -> P(event in cell during horizon).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    """One model's fit + predictions on a test set."""

    model_name: str
    feature_set: str
    y_pred_proba: np.ndarray
    y_true: np.ndarray
    n_train: int
    n_test: int
    n_positive_train: int
    n_positive_test: int
    hyperparams: dict = field(default_factory=dict)
    calibration_method: str = "none"
    fit_time_s: float = 0.0


def fit_poisson_baseline(
    y_train: np.ndarray, exposure_years_train: float,
    poisson_rate_per_year_at_origin: float, horizon_years: float,
    n_test: int,
) -> ModelResult:
    """Model 0: Poisson baseline (expanding-window rate at each origin).

    The Poisson 'prediction' for every test cell is the same: P = 1 - exp(-λΔt)
    where λ is the expanding-window rate at the forecast origin.
    """
    p = 1.0 - math.exp(-poisson_rate_per_year_at_origin * horizon_years)
    y_pred = np.full(n_test, p)
    return ModelResult(
        model_name="poisson", feature_set="none",
        y_pred_proba=y_pred, y_true=np.array([]),  # filled by caller
        n_train=0, n_test=n_test, n_positive_train=0, n_positive_test=0,
        hyperparams={"rate": poisson_rate_per_year_at_origin},
        calibration_method="analytic",
    )


def fit_logistic_l2(X_train, y_train, X_test, sample_weight=None, C=1.0):
    """Model 1: L2-regularized logistic regression."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        penalty="l2", C=C, class_weight="balanced",
        max_iter=1000, random_state=42, solver="lbfgs",
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_weight)
    y_pred = clf.predict_proba(X_test_s)[:, 1]
    return y_pred, clf, scaler


def fit_logistic_elasticnet(X_train, y_train, X_test, sample_weight=None,
                             C=1.0, l1_ratio=0.5):
    """Model 2: Elastic Net logistic regression."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        penalty="elasticnet", C=C, l1_ratio=l1_ratio, class_weight="balanced",
        max_iter=2000, random_state=42, solver="saga",
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_weight)
    y_pred = clf.predict_proba(X_test_s)[:, 1]
    return y_pred, clf, scaler


def fit_random_forest(X_train, y_train, X_test, sample_weight=None,
                      n_estimators=200, max_depth=8):
    """Model 3: Random Forest."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred, clf, None


def fit_gradient_boosting(X_train, y_train, X_test, sample_weight=None,
                          n_estimators=200, max_depth=3, learning_rate=0.1):
    """Model 4: Gradient Boosting."""
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42,
    )
    # class_weight not directly supported; use sample_weight
    if sample_weight is None:
        # Compute balanced sample weights
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            sample_weight = np.where(y_train == 1, n_neg / n_pos, 1.0)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred, clf, None


def fit_calibrated_gradient_boosting(X_train, y_train, X_test,
                                     n_estimators=200, max_depth=3,
                                     learning_rate=0.1, method="isotonic"):
    """Model 5: Calibrated Gradient Boosting (isotonic or Platt)."""
    base = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42,
    )
    # Compute sample weights for imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    sw = np.where(y_train == 1, n_neg / max(n_pos, 1), 1.0) if n_pos > 0 else None
    # CalibratedClassifierCV uses internal CV; for small data use cv=3
    cv = 3 if len(y_train) >= 30 else 2
    clf = CalibratedClassifierCV(base, method=method, cv=cv)
    clf.fit(X_train, y_train, sample_weight=sw)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred, clf, None
