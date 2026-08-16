"""Stage 4 statistical baselines: Poisson, Gutenberg-Richter, spatial, backtest.

This package implements the statistical baseline layer that Stage 5 ETAS
must beat. Nothing here uses aftershock triggering, Coulomb stress, ML, or
transfer learning — those belong to later stages.

Probability interpretation discipline (enforced throughout):
  - lambda : earthquake rate (events per year), estimated as N / T
  - expected count over dt : lambda * dt  (NOT a probability)
  - P(N >= 1 | dt) : 1 - exp(-lambda * dt)  (probability of >=1 event)
  - cell probability : same formula with the cell's lambda
  - everything is conditional on the observed catalog and the working Mc

We never use the word "risk" (which requires exposure/vulnerability).
"""

from .uncertainty import (
    poisson_rate_ci_garwood,
    poisson_rate_ci_jeffreys,
    probability_ci_from_rate_ci,
    bootstrap_bvalue_ci,
)
from .poisson import (
    PoissonRateEstimate,
    TemporalPoissonResult,
    estimate_temporal_poisson,
    probability_at_least_one,
    expected_count,
)
from .gutenberg_richter import (
    GRResult,
    fit_gutenberg_richter,
    fit_gr_multiple_thresholds,
)
from .spatial import (
    GridConfig,
    GridCell,
    SpatialGrid,
    build_spatial_grid,
)
from .forecast import (
    CellForecast,
    SpatialForecast,
    forecast_spatial,
)
from .large_events import (
    LargeEventAssessment,
    assess_large_events,
)
from .backtest import (
    BacktestOrigin,
    BacktestResult,
    run_chronological_backtest,
    brier_score,
    log_likelihood_score,
    reliability_diagram,
    information_gain,
    roc_auc,
)
from .report import (
    generate_stage4_report,
    save_stage4_artifacts,
)

__all__ = [
    "poisson_rate_ci_garwood",
    "poisson_rate_ci_jeffreys",
    "probability_ci_from_rate_ci",
    "bootstrap_bvalue_ci",
    "PoissonRateEstimate",
    "TemporalPoissonResult",
    "estimate_temporal_poisson",
    "probability_at_least_one",
    "expected_count",
    "GRResult",
    "fit_gutenberg_richter",
    "fit_gr_multiple_thresholds",
    "GridConfig",
    "GridCell",
    "SpatialGrid",
    "build_spatial_grid",
    "CellForecast",
    "SpatialForecast",
    "forecast_spatial",
    "LargeEventAssessment",
    "assess_large_events",
    "BacktestOrigin",
    "BacktestResult",
    "run_chronological_backtest",
    "brier_score",
    "log_likelihood_score",
    "reliability_diagram",
    "information_gain",
    "roc_auc",
    "generate_stage4_report",
    "save_stage4_artifacts",
]
