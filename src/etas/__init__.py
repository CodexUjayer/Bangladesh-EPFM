"""ETAS (Epidemic-Type Aftershock Sequence) model for Bangladesh.

Stage 5. The conditional intensity has the explicit form:

    λ(x, y, t | H_t) = μ(x, y)
                     + Σ_{i: t_i < t}  K · exp[α (M_i − M_c)]
                       · g(t − t_i; c, p)
                       · f(x − x_i, y − y_i; M_i, σ, γ, q)

where:
  μ(x, y)     background rate (uniform or spatially-varying via KDE)
  K           productivity amplitude
  α           magnitude-scaling exponent (productivity ∝ 10^{α(M-Mc)})
  c, p        Omori-Utsu temporal parameters: g(τ) = (p-1) c^{p-1} / (τ+c)^p
              (normalized so ∫_0^∞ g(τ) dτ = 1)
  f(·)        spatial triggering kernel (power-law or Gaussian; normalized)
  σ, γ, q     spatial-kernel parameters
  M_c         magnitude threshold (lower threshold for fitting)

Branching ratio n = ∫ K·10^{α(M-Mc)} · g(τ) · f(x,y) dτ dx dy dM
                = K · E[10^{α(M-Mc)}]  (since g and f are normalized)
which estimates the average number of direct aftershocks per event.

We do NOT copy parameter values from any other region. Parameters are
estimated from the Bangladesh catalog by MLE (see estimation.py) with
parameter bounds; if the data do not support reliable estimation of a
parameter, that is reported explicitly (see identifiability flags).
"""

from .model import (
    ETASParams,
    ETASModel,
    conditional_intensity,
)
from .omori import omori_utsu_g, omori_normalization
from .spatial_kernels import (
    power_law_spatial_kernel,
    gaussian_spatial_kernel,
    spatial_normalization,
)
from .background import (
    UniformBackground,
    KDEBackground,
    BackgroundRate,
)
from .estimation import (
    ETASFitResult,
    fit_etas_mle,
    parameter_identifiability,
)
from .branching import compute_branching_ratio, branching_plausibility
from .forecast import (
    ETASForecast,
    forecast_temporal,
    forecast_spatial,
)
from .residuals import (
    ResidualDiagnostics,
    compute_residuals,
)
from .backtest import (
    ETASBacktestResult,
    run_etas_backtest,
    event_conditioned_backtest,
)
from .report import (
    generate_stage5_report,
    save_stage5_artifacts,
)
from .event_conditioned import (
    ConditionedOrigin,
    ConditionedBacktestResult,
    build_conditioned_origins,
    run_full_conditioned_backtest,
)
from .sensitivity import (
    SensitivityResult,
    SensitivitySummary,
    DEFAULT_EXTERNAL_PARAMS,
    PUBLISHED_PRIORS,
    run_sensitivity_analysis,
)
from .depth_analysis import DepthGroupResult, analyze_depth_dependence
from .omori_diagnostic import OmoriDiagnosticResult, compute_omori_diagnostic
from .spatial_diagnostic import SpatialDiagnosticResult, compute_spatial_diagnostic
from .validation_report import (
    generate_stage5_validation_report,
    save_stage5_validation_artifacts,
)

__all__ = [
    "ETASParams",
    "ETASModel",
    "conditional_intensity",
    "omori_utsu_g",
    "omori_normalization",
    "power_law_spatial_kernel",
    "gaussian_spatial_kernel",
    "spatial_normalization",
    "UniformBackground",
    "KDEBackground",
    "BackgroundRate",
    "ETASFitResult",
    "fit_etas_mle",
    "parameter_identifiability",
    "compute_branching_ratio",
    "branching_plausibility",
    "ETASForecast",
    "forecast_temporal",
    "forecast_spatial",
    "ResidualDiagnostics",
    "compute_residuals",
    "ETASBacktestResult",
    "run_etas_backtest",
    "event_conditioned_backtest",
    "generate_stage5_report",
    "save_stage5_artifacts",
    "ConditionedOrigin",
    "ConditionedBacktestResult",
    "build_conditioned_origins",
    "run_full_conditioned_backtest",
    "SensitivityResult",
    "SensitivitySummary",
    "DEFAULT_EXTERNAL_PARAMS",
    "PUBLISHED_PRIORS",
    "run_sensitivity_analysis",
    "DepthGroupResult",
    "analyze_depth_dependence",
    "OmoriDiagnosticResult",
    "compute_omori_diagnostic",
    "SpatialDiagnosticResult",
    "compute_spatial_diagnostic",
    "generate_stage5_validation_report",
    "save_stage5_validation_artifacts",
]
