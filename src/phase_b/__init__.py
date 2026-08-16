"""Phase B experiments: ETAS-vs-SP direct comparison, spatial holdout, depth-stratified,
uncertainty propagation, power analysis, Mc sensitivity, validation design,
multiple-comparison control.

All experiments use the Phase-A corrected code (base-10 ETAS, declustered
background, fixed base-rate check). Strict chronological expanding-window
evaluation. No models are tuned on the evaluation period.
"""

from .etas_vs_sp import run_etas_vs_sp_comparison
from .spatial_holdout import run_spatial_holdout
from .depth_stratified import run_depth_stratified_analysis
from .uncertainty import run_uncertainty_propagation
from .power_analysis import run_power_analysis
from .mc_sensitivity import run_mc_sensitivity
from .validation_design import run_validation_design_analysis
from .multiple_comparison import run_multiple_comparison_control

__all__ = [
    "run_etas_vs_sp_comparison",
    "run_spatial_holdout",
    "run_depth_stratified_analysis",
    "run_uncertainty_propagation",
    "run_power_analysis",
    "run_mc_sensitivity",
    "run_validation_design_analysis",
    "run_multiple_comparison_control",
]
