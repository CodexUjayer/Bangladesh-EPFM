"""Coulomb stress transfer module (Stage 6).

DATA-INTEGRITY RULE: Real Coulomb forecasting requires validated fault
geometry (strike/dip/rake/depth) for RECEIVER faults. If unavailable, the
module operates in DATA-LIMITED mode. See data_audit.audit_coulomb_data().
"""

from .model import (
    ElasticParams,
    ReceiverFault,
    SourceEarthquake,
    okada_point_source,
    compute_cumulative_dcfs,
)
from .coupling import (
    CouplingFormulation,
    CouplingParams,
    stress_to_rate_factor,
    document_formulation,
)
from .data_audit import (
    CoulombDataAudit,
    FieldAudit,
    audit_coulomb_data,
    save_data_audit,
)
from .forecast import (
    CoulombForecast,
    StressDiagnostic,
    build_receiver_grid,
    build_source_earthquakes,
    forecast_coulomb_modulated_poisson,
    stress_forecast_diagnostic,
)

__all__ = [
    "ElasticParams",
    "ReceiverFault",
    "SourceEarthquake",
    "okada_point_source",
    "compute_cumulative_dcfs",
    "CouplingFormulation",
    "CouplingParams",
    "stress_to_rate_factor",
    "document_formulation",
    "CoulombDataAudit",
    "FieldAudit",
    "audit_coulomb_data",
    "save_data_audit",
    "CoulombForecast",
    "StressDiagnostic",
    "build_receiver_grid",
    "build_source_earthquakes",
    "forecast_coulomb_modulated_poisson",
    "stress_forecast_diagnostic",
]
