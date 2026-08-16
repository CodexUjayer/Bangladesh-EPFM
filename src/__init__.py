"""Bangladesh Probabilistic Earthquake Forecasting System.

A research-grade probabilistic seismicity forecasting project for Bangladesh
and surrounding tectonic regions. This package is organized into two
scientifically distinct products:

PRODUCT 1 — Short-Term Seismicity Forecast (operational, primary)
    Horizons: 24h, 7d, 30d
    Methods : ETAS + Coulomb (when fault data exist) + ML/transfer-learning
    Output  : P(>=1 event >= Mc) per grid cell, expected rate, uncertainty

PRODUCT 2 — Long-Term Large-Earthquake Hazard Research (research, secondary)
    Horizons: months / years / decades
    Methods : Gutenberg-Richter recurrence, Bayesian priors, fault-based
              moment budgets, PSHA-compatible concepts
    Output  : recurrence rates, Mmax distributions, NOT short-term prediction

These products share a common data pipeline (ingestion -> harmonization ->
completeness -> declustering) but diverge in modeling assumptions and must
NEVER be conflated. See STAGE2_DATA_SPEC.md for the full distinction.
"""

__version__ = "0.1.0-stage2"
