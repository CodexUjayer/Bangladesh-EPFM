# Table Catalog — Bangladesh Earthquake Forecasting Study

| Table | Section | Description | Key Data | Status |
|-------|---------|-------------|----------|--------|
| T1 | Data Sources | Catalog sources, status, N events, floor | USGS (2,293), ISC (5,576), merged (5,779) | VALIDATED |
| T2 | Catalog Statistics | N, exposure, mag range, depth stats, magnitude types | 5,779 events, 51.89 yr, M2.4–7.2, mean depth 52.6 km | VALIDATED |
| T3 | Completeness | Mc by 4 methods (MAXC, GFT, EMR, Stepp) | MAXC=4.05, GFT=5.65, EMR=4.15, Stepp=4.10; rec=4.13 | VALIDATED |
| T4 | GR Parameters | b, σ_b, a, N, Mc | b=0.808±0.010, a=5.89, N=3,436, Mc=4.13 | VALIDATED |
| T5 | Model Definitions | Uniform Poisson, Spatial Poisson, ETAS, ML, Coulomb | See paper §3 | VALIDATED |
| T6 | Validation Design | Dev/selection/eval split, origins, horizons, thresholds | Dev<2006, Sel 2006–2015, Eval 2015–2024 | VALIDATED |
| T7 | Final Model Comparison | Brier, ECE, beats-SP for all models on eval period | SP=0.0242, ETAS=0.4355, ML=0.0327 | VALIDATED |
| T8 | Spatial Holdout | SP vs ML per quadrant (held-out cells) | SP wins all 4 quadrants | VALIDATED |
| T9 | Sensitivity | Mc (3.8–4.5), grid (0.5°–2.0°), data source (USGS/ISC/merged) | Ranking unchanged | SENSITIVITY |
| T10 | Uncertainty | Aleatory, epistemic, total σ for M4.5–M7.0 | M≥7: total σ > point estimate | DATA-LIMITED |
| T11 | Large-Event Recurrence | N, rate, 95% CI, P(1yr), power for M6–M7 | M≥7: N=1, CI [0.0005, 0.14] | DATA-LIMITED |
| T12 | Data Limitations | 14 documented limitations | GCMT, BMD, ISC-GEM, historical, receiver faults | DATA-LIMITED |
| T13 | Model-Status Hierarchy | 7 models ranked with status labels | SP=VALIDATED; ETAS=PRELIMINARY; ML=no skill; Coulomb=DATA-LIMITED | VALIDATED |
