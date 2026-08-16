# Figure Catalog — Bangladesh Earthquake Forecasting Study

> All figures are generated from the frozen FINAL_v1.0 system. Every figure is reproducible from the source code and data files listed in the reproducibility guide.

| Figure | File | Description | Data Source | Status |
|--------|------|-------------|-------------|--------|
| 1 | `fig01_study_region.png` | Bangladesh study region and earthquake epicenter map | USGS+ISC merged catalog (5,779 events) | VALIDATED |
| 2 | `fig02_fmd_gr.png` | Magnitude-frequency distribution, Mc, and Gutenberg-Richter fit | Merged catalog; Mc=4.13 (MAXC); b=0.808 (MLE) | VALIDATED |
| 3 | `fig03_depth.png` | Depth distribution (shallow/intermediate/deep) and depth vs. magnitude | Merged catalog; mean depth 52.6 km | VALIDATED |
| 4 | `fig04_spatial_rate.png` | Spatial Poisson seismicity-rate map (M≥4.13, events/year/cell) | Causal expanding-window rate, 1° grid | VALIDATED |
| 5 | `fig05_temporal.png` | Annual earthquake count and temporal completeness Mc(t) | Merged catalog, 5-year rolling MAXC | VALIDATED |
| 6 | `fig06_omori.png` | Omori post-mainshock clustering diagnostic (non-parametric R(Δt)) | M≥5 (N=534) and M≥6 (N=22) mainshocks | DIAGNOSTIC |
| 7 | `fig07_model_comparison.png` | Model comparison (Brier score, 7d, untouched eval 2015–2024) | Final validation results | VALIDATED |
| 8 | `fig08_calibration.png` | Reliability/calibration diagram (Spatial Poisson, 7d, eval period) | Final validation, 9 origins × 64 cells | VALIDATED |
| 9 | `fig09_spatial_holdout.png` | Spatial holdout: ML does not generalize to held-out quadrants | 4-fold quadrant holdout | VALIDATED |
| 10 | `fig10_sensitivity.png` | Sensitivity: b-value vs. Mc; SP Brier vs. grid size | Mc=3.8–4.5; grid=0.5°–2.0° | SENSITIVITY |
| 11 | `fig11_forecast_map.png` | Final forecast: P(≥1 M≥4.13 in 7 days) per cell (Spatial Poisson) | Causal expanding-window, 1° grid | VALIDATED |
| 12 | `fig12_large_event_uncertainty.png` | Large-earthquake rate uncertainty with 95% CIs (Garwood) | M≥4.5 through M≥7.0; N=1 for M≥7 | DATA-LIMITED (M≥7) |
