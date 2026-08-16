# Reproducibility Guide — Bangladesh Earthquake Forecasting System (FINAL_v1.0_FROZEN)

## 1. Input Data

| File | Source | N events | Access |
|------|--------|----------|--------|
| `data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv` | USGS FDSN API | 2,293 | `https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=1973-01-01&endtime=2025-01-01&minmagnitude=2.5&minlatitude=20&maxlatitude=28&minlongitude=88&maxlongitude=96&eventtype=earthquake&orderby=time-asc` |
| `data/raw/isc/isc_bangladesh_1973_2025_m3.txt` | ISC FDSN API | 5,576 | `http://www.isc.ac.uk/fdsnws/event/1/query?format=text&starttime=1973-01-01&endtime=2025-01-01&minlat=20&maxlat=28&minlon=88&maxlon=96&minmag=3.0&orderby=time-asc` |

## 2. Software Environment

- Python 3.12+
- Key packages: numpy, scipy, scikit-learn, matplotlib, pandas
- No GPU required; all models fit in <2 minutes on a standard CPU

## 3. Source-Code Entry Points

| Runner | Purpose | Output |
|--------|---------|--------|
| `run_stage3.py` | Catalog harmonization, QC, completeness, declustering | `outputs/stage3_report.md` |
| `run_stage4.py` | Poisson + GR baselines | `outputs/stage4_report.md` |
| `run_stage5.py` | ETAS fitting (base-10, GK background) | `outputs/stage5_report.md` |
| `run_stage5_validation.py` | ETAS event-conditioned backtest | `outputs/stage5_validation_report.md` |
| `run_stage6.py` | Coulomb data audit + prototype | `outputs/stage6_report.md` |
| `run_stage7.py` | ML vs uniform Poisson (SUPERSEDED) | `outputs/stage7_report.md` |
| `run_stage7b.py` | ML vs Spatial Poisson | `outputs/stage7b_report.md` |
| `run_phase_b.py` | Missing validation experiments | `outputs/PHASE_B_REPORT.md` |
| `run_phase_c.py` | ISC acquisition + catalog merge | `outputs/PHASE_C_REPORT.md` |
| `run_phase_d.py` | Full revalidation on expanded catalog | `outputs/PHASE_D_REPORT.md` |
| `run_stage8.py` | Depth-stratified ETAS + short-horizon | `outputs/STAGE8_REPORT.md` |
| `run_final.py` | **Final validation & freeze** | `outputs/FINAL_BANGLADESH_EARTHQUAKE_FORECASTING_REPORT.md` |

## 4. Reproducing the Final Results

```bash
# 1. Ensure data files are in data/raw/{usgs,isc}/
# 2. Run the final validation:
python3 run_final.py

# 3. All final outputs are in outputs/:
#    - FINAL_BANGLADESH_EARTHQUAKE_FORECASTING_REPORT.md
#    - final_forecasts.csv
#    - final_model_comparison.csv
#    - final_uncertainty.csv
#    - final_validation_results.csv
#    - final_sensitivity.csv
#    - final_data_quality.csv
#    - final_model_metadata.json
```

## 5. Key Parameters (Frozen)

| Parameter | Value | Source |
|-----------|-------|--------|
| Catalog | USGS+ISC merged | Phase C |
| Matching window | 120s, 50km | Stage 2 |
| Mc | 4.13 (median of MAXC/GFT/EMR/Stepp) | Phase C |
| b-value | 0.808 ± 0.010 | Phase C/D |
| Grid | 1.0° (64 cells) | Stage 4 |
| Dev period | 1973–2006 | Final run |
| Selection period | 2006–2015 | Final run |
| Eval period | 2015–2024 (untouched) | Final run |
| ETAS formulation | Base-10: K·10^{α(M-Mc)} | Phase A |
| ETAS background | Gardner-Knopoff declustered KDE | Phase A |
| ML features | 43 causal features (ML-F set) | Stage 7 |
| Random seed | 42 | All stages |

## 6. Random Seeds

All stochastic operations use seed=42 for reproducibility:
- Block bootstrap: `np.random.default_rng(42)`
- ML training: `random_state=42` in scikit-learn
- Origin subsampling: `np.random.default_rng(42)`

## 7. Provenance Chain

Every result traces through:
```
USGS CSV + ISC TXT
  → build_canonical_events (120s/50km matching)
    → estimate_completeness (4 methods)
      → fit_gutenberg_richter (Aki-Utsu MLE)
        → causal_spatial_rate (expanding-window)
          → evaluate_model (Brier, ECE, log-lik)
            → final_model_metadata.json (FROZEN)
```
