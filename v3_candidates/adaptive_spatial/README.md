# FINAL_v3.0_CANDIDATE_ADAPTIVE_SPATIAL

> **STATUS: REJECTED (Verdict D — WORSE)**
> Retrospective validation complete. Posterior predictive check FAILED.
> **NOT deployed prospectively.**
> FINAL_v1.0_FROZEN remains PRODUCTION.
> FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL remains unchanged.

## What this is

A controlled MODEL DEVELOPMENT experiment testing whether **adaptive spatial
smoothing** of historical earthquake locations can improve probabilistic
forecasting for Bangladesh beyond the existing v1 (Spatial Poisson) and
v2 (Bayesian hierarchical) baselines.

This is Priority 2 from the reliability improvement roadmap.

## Scientific question

> Does replacing the rigid 1° grid-rate estimation with a spatially
> continuous adaptive kernel estimator provide genuine, reproducible,
> statistically defensible improvement in earthquake probability
> forecasting for Bangladesh?

## Model family

Four scientifically defensible variants were implemented and tested:

| Variant | Kernel | Adaptive | Selected parameter |
|---------|--------|----------|--------------------|
| A | Gaussian | No (fixed h) | h = 0.25° |
| B | Gaussian | Yes (NN k) | k = 10 |
| C | Epanechnikov | No (fixed h) | h = 0.5° |
| D | Epanechnikov | Yes (NN k) | k = 50 |

Selection was performed on a SELECTION period (2010-2014, 5 yearly origins)
using ONLY development-period data. The UNTOUCHED evaluation period
(2015-2023, 9 yearly origins) was used for final retrospective testing.

## Result

**Verdict: D. WORSE — reject.**

### Why rejected

1. **Posterior predictive check FAILS.** The smoothed rate field
   over-concentrates seismicity:
   - Simulated total events (mean 2365, 95% CI [2268, 2457]) exceeds
     observed (1890).
   - Simulated max cell count (mean 397, CI [362, 437]) exceeds observed
     (217).
   - Simulated Gini coefficient (0.684, CI [0.667, 0.699]) exceeds observed
     (0.649).
   - Simulated top-3 cell fraction (0.413, CI [0.393, 0.434]) exceeds
     observed (0.310).

   Kernel smoothing with small bandwidths creates sharper peaks at
   historical event clusters than the real seismicity exhibits. The
   bootstrap uncertainty does not fix this — it is a systematic bias
   in the estimator.

2. **No statistically significant Brier improvement.** Mean ΔBrier(v3−v1)
   = −0.00015; bootstrap CIs include zero in 4/4 configs vs v1 and 4/4
   vs v2.

3. **Calibration worse.** Mean ΔECE(v3−v1) = +0.00107; (v3−v2) = +0.00156
   (positive = v3 worse).

4. **Spatial holdout mixed.** v3 beats v1 in 2/4 quadrants, v2 in 2/4.

### What DID work

- **Grid stability:** v3 IS more stable across grid choices (0.5°/1.0°/2.0°)
  than v1. Brier range across grids: v1 = 0.00370, v3 = 0.00028. This
  confirms the theoretical advantage of a continuous field over a rigid
  grid. But this advantage does not translate to better predictive skill.

- **Sparse-cell behaviour:** v3 assigns non-zero probabilities to
  zero-event cells via kernel leakage from neighbours, with adaptive
  bandwidths broadening to 3.05° in zero-event cells. This is
  qualitatively better than v1's Garwood upper bound, but does not
  improve predictive skill.

## Files

| File | Description |
|------|-------------|
| `model.py` | Adaptive spatial smoothing implementation (4 variants, bootstrap, PPC) |
| `model_metadata.json` | Final metadata with verdict |
| `README.md` | This file |

## Companion artifacts (in `outputs/`)

| File | Description |
|------|-------------|
| `V3_ADAPTIVE_SPATIAL_REPORT.md` | Full experiment report (16 sections) |
| `v3_adaptive_results.csv` | Per-variant × config Brier/log-lik/ECE/sharpness + bootstrap CIs |
| `v3_adaptive_uncertainty.csv` | Per-origin × config uncertainty metrics |
| `v3_adaptive_calibration.csv` | 7-bin reliability per origin |
| `v3_adaptive_grid_sensitivity.csv` | 0.5°/1.0°/2.0° grid comparison |
| `v3_adaptive_bandwidth_sensitivity.csv` | All bandwidth/k candidates on selection period |
| `v3_adaptive_holdout.csv` | 4-quadrant spatial holdout |
| `v3_adaptive_sparse_cells.csv` | Zero/low/moderate/high cell analysis |
| `v3_adaptive_posterior_predictive.csv` | PPC statistics |
| `v3_adaptive_model_metadata.json` | Final metadata |

## Runner

```bash
cd bangladesh_eq_forecast
python run_v3_experiment.py
```

Runtime: ~2 minutes (vectorised kernel evaluation, cached v1/v2 reference
forecasts).

## Integrity

- **FINAL_v1.0_FROZEN:** source code unchanged; ledger unchanged; scores unchanged.
- **FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL:** source code unchanged; ledger unchanged; scores unchanged.
- **No evaluation-period leakage:** bandwidth/k selection used ONLY 2010-2014 data.
- **No forecast rewriting, no cherry-picking, no post-hoc threshold selection.**
- **No fabricated data or performance.**
- **No deterministic earthquake predictions.**

All v3 artifacts are in a SEPARATE namespace (`v3_candidates/adaptive_spatial/`
and `outputs/v3_adaptive_*`). No v1 or v2 file was modified, overwritten,
or deleted.

## Scientific takeaway

Adaptive spatial smoothing is theoretically appealing and does produce a
more grid-stable forecast field. However, for the Bangladesh catalog,
kernel smoothing with bandwidths small enough to capture local structure
also over-concentrates seismicity at historical cluster centres. The
resulting simulated catalogs are too peaked (higher Gini, higher top-3
fraction, higher max cell count) compared to observations.

This is a **null result with a clear mechanism**: the smoothing kernel
imposes a spatial structure (Gaussian or Epanechnikov peaks) that is
sharper than the true underlying seismicity field. The rigid 1° grid
of v1, despite its discontinuities, happens to better match the true
spatial scale of Bangladesh seismicity.

**A null result is scientifically valuable.** The v3 candidate is
rejected. FINAL_v1.0_FROZEN remains the production model.
