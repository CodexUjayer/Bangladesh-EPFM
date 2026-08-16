# Final Executive Summary — Bangladesh Earthquake Forecasting System

**System Version: FINAL_v1.0_FROZEN**

## What was studied?

Whether probabilistic earthquake forecasting for Bangladesh can improve upon historical spatial seismicity rates, using statistical (ETAS), machine-learning, and physics-based (Coulomb) approaches.

## What data were used?

- **USGS ComCat** (2,293 events, 1973–2024, floor M3.2)
- **ISC Bulletin** (5,576 events, 1973–2024, floor M2.4) — acquired during this study
- **Merged catalog**: 5,779 events, Mc≈4.13, b≈0.808, 51.9 years
- **Unavailable**: GCMT (focal mechanisms), BMD (local events), ISC-GEM (historical), published fault geometry

## What model won?

**Spatial Poisson** — the historical spatial seismicity-rate model. It assigns each grid cell a rate based on past earthquake activity and converts it to a probability using P(≥1 event) = 1 − exp(−λΔt).

## How was it validated?

- **Chronological**: expanding-window training; the model never saw future data
- **Untouched evaluation**: 2015–2024 reserved as a genuinely untouched test period
- **Spatial holdout**: 4-fold quadrant holdout testing whether ML generalizes or memorizes
- **Statistical**: block bootstrap CIs, permutation tests, multiple-comparison correction

## What did ETAS show?

- The locally-fitted ETAS model selected **K≈0** (no triggering detected in-sample)
- This result is **robust**: it survived 2.4× more data (ISC integration), corrected base-10 formulation, declustered background, and depth stratification (shallow, intermediate, deep all K≈0)
- ETAS does **not** beat Spatial Poisson at any horizon (1 hour through 90 days)
- **Interpretation**: The standard ETAS formulation appears misspecified for this catalog, not evidence that triggering is absent

## What did ML show?

- ML (gradient boosting, logistic regression) initially appeared to beat Poisson — but this was an artifact of comparing against **uniform** Poisson
- When compared against **Spatial Poisson**, ML loses at every configuration
- ML **fails the spatial holdout**: it memorizes which cells are historically active but does not learn transferable relationships
- **Interpretation**: ML adds no incremental information beyond spatial rate estimation

## What did Coulomb show?

- Coulomb forecasting is **disabled** — no validated receiver-fault geometry exists
- GCMT (focal mechanisms) could not be downloaded; GEM GAFD fault traces lack dip/rake
- A mathematical prototype was implemented and unit-tested with synthetic geometry
- **No Bangladesh Coulomb forecast is produced**

## What can the system forecast?

- **Probabilistic rate** of M≥4.5+ earthquakes per 1° grid cell over 7–30 day horizons
- **Spatial probability maps** showing where events are more likely (based on historical rates)
- **Uncertainty intervals** on these probabilities

## What can it NOT forecast?

- The exact time, location, or magnitude of any specific earthquake
- Reliable M≥6.5+ probabilities (too few events)
- Short-term aftershock sequences (ETAS cannot capture them despite real clustering)
- Coulomb stress effects (no receiver-fault data)
- Any deterministic prediction

## How uncertain are large-earthquake estimates?

| Threshold | N events | Rate (1/yr) | 95% CI | P(1 year) | Warning |
|-----------|----------|------------|--------|-----------|---------|
| M≥6.0 | 22 | 0.42 | [0.27, 0.65] | 0.37 | Moderate uncertainty |
| M≥6.5 | 8 | 0.15 | [0.07, 0.33] | 0.15 | High uncertainty |
| M≥7.0 | 1 | 0.019 | [0.0005, 0.14] | 0.02 | **VERY HIGH — not a precise forecast** |

## What are the most important limitations?

1. **No GCMT** → no focal mechanisms → Coulomb disabled, ETAS spatial kernels uninformed
2. **No BMD** → local M2-3 events missing → Mc could be lower, more aftershocks for ETAS
3. **M≥7 N=1** → recurrence estimates span an order of magnitude
4. **Standard ETAS misspecified** → the catalog has real Omori clustering (R≈24×) but ETAS cannot convert it to forecast skill
5. **52-year observation period** → short for M≥7 recurrence estimation
6. **No historical catalog** → pre-1900 great earthquakes (1762 Arakan, 1897 Shillong) absent

## Key positive finding

The catalog **does** exhibit strong short-lag post-mainshock temporal clustering (R≈24× for M≥5, 290× for M≥6 at Δt≈0.01 days). This is real observed evidence. The inability of standard ETAS to convert this into forecast skill is a **model limitation**, not evidence that triggering is absent.

## Final conclusion

> Historical spatial seismicity rates provide the strongest validated probabilistic forecasting baseline for the available Bangladesh earthquake catalog. Under strict chronological, spatial, and spatiotemporal validation, the tested ETAS and machine-learning formulations did not demonstrate statistically defensible incremental predictive skill beyond this spatial baseline. The catalog nevertheless exhibits strong short-lag post-mainshock temporal clustering, so the inability of the tested ETAS formulations to improve forecasts should not be interpreted as evidence that earthquake triggering is absent. Forecast uncertainty becomes substantial for rare large-magnitude events, and missing local, focal-mechanism, and historical data limit stronger physical inference.
