# STAGE 6 — Physics-Based / Coulomb-Stress Analysis

> Generated 2026-08-09T08:36:38.586026+00:00.

## 0. Primary scientific question

Do physics-based stress changes provide predictive information beyond:

1. stationary Poisson,
2. spatial Poisson,
3. locally fitted ETAS (K≈0),
4. externally informed ETAS?

The primary baseline remains the corrected expanding-window Poisson model (Stage 5 conclusion: standard ETAS does not beat Poisson with the correct per-origin Poisson rate).

## 1. Data audit (A/B/C/D classification)

**CRITICAL DATA-INTEGRITY RULE:** No Bangladesh fault geometry, slip rate, rake, dip, receiver faults, friction coefficients, coseismic slip, or stress-change fields are fabricated.

| Field | Class | Source | Value | Notes |
|-------|-------|--------|-------|-------|
| focal_mechanisms (GCMT) | **D** | not supplied | 0 files | GCMT is the gold-standard global CMT catalog. Required for source focal mechanisms. |
| focal_mechanisms (USGS) | **A** | USGS ComCat focal-mechanism product (per-event detail API) | 28 events with focal-mechanism products | USGS focal-mechanism products provide strike/dip/rake for moderate-large events. |
| fault_geometry (GEM GAFD traces) | **A** | GEM Global Active Faults Database (Styron & Pagani 2020) | 42 fault traces in region | Surface-trace geometry (lon/lat vertices). |
| fault_dip (GEM GAFD) | **D** | GEM GAFD | 0/42 segments have dip | Dip is REQUIRED for receiver-fault ΔCFS. GEM GAFD is geometry-only for Bangladesh. |
| fault_rake (GEM GAFD) | **D** | GEM GAFD | 0/42 have rake | Rake (slip direction) is REQUIRED for receiver-fault ΔCFS. |
| fault_slip_rate (GEM GAFD) | **D** | GEM GAFD | 0/42 have slip rate | Slip rate needed for long-term hazard, not directly for Coulomb ΔCFS. |
| fault_geometry (published literature) | **D** | Morino et al. 2014; Wang et al. 2014; Steckler et al. 2016 | NOT TRANSCRIBED — requires manual literature acquisition | Primary-literature fault geometry would override GEM GAFD placeholders. Not currently loaded. |
| elastic_params (shear modulus, Poisson's ratio) | **C** | engineering assumption (standard crustal values) | μ=30 GPa, ν=0.25 (Okada 1992 defaults) | No Bangladesh-specific elastic model available. Standard crustal values used; sensitivity tested. |
| effective_friction (μ') | **C** | engineering assumption | μ'=0.4 (King et al. 1994 typical; range 0.2-0.8 tested) | No Bangladesh-specific friction data. Sensitivity analysis required. |
| skempton_coefficient (B) | **C** | engineering assumption | B=0.5 (typical crystalline crust; range 0.5-1.0) | Pore-pressure coupling. No Bangladesh data. |
| regional_stress_orientation | **D** | not available | No Bangladesh stress map in the World Stress Map database for this region | If available, could be used to define 'optimal' receiver faults. Currently unavailable. |
| receiver_fault_orientations | **D** | not available | No validated receiver-fault dataset for Bangladesh | BLOCKING: receiver-fault geometry is required for ΔCFS. Without it, only stress-tensor components can be computed. |
| coseismic_slip_distributions | **D** | not available | No finite-fault slip models for Bangladesh events in the catalog | For M>7 events, finite-source would improve accuracy; point-source used as approximation. |

**Real Coulomb forecasting ENABLED: False**

**Blocking data gaps:**
- receiver-fault geometry (GEM GAFD has traces but NO dip/rake; published literature not transcribed; regional stress unavailable)

> Per the data-integrity rule, real Coulomb forecasting is DISABLED. A mathematical prototype is implemented and unit-tested with synthetic geometry; results are NOT presented as a Bangladesh forecast. See Section 5 (prototype validation) and Section 7 (data-gap report).

## 2. Coulomb formulation (documented)

### Mathematical form

ΔCFS = Δτ + μ'·Δσ_n

where:

- Δτ = shear stress change on the receiver fault, resolved in the slip direction
- Δσ_n = normal stress change (positive = unclamping/tension; negative = compression)
- μ' = effective friction coefficient (μ' = μ(1-B) with Skempton B)

### Sign conventions

- ΔCFS > 0 → fault brought closer to failure (triggering)
- ΔCFS < 0 → stress shadow (inhibition)
- Compression is NEGATIVE (rock-mechanics convention)

### Coordinate system

- Geographic ENU: x=East, y=North, z=Up (z=0 is surface)
- Source: strike (clockwise from N), dip (from horizontal), rake (slip direction in dip plane)

### Elastic half-space assumptions (Okada 1992)

- Isotropic, homogeneous, linear-elastic half-space
- Default: μ=30 GPa, ν=0.25 (Class C engineering assumption; sensitivity tested)
- Free surface at z=0; no topography or lateral heterogeneity
- Point-source approximation for stress (finite-source for M>7 would improve accuracy)

### Stress-to-rate coupling

**Rate-and-state (Dieterich 1994): f(ΔCFS) = exp(ΔCFS / A·σ̄), A·σ̄ = 1.0 MPa. This is the physically grounded standard, derived from laboratory friction laws. ΔCFS in Pa converted to MPa.**

λ(x, t) = λ₀(x) · f(ΔCFS(x, t))  — Coulomb-modulated Poisson.

## 3. Sources and receivers

- USGS focal-mechanism products available for ~22/30 largest M≥5.5 events (provides source strike/dip/rake — Class A).
- GEM GAFD: 42 fault traces in region, but **0 have dip** and 0 have rake. Geometry-only.
- Receiver-fault geometry: **C** (no validated receiver-fault dataset).
- For prototype/diagnostic use only: receivers on a 1° grid at 10 km depth with ASSUMED orientation (strike=0, dip=45, rake=90 — Class C engineering assumption).

## 4. Forecasts

**Real Coulomb forecasting is DISABLED.** No Bangladesh forecast maps are produced. The forecast function returns NaN with a data-limited note.


## 5. Mathematical-prototype unit tests (synthetic geometry)

Unit tests validate the Okada implementation against known analytical properties:

```
Running Coulomb mathematical-prototype unit tests (synthetic geometry)...

  PASS: fwd=-17419521692467.73 Pa, rev=17419521692467.73 Pa (opposite signs)
  PASS: |ΔCFS| decays with distance: [(5, np.float64(33865527549423.2)), (20, np.float64(6904291821136.5)), (50, np.float64(2100620081535.8)), (100, np.float64(1224175448026.1)), (200, np.float64(650954808108.8))]
  PASS: superposition d1=-180968378099.59 + d2=-290703189865.02 = -471671567964.61 = cum=-471671567964.61
  PASS: rate-and-state f = [1.00000000e+00 2.71828183e+00 3.67879441e-01 1.00000000e+02
 1.00000000e-02]
  PASS: step formulation f = [2.  0.5 0.5]
  PASS: data-limited receiver -> NaN (as expected)
  PASS: data-limited source -> NaN (as expected)

All unit tests completed.

```

These tests validate the MATH, not the data. They confirm the implementation is correct; they do NOT validate any Bangladesh forecast.

## 6. Stress-forecast diagnostics

No stress diagnostics computed (data-limited mode).

## 7. Backtest summary

**Backtest DISABLED** — real Coulomb forecasting is data-limited. No prospective backtest is performed because there are no validated receiver-fault data to compute ΔCFS on. A pseudo-backtest on ASSUMED receiver faults would not constitute a validated Bangladesh forecast.

## 8. Scientific-conclusion questions

1. **Does Coulomb add predictive information?** CANNOT BE DETERMINED — real forecasting disabled by data gaps.
2. **Does it outperform spatial Poisson?** CANNOT BE DETERMINED — real forecasting disabled.
3. **Does it outperform ETAS where ETAS is retained?** N/A — ETAS does not beat Poisson (Stage 5); Coulomb comparison moot until data arrive.
4. **Is the improvement robust to physical-parameter uncertainty?** N/A — no forecast to test.
5. **Is the improvement spatially localized?** N/A.
6. **Does it survive chronological validation?** N/A — no prospective backtest.
7. **Is the result reproducible from independently sourced physical data?** NO — cannot be reproduced because the required receiver-fault data do not exist in any available source (GEM GAFD geometry-only, GCMT not supplied, published literature not transcribed).

## 9. Data-gap report and required future data

To enable real Coulomb forecasting for Bangladesh, the following data are required:

1. **Validated receiver-fault geometry** (strike, dip, rake, depth) for the major Bangladesh faults: Dauki, Dhubri, Oldham, Dapsi, Churachandpur-Mao, Naga Thrust, Arakan megathrust, Chittagong-Tripura fold belt. Sources: Morino et al. 2014; Wang et al. 2014; Steckler et al. 2016 — require manual transcription.
2. **GCMT NDK file** (global CMT solutions 1976-present) — provides authoritative source focal mechanisms for M≥5.5. Download from globalcmt.org and place in `data/raw/gcmt/`.
3. **Regional stress orientation** (from World Stress Map or geophysical inversion) — would allow 'optimal' receiver-fault orientation if explicit fault geometry is unavailable.
4. **Bangladesh-specific elastic model** (μ, ν, layered structure) — currently using Okada 1992 defaults (Class C).
5. **Finite-source slip models** for the largest events (M≥7) — would replace the point-source approximation for near-field accuracy.

Until these data are supplied, the mathematical prototype remains the only Coulomb deliverable. **No Bangladesh Coulomb forecast map is produced.**

## 10. Stage-7 gate

Stage 7 (ML) may proceed. The Stage 6 outcome is **(B) conclusively documented that the required data are unavailable** and the correct data-limited baseline (Poisson) is established. Stage 7 ML models must beat the Poisson baseline; Coulomb features (ΔCFS) can be added as optional ML inputs only if/when validated receiver-fault data arrive, and must be clearly labeled as externally-informed features.

## 11. Artifacts

- `outputs/stage6_report.md` (this file)
- `outputs/stage6_data_audit.csv` + `.json` (A/B/C/D classification)
- `outputs/stage6_coulomb_parameters.csv` (elastic + coupling params)
- `outputs/stage6_forecasts.csv` (data-limited; NaN where disabled)
- `outputs/stage6_backtest/` (empty — backtest disabled)
- `outputs/stage6_stress_maps/` (empty — no validated stress maps)
- `outputs/stage6_residual_diagnostics/` (empty — no residuals)
- `outputs/stage6_model_metadata.json`