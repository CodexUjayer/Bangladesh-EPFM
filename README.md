# Bangladesh Earthquake Forecasting Platform

**A publication-grade, real-time probabilistic earthquake forecasting system for Bangladesh and the surrounding plate-boundary region.**

> **Author:** Ujayer Hasnat — Data Scientist, Data Analyst
> **Production Model:** FINAL_v1.0_FROZEN (Spatial Poisson)
> **Catalog:** 5,779 events (USGS + ISC merged), 1973–2024, Mc = 4.13, b = 0.808
> **Validation:** Brier 0.0242 (7-day), 0.0763 (30-day) on untouched 2015–2024 evaluation period

---

## Scientific Disclaimer

This system provides **probabilistic forecasts only**. It cannot predict the exact time, location, or magnitude of earthquakes. Large-event probabilities (M≥7) remain highly uncertain. This platform should **not** be used as an emergency warning system. Every probability includes uncertainty; consult the 95% confidence intervals.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dependencies](#dependencies)
4. [Environment Setup](#environment-setup)
5. [How to Run the Dashboard](#how-to-run-the-dashboard)
6. [How to Run the Forecasting Pipeline](#how-to-run-the-forecasting-pipeline)
7. [How to Update Catalogs](#how-to-update-catalogs)
8. [How to Use the APIs](#how-to-use-the-apis)
9. [How to Regenerate Forecasts](#how-to-regenerate-forecasts)
10. [How to Reproduce the Paper](#how-to-reproduce-the-paper)
11. [Project Structure](#project-structure)
12. [Model Hierarchy](#model-hierarchy)
13. [Audit and Integrity](#audit-and-integrity)
14. [License and Citation](#license-and-citation)

---

## Project Overview

This project develops, validates, and deploys a probabilistic earthquake forecasting system for Bangladesh. The production model (FINAL_v1.0_FROZEN) is a Spatial Poisson model that estimates per-cell seismicity rates on a 1°×1° grid (64 cells covering 20–28°N, 88–96°E) and converts them to probabilities via P(N≥1) = 1 − exp(−λΔt) with Garwood 95% confidence intervals.

Three candidate extensions were developed and formally rejected:
- **v2 (Bayesian Hierarchical Spatial):** ΔBrier ≈ 0; verdict B (uncertainty improvement only)
- **v3 (Adaptive Spatial Smoothing):** Posterior predictive check FAIL; verdict D (REJECTED)
- **v4 (Region-Specific ETAS):** K≈0 in all variants; verdict D (REJECTED)

The platform includes a real-time dashboard with an interactive Leaflet map, live USGS+ISC earthquake feed, validation metrics, publication figures, and a 45-page scientific paper.

---

## Installation

### Prerequisites

- **Node.js** 18+ and **Bun** (for the Next.js dashboard)
- **Python** 3.10+ (for the forecasting pipeline and scientific code)
- **Tectonic** 0.15+ (for compiling the LaTeX paper, optional)

### Step 1: Clone or extract the project

```bash
# If using the ZIP archive:
unzip BANGLADESH_EARTHQUAKE_FORECASTING_COMPLETE_PROJECT.zip
cd bangladesh_eq_forecast
```

### Step 2: Install Python dependencies

```bash
cd bangladesh_eq_forecast
pip install -r requirements.txt
```

Key Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

### Step 3: Install Node.js dependencies (for the dashboard)

The dashboard is in the parent Next.js project. From the project root:

```bash
cd ..  # Navigate to the Next.js project root (contains package.json)
bun install
```

---

## Dependencies

### Python (scientific code)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Numerical computation |
| scipy | ≥1.10 | Statistics, optimization |
| pandas | ≥2.0 | Data manipulation |
| matplotlib | ≥3.7 | Figure generation |
| scikit-learn | ≥1.3 | Machine learning (v1 ML experiments) |

See `bangladesh_eq_forecast/requirements.txt` for the full list.

### Node.js (dashboard)

| Package | Version | Purpose |
|---------|---------|---------|
| next | ^16.1.1 | Web framework |
| react | ^19.0.0 | UI library |
| leaflet | ^1.9.4 | Interactive maps |
| react-leaflet | ^5.0.0 | React bindings for Leaflet |
| recharts | ^2.15.4 | Charts |
| tailwindcss | ^4 | Styling |
| shadcn/ui | — | UI components |

See `package.json` for the full list.

### System (optional)

| Tool | Purpose |
|------|---------|
| tectonic ≥0.15 | Compile the LaTeX paper to PDF |
| pdfinfo | Verify PDF metadata (optional) |

---

## Environment Setup

### 1. Python environment

```bash
cd bangladesh_eq_forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Node.js environment

```bash
# From the Next.js project root (parent of bangladesh_eq_forecast/)
bun install
```

### 3. Catalog data

The raw earthquake catalogs are pre-bundled in `bangladesh_eq_forecast/data/raw/`:
- `usgs/usgs_bangladesh_1973_2025_m25.csv` — USGS ComCat (2,293 events)
- `isc/isc_bangladesh_1973_2025_m3.txt` — ISC Bulletin (5,533 events)

No additional data download is required to run the system. To update with live data, see [How to Update Catalogs](#how-to-update-catalogs).

### 4. Tectonic (optional, for paper compilation)

```bash
# Install tectonic (see https://tectonic-typesetting.github.io/)
# On Linux:
curl -fsSL https://drop-sh.fullyjustified.net | sh
# Or via cargo:
cargo install tectonic
```

---

## How to Run the Dashboard

The dashboard is a Next.js application with an interactive Leaflet map, live earthquake feed, validation metrics, and publication figures.

### Starting the dev server

```bash
# From the Next.js project root (parent of bangladesh_eq_forecast/)
bun run dev
```

The server starts on `http://localhost:3000`.

### Dashboard tabs

| Tab | Content |
|-----|---------|
| **Forecast** | Interactive Leaflet map (forecast grid + historical earthquakes + GEM faults + live feed), current probabilities panel, live earthquake feed, human-readable forecast statement |
| **Predictions vs Reality** | Side-by-side forecast-vs-observed comparison table |
| **Live Accuracy** | Evidence level (0–4), cumulative Brier/log-score/ECE/sharpness, per-config hit rate / false alarm / precision / recall / F1 |
| **Forecast History** | Immutable SHA-256-hashed forecast ledger |
| **Publication Figures** | 16 figures at 300 DPI with captions |
| **Model Hierarchy** | v1 (PRODUCTION), v2 (CANDIDATE), v3 (REJECTED), v4 (REJECTED) |

### Auto-refresh

The dashboard auto-refreshes every 5 minutes. A manual "Refresh" button is available in the header. The "Re-run Pipeline" button triggers a new forecast generation.

---

## How to Run the Forecasting Pipeline

The pipeline fetches live USGS+ISC data, generates v1 (Spatial Poisson) and v2 (Bayesian) forecasts, and saves them to immutable SHA-256-hashed ledgers.

### Single-model pipeline (v1 only)

```bash
cd bangladesh_eq_forecast
python3 live/pipeline.py
```

This generates `live/latest_forecast.json` (the snapshot the dashboard reads).

### Parallel pipeline (v1 + v2)

```bash
cd bangladesh_eq_forecast
python3 live/parallel_pipeline.py
```

This generates:
- `live/latest_parallel.json` (dual snapshot)
- `live/forecast_ledger/v1/forecast_YYYY-MM-DD_HHMMSS.json`
- `live/forecast_ledger/v2_bayesian/forecast_YYYY-MM-DD_HHMMSS.json`

### Prospective scoring

When a forecast window completes (7 or 30 days), the scoring engine evaluates it against actual observations:

```bash
cd bangladesh_eq_forecast
python3 -c "
import sys, importlib.util, json
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('ps', 'live/prospective_scoring.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# Score completed forecasts
m.score_completed_forecasts(events)
# Get summary
print(json.dumps(m.get_scoring_summary(), indent=2))
"
```

---

## How to Update Catalogs

### Automatic (live fetch)

The pipeline automatically fetches the latest USGS and ISC data at runtime:

```python
# In live/parallel_pipeline.py:
usgs_events = fetch_usgs(start, end)  # USGS FDSN API
isc_events = fetch_isc(start, end)    # ISC FDSN API
```

The fetched data is cached to:
- `data/raw/usgs/usgs_live_latest.csv`
- `data/raw/isc/isc_live_latest.txt`

### Manual (replace static files)

To use a different catalog snapshot:

1. Download USGS CSV from https://earthquake.usgs.gov/fdsnws/event/1/query
2. Download ISC text from http://www.isc.ac.uk/fdsnws/event/1/query
3. Replace the files in `data/raw/usgs/` and `data/raw/isc/`
4. Re-run the pipeline

### Catalog merge

The `build_canonical_events` function deduplicates USGS+ISC events within a 120-second / 50-km window:

```python
from src.ingestion import build_canonical_events, read_usgs_csv
from src.phase_c.isc_reader import read_isc_text

usgs = read_usgs_csv('data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv')
isc = read_isc_text('data/raw/isc/isc_bangladesh_1973_2025_m3.txt')
events = build_canonical_events(usgs + isc, time_window_s=120.0, spatial_window_km=50.0)
# Result: 5,779 canonical events
```

---

## How to Use the APIs

The dashboard exposes 10 API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/forecast` | GET | Current v1 forecast (4 configs, 64 cells each, with CIs) |
| `/api/prospective` | GET | Prospective scoring summary (evidence level, metrics) |
| `/api/validation` | GET | Extended validation metrics (Brier, ECE, precision, recall, F1) |
| `/api/data-health` | GET | Data source availability and forecast integrity |
| `/api/forecast-history` | GET | Immutable forecast ledger entries |
| `/api/v2-comparison` | GET | v1 vs v2 parallel comparison data |
| `/api/live-earthquakes` | GET | Live USGS+ISC feed (last 30 days) |
| `/api/catalog` | GET | Historical earthquake GeoJSON (4,034 events M≥4.0) |
| `/api/faults` | GET | GEM Global Active Faults Database (42 BD-region faults) |
| `/api/figures` | GET | Publication figure metadata (16 figures) |

### Example API calls

```bash
# Get current forecast
curl http://localhost:3000/api/forecast

# Get live earthquakes
curl http://localhost:3000/api/live-earthquakes?days=30

# Get validation metrics
curl http://localhost:3000/api/validation
```

### Triggering a forecast update

```bash
# POST to the update endpoint
curl -X POST http://localhost:3000/api/update
```

---

## How to Regenerate Forecasts

### Quick regeneration

```bash
# From the Next.js project root:
# 1. Start the dashboard
bun run dev

# 2. Click "Re-run Pipeline" in the dashboard header
# OR
# 3. Call the API directly:
curl -X POST http://localhost:3000/api/update
```

### Manual regeneration (Python)

```bash
cd bangladesh_eq_forecast
python3 live/parallel_pipeline.py
```

This will:
1. Fetch live USGS + ISC data (or use cached files if APIs are unavailable)
2. Merge and deduplicate the catalog
3. Generate v1 (Spatial Poisson) and v2 (Bayesian) forecasts for all 4 configs
4. Save to immutable ledgers with SHA-256 hashes
5. Update `latest_forecast.json` and `latest_parallel.json`

### Forecast configurations

| Config | Magnitude threshold | Horizon |
|--------|-------------------|---------|
| M4.5_7d | M ≥ 4.5 | 7 days |
| M4.5_30d | M ≥ 4.5 | 30 days |
| M5.0_7d | M ≥ 5.0 | 7 days |
| M5.0_30d | M ≥ 5.0 | 30 days |

---

## How to Reproduce the Paper

### Prerequisites

- Tectonic 0.15+ (for LaTeX compilation)
- The 16 figures in `outputs/figures/` (already at 300 DPI)

### Steps

1. **Regenerate figures** (optional — they are already at 300 DPI):

```bash
cd bangladesh_eq_forecast
python3 run_regenerate_figures.py
```

2. **Compile the paper:**

```bash
cd FINAL_PUBLICATION_PACKAGE
tectonic BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.tex
```

This produces `BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.pdf` (45 pages).

3. **Verify the PDF:**

```bash
pdfinfo BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.pdf
# Should show: Pages: 45, Author: Ujayer Hasnat
```

### Paper structure

The paper has 22 numbered sections + 10 appendices:
- Title page, Abstract, Keywords
- Introduction, Tectonic setting, Literature review
- Data, Catalog preprocessing, Mc estimation, Gutenberg-Richter
- Methodology, Spatial Poisson, ETAS, ML, Bayesian, Adaptive, Region-specific ETAS
- Validation framework, Retrospective evaluation, Prospective monitoring
- Results, Discussion, Limitations, Future work, Conclusion
- About the Author, References (26 citations)
- Appendices A–J (mathematical derivations, parameter estimation, PPC details, etc.)

---

## Project Structure

```
bangladesh_eq_forecast/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── AUDIT_REPORT.md                    # Scientific audit (PASS/WARNING/FAIL)
│
├── src/                               # Source code (frozen v1.0 model)
│   ├── ingestion/                     # Catalog ingestion & merging
│   ├── preprocessing/                 # Catalog QC and reporting
│   ├── completeness/                  # Mc estimation (MAXC, GFT, EMR, Stepp)
│   ├── declustering/                  # Gardner-Knopoff, Reasenberg
│   ├── baselines/                     # Spatial Poisson, uncertainty (Garwood)
│   ├── etas/                          # ETAS model (K≈0 finding)
│   ├── coulomb/                       # Coulomb stress (disabled)
│   ├── ml/                            # ML experiments + spatial Poisson baseline
│   ├── features.py                    # Feature engineering
│   ├── phase_b/                       # Validation experiments
│   ├── phase_c/                       # ISC catalog acquisition
│   └── result_status.py              # Result status helpers
│
├── live/                              # Live forecasting pipeline
│   ├── pipeline.py                    # Single-model (v1) pipeline
│   ├── parallel_pipeline.py           # Dual (v1+v2) pipeline
│   ├── prospective_scoring.py         # Scoring engine
│   ├── latest_forecast.json           # v1 snapshot (dashboard source)
│   ├── latest_parallel.json           # v1+v2 parallel snapshot
│   └── forecast_ledger/               # Immutable SHA-256-hashed ledgers
│       ├── v1/                        # Production forecasts
│       └── v2_bayesian/               # Candidate forecasts
│
├── data/
│   ├── raw/
│   │   ├── usgs/                      # USGS ComCat CSV
│   │   └── isc/                       # ISC Bulletin text
│   └── external/
│       └── gem_gafd.geojson           # GEM Global Active Faults
│
├── outputs/                           # All scientific outputs
│   ├── figures/                       # 16 publication figures (300 DPI)
│   ├── final_*.csv                    # Frozen v1.0 results
│   ├── v2_bayesian_*.csv              # v2 Bayesian results
│   ├── v3_adaptive_*.csv              # v3 Adaptive results
│   ├── v4_*.csv                       # v4 ETAS results
│   ├── stage{1-8}_*.md                # Stage reports
│   ├── V2_BAYESIAN_REPORT.md
│   ├── V3_ADAPTIVE_SPATIAL_REPORT.md
│   ├── V4_REGION_SPECIFIC_ETAS_REPORT.md
│   ├── FINAL_BANGLADESH_EARTHQUAKE_FORECASTING_REPORT.md
│   └── FIGURE_CAPTIONS.md
│
├── v2_candidates/bayesian_spatial/    # v2 candidate model
├── v3_candidates/adaptive_spatial/    # v3 candidate model (REJECTED)
├── v4_candidates/region_specific_etas/# v4 candidate model (REJECTED)
│
├── FINAL_PUBLICATION_PACKAGE/
│   ├── BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.pdf   # ← Final paper (45 pages)
│   ├── BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.tex   # LaTeX source
│   ├── BANGLADESH_EARTHQUAKE_FORECASTING_PAPER_V3.md    # Markdown source
│   ├── FINAL_EXECUTIVE_SUMMARY.md
│   ├── FINAL_PROVENANCE_MANIFEST.json
│   ├── REPRODUCIBILITY_GUIDE.md
│   └── figures/                       # Figure copies
│
├── run_stage{3,4,5,6,7,7b,8}.py       # Stage runners (frozen experiments)
├── run_phase_b.py                     # Phase B validation runner
├── run_phase_c.py                     # Phase C ISC acquisition
├── run_phase_d.py                     # Phase D revalidation
├── run_final.py                       # Final validation run
├── run_v2_experiment.py               # v2 Bayesian experiment
├── run_v2_reliability.py              # v2 reliability validation
├── run_v3_experiment.py               # v3 Adaptive experiment
├── run_v4_experiment.py               # v4 ETAS experiment
├── run_regenerate_figures.py          # Figure regeneration script
│
└── configs/
    ├── data_sources.yaml
    └── study_region.yaml
```

---

## Model Hierarchy

| Model | Version | Status | Brier (7d) | Verdict |
|-------|---------|--------|------------|---------|
| Spatial Poisson | FINAL_v1.0_FROZEN | **PRODUCTION** | 0.0242 | Validated |
| Bayesian Hierarchical | FINAL_v2.0_CANDIDATE | CANDIDATE | ΔBrier ≈ 0 | B (uncertainty improvement only) |
| Adaptive Spatial | FINAL_v3.0_CANDIDATE | REJECTED | PPC FAIL | D (over-concentrates seismicity) |
| Region-Specific ETAS | FINAL_v4.0_CANDIDATE | REJECTED | K≈0 | D (contradiction unresolved) |

### Scientific conclusion

Historical spatial seismicity rates provide the strongest validated probabilistic forecasting baseline for the available Bangladesh earthquake catalog. Under strict chronological, spatial, and spatiotemporal validation, the tested ETAS and machine-learning formulations did not demonstrate statistically defensible incremental predictive skill beyond this baseline.

The catalog exhibits strong short-lag post-mainshock temporal clustering (R≈24×), but this does not translate into ETAS forecast skill. The R≈24× / K≈0 contradiction is **not** caused by ETAS misspecification — it survives depth-stratification, depth-dependent spatial kernels, and modified temporal kernels.

---

## Audit and Integrity

A comprehensive scientific audit was performed (`AUDIT_REPORT.md`). Results:

| Check | Result |
|-------|--------|
| All probabilities in [0,1] | ✅ 256/256 PASS |
| All CIs correctly ordered | ✅ 256/256 PASS |
| Brier scores match frozen CSVs | ✅ 4/4 PASS |
| ECE values match frozen CSVs | ✅ 4/4 PASS |
| All 16 figures valid 300-DPI PNGs | ✅ 16/16 PASS |
| PDF 45 pages, correct numbers | ✅ PASS |
| SHA-256 ledger hashes verified | ✅ 3/3 PASS |
| v1.0_FROZEN source unchanged | ✅ PASS |

**Overall: PASS** (567/569 values PASS, 2 non-critical warnings).

### Integrity guarantees

- FINAL_v1.0_FROZEN source code: **unchanged** (frozen Aug 9, 2026)
- All forecast ledgers: **unchanged** (SHA-256 verified)
- No evaluation-period leakage
- No fabricated data or performance
- No deterministic earthquake predictions
- Every probability includes 95% uncertainty interval

---

## License and Citation

### Citation

```bibtex
@misc{hasnat2026bangladesh,
  author       = {Ujayer Hasnat},
  title        = {Probabilistic Earthquake Forecasting for Bangladesh:
                  A Spatial Poisson Baseline and the Failure of ETAS,
                  Bayesian Hierarchical, and Adaptive Smoothing Candidates},
  year         = {2026},
  howpublished = {Independent research},
  note         = {FINAL\_v1.0\_FROZEN production model}
}
```

### Contact

**Ujayer Hasnat** — Data Scientist, Data Analyst

---

*This platform provides probabilistic forecasts only. Not deterministic predictions. Not an emergency warning system.*
