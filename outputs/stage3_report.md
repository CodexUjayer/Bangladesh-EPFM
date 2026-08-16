# STAGE 3 — Catalog Preprocessing, Completeness & Declustering Report

> Generated 2026-08-09T07:28:58.898457+00:00 from ACTUAL ingested catalog files. No fabricated numbers. Missingness reported explicitly.

## 1. Data sources loaded

- **usgs** — 2,126 observations — `/home/z/my-project/bangladesh_eq_forecast/data/raw/usgs/usgs_bangladesh_1973_2025_m4.csv`

### Not loaded (missingness)

- **gcmt** — NOT LOADED (no local file supplied)
- **isc-gem** — NOT LOADED (no local file supplied)

## 2. Catalog overlap & duplicate rate

- Total observations ingested: **2,126**
- Canonical events after matching: **2,126**
- Distinct source catalogs: **1**
- Multi-source events (>=2 catalogs): **0** (overlap fraction 0.000)
- Mean observations per event: **1.000**
- Per-source observation counts:
  - usgs: 2,126

- Within-source duplicate candidates (tight 30s / 10km window):
  - usgs: 0 duplicate pairs
- Within-source duplicate rate: **0.0000**
- _Single-source catalog: cross-source overlap not applicable._

## 3. Usable temporal coverage

- Time range: **1973-02-10T04:25:29.700000+00:00** → **2024-12-20T13:13:08.914000+00:00**
- Span: **51.9 years**
- Distinct years with events: **52**
- Events per decade:
  - 1970s: 113
  - 1980s: 274
  - 1990s: 407
  - 2000s: 426
  - 2010s: 560
  - 2020s: 346

## 4. Spatial coverage

- Latitude range: **20.02** → **27.99**
- Longitude range: **88.00** → **96.00**
- Depth range: **0.6** → **184.8** km
- Depth mean / median: **63.4** / **57.6** km

## 5. Magnitude distribution (original, as reported)

- Original magnitude range: **4.00** → **7.30**
- Histogram (0.1-unit bins, top 15):
  - M≈4.0: 139
  - M≈4.1: 154
  - M≈4.2: 219
  - M≈4.3: 255
  - M≈4.4: 243
  - M≈4.5: 241
  - M≈4.6: 218
  - M≈4.7: 163
  - M≈4.8: 125
  - M≈4.9: 83
  - M≈5.0: 65
  - M≈5.1: 56
  - M≈5.2: 43
  - M≈5.3: 28
  - M≈5.4: 16

- Mw available: **2,124** / 2,126 events (99.9%)
- Mw MISSING: **2**
- Mw-missing reasons (count of events):
  - 1× mb=6.5 is outside the validity range of published mb->Mw relations (3.5-6.2); Mw left missing.
  - 1× No validated global ml->Mw relation exists; no Bangladesh-specific relation is published. Mw left mi

## 6. Magnitude-type distribution (original)

| Type | Count | Fraction |
|------|-------|----------|
| mb | 1,963 | 92.3% |
| mwc | 51 | 2.4% |
| mww | 45 | 2.1% |
| mw | 39 | 1.8% |
| mwb | 20 | 0.9% |
| ms | 5 | 0.2% |
| mwr | 2 | 0.1% |
| ml | 1 | 0.0% |

## 7. Magnitude of completeness (Mc)

- Magnitude scale used: **Mw (derived/authoritative; events with missing Mw excluded)**
- Events used: **2,124**
- MAXC: **4.65** ± 0.05 
- GFT (95%): **4.95** ± 0.05 (No Mc reached confidence 0.95; reporting best R=0.935.)
- EMR: **4.45** 
- Stepp: **4.40** ± 0.05 

- **Recommended Mc: 4.55** (method: median(MAXC,GFT,EMR,Stepp))
  - Rationale: Median of 4 finite estimates: MAXC=4.65, GFT=4.95, EMR=4.45, Stepp=4.40.
  - Events above recommended Mc: **1,831**
  - Events below recommended Mc: **293** (excluded from Mw-based rate/b-value/ETAS estimation)
  - _NOTE: One or more Mc methods reported warnings (see fields)._

## 8. Mc(t) — time-varying completeness (MAXC, 5-year rolling)

| Period | Mc (MAXC) | N events |
|--------|-----------|----------|
| 1973-1977 | 5.15 | 61 |
| 1974-1978 | 5.15 | 72 |
| 1975-1979 | 5.15 | 84 |
| 1976-1980 | 5.15 | 90 |
| 1977-1981 | 4.95 | 101 |
| 1978-1982 | 4.95 | 124 |
| 1979-1983 | 5.15 | 119 |
| 1980-1984 | 5.15 | 119 |
| 1981-1985 | 5.15 | 121 |
| 1982-1986 | 5.15 | 123 |
| 1983-1987 | 5.15 | 120 |
| 1984-1988 | 5.15 | 145 |
| 1985-1989 | 5.15 | 155 |
| 1986-1990 | 5.15 | 166 |
| 1987-1991 | 5.15 | 188 |
| 1988-1992 | 4.65 | 206 |
| 1989-1993 | 4.65 | 199 |
| 1990-1994 | 4.65 | 196 |
| 1991-1995 | 4.65 | 210 |
| 1992-1996 | 4.65 | 225 |
| 1993-1997 | 4.65 | 217 |
| 1994-1998 | 4.65 | 211 |
| 1995-1999 | 4.65 | 211 |
| 1996-2000 | 4.65 | 200 |
| 1997-2001 | 4.65 | 180 |
| 1998-2002 | 4.65 | 167 |
| 1999-2003 | 4.65 | 170 |
| 2000-2004 | 4.65 | 190 |
| 2001-2005 | 4.65 | 200 |
| 2002-2006 | 4.65 | 207 |
| 2003-2007 | 4.65 | 223 |
| 2004-2008 | 4.65 | 245 |
| 2005-2009 | 4.65 | 235 |
| 2006-2010 | 4.65 | 213 |
| 2007-2011 | 4.65 | 220 |
| 2008-2012 | 4.65 | 245 |
| 2009-2013 | 4.65 | 237 |
| 2010-2014 | 4.65 | 273 |
| 2011-2015 | 4.65 | 312 |
| 2012-2016 | 4.65 | 314 |
| 2013-2017 | 4.65 | 306 |
| 2014-2018 | 4.65 | 310 |
| 2015-2019 | 4.65 | 287 |
| 2016-2020 | 4.65 | 301 |
| 2017-2021 | 4.65 | 330 |
| 2018-2022 | 4.65 | 332 |
| 2019-2023 | 4.65 | 343 |

## 9. Spatial Mc (MAXC per subregion)

| Region | Mc (MAXC) | N events |
|--------|-----------|----------|
| shillong_plateau | 4.65 | 156 |
| indo_burman_fold_belt | 4.65 | 1391 |
| arakan_megathrust | 4.65 | 500 |
| bangladesh_platform | 4.65 | 133 |
| chittagong_tripura_fold_belt | 4.65 | 54 |
| surrounding_himalaya | 4.65 | 302 |

## 10. Gutenberg-Richter b-value

- Magnitude scale: **Mw (derived/authoritative; events with missing Mw excluded)**
- Mc used: **4.55**
- **MLE (Aki-Utsu) b = 1.069 ± 0.019** (Shi-Bolt; N=1985)
- Cross-check b/sqrt(N): 0.024
- LS (log10 cumulative) b = 1.203 ± 0.131 (LS b-value is biased for binned data; MLE preferred.)
- a-value (MLE, at Mc): **8.160**

## 11. Declustering results

### Gardner-Knopoff (Knopoff 2000 windows; global, no Bangladesh adjustment)

- Total events: **2,126**
- Mainshocks (independent): **202**
- Aftershocks: **1,052**
- Foreshocks: **872**
- Clusters: **202**
- Independent fraction: **0.095**

### Reasenberg (1985; Wells & Coppersmith 1994 radii)

- Total events: **2,126**
- Mainshocks: **1,603**
- Aftershocks: **461**
- Foreshocks: **62**
- Clusters: **1,603**
- Independent fraction: **0.754**

## 12. ETAS sufficiency assessment

- Events above recommended Mc: **1,831**
- Mainshocks above Mc (GK): **202**
- Clusters with >=1 aftershock: **130**
- Temporal span: **51.9 years**
- Approx. mainshock rate above Mc: **35.308 / yr**

- **ASSESSMENT: Catalog appears SUFFICIENT for ETAS fitting (Stage 5).**

- _Note: ETAS parameter stability (especially α, the productivity exponent) requires a reasonable number of aftershock sequences. If insufficient locally, Stage 5 will use hierarchical Bayesian priors from analogous regions, clearly labeled as externally informed._

## 13. Provenance summary

Every canonical event carries a full provenance trail. Example (first event):
```
  acquired_local_file: Read 2126 observations from /home/z/my-project/bangladesh_eq_forecast/data/raw/usgs/usgs_b
  canonical_matched: Matched 2126 observations into 2126 canonical events (time window +/- 60.0s, spatial windo
  origin_selected: Origin chosen from usgs (rule: min(_QUALITY_RANK[reviewed], horizontal_unc=None, n_station
  magnitude_selected: Magnitude chosen from usgs; original type mb; Mw status: converted.
  mw_derived: Derived Mw via scordilis2006_mb_to_mw (Scordilis (2006), J. Seismology 10, 225-236); statu
```
Each derived Mw records conversion_method, conversion_source, conversion_uncertainty, and validity_range. Events with missing Mw record the reason in a `mw_left_missing` provenance step.

## 14. Assumptions & limitations

- Magnitude scale: Mw (derived/authoritative; events with missing Mw excluded)
- GCMT focal mechanisms: NOT LOADED (no local NDK supplied)
- ISC-GEM historical anchor: NOT LOADED (no local CSV supplied)
- Declustering window relations are GLOBAL (Knopoff 2000; Wells & Coppersmith 1994); no Bangladesh-specific adjustment is published.
- Mc methods are statistical; the recommended Mc is the median of 4 methods and should be sanity-checked against the catalog's instrumentation history.
- Numbers in this report come ONLY from the actual ingested files listed in Section 1.