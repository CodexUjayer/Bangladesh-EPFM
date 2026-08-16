# STAGE 3 DATA-ACQUISITION AUDIT — M>=4 vs M>=2.5 Catalogs

> Generated 2026-08-09T07:37:10.016405+00:00.

## 1. Audit objective

The original Stage 3 catalog was acquired with `minmagnitude=4.0` in the USGS FDSN query, producing 2,126 events hard-truncated at M=4.0. Mc=4.55 was estimated from that catalog. **The question is whether Mc=4.55 is a genuine completeness threshold or an artifact of truncating the catalog at M=4.0.** This audit acquires a lower-threshold catalog (M>=2.5) and re-runs all completeness estimators to distinguish:

  - **(A)** Mc estimated from a genuinely low-threshold catalog
  - **(B)** Mc estimated from a catalog already truncated near Mc

## 2. File-level audit

### M>=4 file

- Path: `/home/z/my-project/bangladesh_eq_forecast/data/raw/usgs/usgs_bangladesh_1973_2025_m4.csv`
- Events: **2,126**
- Magnitude range: **4.00 – 7.30**
- Events with M<4.0: **0**
- Events with M<3.5: **0**
- Events with M<3.0: **0**
- Events with M<2.5: **0**
- **HARD TRUNCATION: YES — the file contains ZERO events below M=4.0. This is a query artifact (`minmagnitude=4.0`), NOT a property of USGS ComCat.**

### M>=2.5 file (re-acquired for this audit)

- Path: `/home/z/my-project/bangladesh_eq_forecast/data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv`
- Events: **2,293**
- Magnitude range: **3.20 – 7.30**
- Events with M<4.0: **167**
- Events with M<3.5: **3**
- Events with M<3.0: **0**
- Events with M<2.5: **0**
- **HARD TRUNCATION: PARTIAL — the query requested M>=2.5 but USGS ComCat returned NOTHING below M=3.2. This is a genuine data limitation of USGS ComCat for this region (sparse seismographic network), NOT a query artifact. The catalog does NOT actually reach M2.5.**

### Low-end magnitude bin counts

| M bin | M>=4 file | M>=2.5 file |
|-------|-----------|-------------|
| 2.5 | 0 | 0 |
| 2.6 | 0 | 0 |
| 2.7 | 0 | 0 |
| 2.8 | 0 | 0 |
| 2.9 | 0 | 0 |
| 3.0 | 0 | 0 |
| 3.1 | 0 | 0 |
| 3.2 | 0 | 1  <-- truncation |
| 3.3 | 0 | 0 |
| 3.4 | 0 | 2  <-- truncation |
| 3.5 | 0 | 12  <-- truncation |
| 3.6 | 0 | 14  <-- truncation |
| 3.7 | 0 | 26  <-- truncation |
| 3.8 | 0 | 48  <-- truncation |
| 3.9 | 0 | 64  <-- truncation |
| 4.0 | 139 | 139 |
| 4.1 | 154 | 154 |
| 4.2 | 219 | 219 |
| 4.3 | 255 | 255 |
| 4.4 | 243 | 243 |
| 4.5 | 241 | 241 |
| 4.6 | 218 | 218 |
| 4.7 | 163 | 163 |
| 4.8 | 125 | 125 |
| 4.9 | 83 | 83 |
| 5.0 | 65 | 65 |
| 5.1 | 56 | 56 |

## 3. Completeness (Mc) — re-estimated on both catalogs

- Magnitude scale: **Mw (derived/authoritative; events with missing Mw excluded)** (same for both)

| Method | M>=4 catalog | M>=2.5 catalog |
|--------|--------------|----------------|
| MAXC | 4.65 | 4.65 |
| GFT (95%) | 4.95 | 4.95 |
| EMR | 4.45 | 4.55 |
| Stepp | 4.40 | 4.00 |
| **Recommended (median)** | **4.55** | **4.60** |
| Events above recommended Mc | 1,831 | 1,831 |
| Events below recommended Mc | 293 | 455 |

## 4. b-value — re-estimated on both catalogs

### b at each catalog's own recommended Mc

| | M>=4 catalog | M>=2.5 catalog |
|---|---|---|
| Mc used | 4.55 | 4.60 |
| b (MLE Aki-Utsu) | 1.069 | 1.115 |
| sigma_b (Shi-Bolt) | 0.019 | 0.021 |
| a-value | 8.160 | 8.394 |
| N events used | 1,985 | 1,831 |

### b at a FIXED Mc=4.5 (cross-catalog comparability)

| | M>=4 catalog | M>=2.5 catalog |
|---|---|---|
| b (MLE) | 0.951 | 0.951 |
| sigma_b | 0.015 | 0.015 |
| a-value | 7.579 | 7.579 |
| N events used | 1,985 | 1,985 |

## 5. Mc(t) — time-varying completeness (5-year rolling MAXC)

| Period | M>=4 Mc | M>=2.5 Mc | M>=4 N | M>=2.5 N |
|--------|---------|-----------|--------|----------|
| 1973-1977 | 5.15 | 5.15 | 61 | 64 |
| 1974-1978 | 5.15 | 5.15 | 72 | 75 |
| 1975-1979 | 5.15 | 5.15 | 84 | 85 |
| 1976-1980 | 5.15 | 5.15 | 90 | 91 |
| 1977-1981 | 4.95 | 4.95 | 101 | 101 |
| 1978-1982 | 4.95 | 4.95 | 124 | 124 |
| 1979-1983 | 5.15 | 5.15 | 119 | 120 |
| 1980-1984 | 5.15 | 5.15 | 119 | 121 |
| 1981-1985 | 5.15 | 5.15 | 121 | 124 |
| 1982-1986 | 5.15 | 5.15 | 123 | 126 |
| 1983-1987 | 5.15 | 5.15 | 120 | 124 |
| 1984-1988 | 5.15 | 5.15 | 145 | 148 |
| 1985-1989 | 5.15 | 5.15 | 155 | 159 |
| 1986-1990 | 5.15 | 5.15 | 166 | 171 |
| 1987-1991 | 5.15 | 5.15 | 188 | 196 |
| 1988-1992 | 4.65 | 4.65 | 206 | 218 |
| 1989-1993 | 4.65 | 4.65 | 199 | 212 |
| 1990-1994 | 4.65 | 4.65 | 196 | 208 |
| 1991-1995 | 4.65 | 4.65 | 210 | 225 |
| 1992-1996 | 4.65 | 4.65 | 225 | 247 |
| 1993-1997 | 4.65 | 4.65 | 217 | 236 |
| 1994-1998 | 4.65 | 4.65 | 211 | 236 |
| 1995-1999 | 4.65 | 4.65 | 211 | 238 |
| 1996-2000 | 4.65 | 4.65 | 200 | 229 |
| 1997-2001 | 4.65 | 4.65 | 180 | 206 |
| 1998-2002 | 4.65 | 4.65 | 167 | 198 |
| 1999-2003 | 4.65 | 4.65 | 170 | 199 |
| 2000-2004 | 4.65 | 4.65 | 190 | 224 |
| 2001-2005 | 4.65 | 4.65 | 200 | 239 |
| 2002-2006 | 4.65 | 4.65 | 207 | 253 |
| 2003-2007 | 4.65 | 4.65 | 223 | 278 |
| 2004-2008 | 4.65 | 4.65 | 245 | 318 |
| 2005-2009 | 4.65 | 4.65 | 235 | 300 |
| 2006-2010 | 4.65 | 4.65 | 213 | 266 |
| 2007-2011 | 4.65 | 4.65 | 220 | 259 |
| 2008-2012 | 4.65 | 4.65 | 245 | 268 |
| 2009-2013 | 4.65 | 4.65 | 237 | 238 |
| 2010-2014 | 4.65 | 4.65 | 273 | 275 |
| 2011-2015 | 4.65 | 4.65 | 312 | 315 |
| 2012-2016 | 4.65 | 4.65 | 314 | 319 |
| 2013-2017 | 4.65 | 4.65 | 306 | 311 |
| 2014-2018 | 4.65 | 4.65 | 310 | 316 |
| 2015-2019 | 4.65 | 4.65 | 287 | 295 |
| 2016-2020 | 4.65 | 4.65 | 301 | 310 |
| 2017-2021 | 4.65 | 4.65 | 330 | 340 |
| 2018-2022 | 4.65 | 4.65 | 332 | 342 |
| 2019-2023 | 4.65 | 4.65 | 343 | 351 |

## 6. Spatial Mc (MAXC per subregion)

| Region | M>=4 Mc | M>=4 N | M>=2.5 Mc | M>=2.5 N |
|--------|---------|--------|-----------|----------|
| arakan_megathrust | 4.65 | 500 | 4.65 | 536 |
| bangladesh_platform | 4.65 | 133 | 4.65 | 146 |
| chittagong_tripura_fold_belt | 4.65 | 54 | 4.65 | 60 |
| indo_burman_fold_belt | 4.65 | 1391 | 4.65 | 1481 |
| shillong_plateau | 4.65 | 156 | 4.65 | 168 |
| surrounding_himalaya | 4.65 | 302 | 4.65 | 341 |

## 7. Declustering (Gardner-Knopoff) comparison

| | M>=4 catalog | M>=2.5 catalog |
|---|---|---|
| Mainshocks | 202 | 213 |
| Aftershocks | 1,052 | 1,129 |
| Foreshocks | 872 | 951 |

## 8. Interpretation

### Was the M>=4 catalog hard-truncated?
**YES.** The M>=4 file contains 0 events below M=4.0. This was a query artifact (`minmagnitude=4.0`), now corrected by the M>=2.5 re-acquisition. The M>=4 file is preserved as a preliminary catalog but **Mc=4.55 derived from it must NOT be treated as a final, scientifically validated completeness threshold.**

### Does the M>=2.5 catalog reach M2.5?
**NO.** Although the USGS FDSN query requested `minmagnitude=2.5`, the returned catalog's minimum magnitude is **3.2**. There are **0 events below M3.0** and **0 events below M2.5**. This is a genuine data limitation: USGS ComCat does not hold M2.5-3.0 events for the Bangladesh region, because the regional seismographic network is sparse (the research report itself notes BMD relies on USGS/IMD and reports only ~100 small quakes/year, mostly M3-4.5, as local BMD detections that are not necessarily in USGS ComCat). **This cannot be fixed by lowering the query threshold.**

### Did Mc change between the two catalogs?
M>=4 recommended Mc = **4.55**; M>=2.5 recommended Mc = **4.60**. Absolute difference = 0.05 magnitude units. 
The estimates are close, but this does NOT validate Mc=4.55: the M>=2.5 catalog only adds 167 events in [3.2, 4.0), which is too few to robustly resolve the completeness rolloff below ~M3.5. The agreement between estimators reflects the fact that BOTH catalogs share the same effective floor (~M3.2-3.5) imposed by USGS ComCat coverage, not that the true Mc has been independently confirmed.

### What is the honest conclusion?
**Neither Mc=4.55 (M>=4) nor the M>=2.5 estimate is a fully validated completeness threshold for the Bangladesh region.** Both are constrained by the fact that USGS ComCat is itself effectively complete only down to ~M3.2-3.5 in this region. To genuinely characterize Mc below ~M3.5, the system needs:

  1. **BMD local network bulletins** (M2-3 events detected locally but not in USGS ComCat) — currently Class D (not obtained).
  2. **ISC bulletin** (aggregates more small events from contributing regional agencies) — not reachable in this environment; accepts local CSV.
  3. **A regional catalog from published literature** (e.g., Haque et al. 2020; Rahman et al. 2020) — requires manual acquisition.

### What does this mean for Stage 4?
- The **M>=2.5 catalog (2,293 events, floor M3.2)** is the better preliminary input and should be used going forward in place of the M>=4 file.
- For Poisson/Gutenberg-Richter baselines (Stage 4) and ETAS (Stage 5), the catalog should be **filtered to a conservative working threshold of M>=4.5** (where USGS ComCat is robustly complete in this region, per MAXC and the FMD peak), with the explicit caveat that the **true Mc may be as low as ~M3.5** and cannot be confirmed without BMD/ISC data. Stepp's method on the M>=2.5 catalog gave Mc=4.00, suggesting the rolloff begins around M3.5-4.0, but the 167 events in [3.2,4.0) are too few to resolve it robustly.
- **Mc must be reported as a range / working threshold (M3.5-4.5), NOT a single validated number**, until a genuinely low-threshold catalog (BMD or ISC) is incorporated.
- The M>=4 file is PRESERVED as `usgs_bangladesh_1973_2025_m4.csv` (preliminary); it is not deleted.

## 9. Required actions before Stage 4 model fitting

1. **Replace** the working catalog with the M>=2.5 file (`usgs_bangladesh_1973_2025_m25.csv`).
2. **Report Mc as a working range (M3.5-4.5)** and use a conservative M>=4.5 filter for model fitting, rather than a single validated completeness magnitude, until BMD/ISC data arrive.
3. **Stage 4 code architecture may proceed** (Poisson, GR, ETAS implementations), but model fitting on the real catalog must carry the completeness caveat above.
4. **When BMD or ISC local files are supplied**, re-run this audit. If they extend below M3.0, the Mc will become genuinely estimable and the caveat can be relaxed.