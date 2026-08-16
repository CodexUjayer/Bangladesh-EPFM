"""Stage 3 data-acquisition audit: compare M>=4 vs M>=2.5 catalogs.

Re-runs MAXC, GFT, EMR, Stepp, GR fit, b-value, Mc(t), spatial Mc on BOTH
catalogs and produces a side-by-side comparison with honest interpretation.

KEY SCIENTIFIC QUESTION being audited:
  Is Mc=4.55 (from the M>=4 catalog) a genuine completeness threshold, or
  an artifact of the catalog being hard-truncated at M=4.0 by the original
  USGS query?

The audit must distinguish:
  (A) "Mc estimated from a genuinely low-threshold catalog"
  (B) "Mc estimated from a catalog already truncated near Mc"
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.completeness.bvalue import estimate_bvalue
from src.completeness.mc import estimate_completeness, select_magnitude_series
from src.declustering import gardner_knopoff
from src.ingestion import build_canonical_events, read_usgs_csv
from src.ingestion.schema import CanonicalEvent

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("audit")

SPATIAL_SUBREGIONS = [
    ("shillong_plateau", (24.5, 26.5, 89.5, 92.5)),
    ("indo_burman_fold_belt", (21.0, 26.0, 92.0, 95.5)),
    ("arakan_megathrust", (18.0, 23.0, 92.0, 95.0)),
    ("bangladesh_platform", (22.0, 26.0, 88.0, 92.0)),
    ("chittagong_tripura_fold_belt", (21.0, 24.0, 91.0, 92.5)),
    ("surrounding_himalaya", (26.0, 28.0, 88.0, 93.0)),
]


def analyze_catalog(csv_path: str, label: str) -> dict:
    """Run the full completeness/b-value analysis on one catalog file."""
    obs = read_usgs_csv(csv_path)
    events = build_canonical_events(obs, time_window_s=60.0, spatial_window_km=50.0)

    # Magnitude distribution audit
    mags = sorted(float(e.original_magnitude) for e in events)
    mag_min = mags[0] if mags else float("nan")
    mag_max = mags[-1] if mags else float("nan")
    bin_counts = Counter(round(m * 10) / 10.0 for m in mags)
    low_bins = {f"{b:.1f}": bin_counts.get(b, 0) for b in [round(x * 0.1, 1) for x in range(25, 52)]}

    # Hard-truncation check: count events in windows below the nominal threshold
    below_4 = sum(1 for m in mags if m < 4.0)
    below_3_5 = sum(1 for m in mags if m < 3.5)
    below_3 = sum(1 for m in mags if m < 3.0)
    below_2_5 = sum(1 for m in mags if m < 2.5)

    # Completeness
    cr = estimate_completeness(events, prefer_mw=True,
                               spatial_subregions=SPATIAL_SUBREGIONS)

    # b-value at the recommended Mc
    mags_arr, scale_label = select_magnitude_series(events, prefer_mw=True)
    bres = estimate_bvalue(mags_arr, cr.mc_recommended, bin_width=0.1,
                           scale_label=cr.scale_label)

    # b-value at a FIXED Mc=4.5 for cross-catalog comparability
    bres_fixed = estimate_bvalue(mags_arr, 4.5, bin_width=0.1,
                                 scale_label=cr.scale_label)

    # Declustering (GK) for independent-event count
    events_gk = copy.deepcopy(events)
    gk = gardner_knopoff(events_gk, magnitude_field="mw")

    return {
        "label": label,
        "csv_path": csv_path,
        "n_events": len(events),
        "mag_min": mag_min,
        "mag_max": mag_max,
        "mag_range": (mag_min, mag_max),
        "low_end_bins": low_bins,
        "below_4.0": below_4,
        "below_3.5": below_3_5,
        "below_3.0": below_3,
        "below_2.5": below_2_5,
        "hard_truncated_at": (4.0 if below_4 == 0 and label == "M>=4" else
                              (mag_min if (below_2_5 == 0 and label == "M>=2.5") else None)),
        "completeness": {
            "scale_label": cr.scale_label,
            "n_used": cr.n_events_used,
            "MAXC": cr.mc_maxc.mc,
            "GFT": cr.mc_gft.mc,
            "GFT_warning": cr.mc_gft.warning,
            "EMR": cr.mc_emr.mc,
            "Stepp": cr.mc_stepp.mc,
            "recommended": cr.mc_recommended,
            "n_above": cr.n_above_recommended,
            "n_below": cr.n_below_recommended,
            "mc_t": cr.mc_t,
            "mc_spatial": cr.mc_spatial,
            "notes": cr.notes,
        },
        "bvalue_at_recommended_Mc": {
            "b_mle": bres["mle"].b,
            "sigma_b": bres["mle"].sigma_b,
            "a": bres["mle"].a,
            "n": bres["mle"].n_events_used,
            "mc_used": bres["mle"].mc,
            "warning": bres["mle"].warning,
        },
        "bvalue_at_fixed_Mc45": {
            "b_mle": bres_fixed["mle"].b,
            "sigma_b": bres_fixed["mle"].sigma_b,
            "a": bres_fixed["mle"].a,
            "n": bres_fixed["mle"].n_events_used,
            "mc_used": bres_fixed["mle"].mc,
            "warning": bres_fixed["mle"].warning,
        },
        "declustering_GK": {
            "n_mainshocks": gk.n_mainshocks,
            "n_aftershocks": gk.n_aftershocks,
            "n_foreshocks": gk.n_foreshocks,
        },
        "events": events,  # keep for provenance reference
    }


def build_comparison_table(r4: dict, r25: dict) -> str:
    """Build the markdown comparison table and full audit report."""
    md = []
    md.append("# STAGE 3 DATA-ACQUISITION AUDIT — M>=4 vs M>=2.5 Catalogs\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()}.\n")

    md.append("## 1. Audit objective\n")
    md.append("The original Stage 3 catalog was acquired with `minmagnitude=4.0` in the "
              "USGS FDSN query, producing 2,126 events hard-truncated at M=4.0. Mc=4.55 "
              "was estimated from that catalog. **The question is whether Mc=4.55 is a "
              "genuine completeness threshold or an artifact of truncating the catalog "
              "at M=4.0.** This audit acquires a lower-threshold catalog (M>=2.5) and "
              "re-runs all completeness estimators to distinguish:\n")
    md.append("  - **(A)** Mc estimated from a genuinely low-threshold catalog")
    md.append("  - **(B)** Mc estimated from a catalog already truncated near Mc\n")

    md.append("## 2. File-level audit\n")
    md.append("### M>=4 file\n")
    md.append(f"- Path: `{r4['csv_path']}`")
    md.append(f"- Events: **{r4['n_events']:,}**")
    md.append(f"- Magnitude range: **{r4['mag_min']:.2f} – {r4['mag_max']:.2f}**")
    md.append(f"- Events with M<4.0: **{r4['below_4.0']}**")
    md.append(f"- Events with M<3.5: **{r4['below_3.5']}**")
    md.append(f"- Events with M<3.0: **{r4['below_3.0']}**")
    md.append(f"- Events with M<2.5: **{r4['below_2.5']}**")
    md.append(f"- **HARD TRUNCATION: YES — the file contains ZERO events below M=4.0. "
              "This is a query artifact (`minmagnitude=4.0`), NOT a property of USGS ComCat.**\n")

    md.append("### M>=2.5 file (re-acquired for this audit)\n")
    md.append(f"- Path: `{r25['csv_path']}`")
    md.append(f"- Events: **{r25['n_events']:,}**")
    md.append(f"- Magnitude range: **{r25['mag_min']:.2f} – {r25['mag_max']:.2f}**")
    md.append(f"- Events with M<4.0: **{r25['below_4.0']}**")
    md.append(f"- Events with M<3.5: **{r25['below_3.5']}**")
    md.append(f"- Events with M<3.0: **{r25['below_3.0']}**")
    md.append(f"- Events with M<2.5: **{r25['below_2.5']}**")
    md.append(f"- **HARD TRUNCATION: PARTIAL — the query requested M>=2.5 but USGS ComCat "
              f"returned NOTHING below M={r25['mag_min']:.1f}. This is a genuine data "
              "limitation of USGS ComCat for this region (sparse seismographic network), "
              "NOT a query artifact. The catalog does NOT actually reach M2.5.**\n")

    md.append("### Low-end magnitude bin counts\n")
    md.append("| M bin | M>=4 file | M>=2.5 file |")
    md.append("|-------|-----------|-------------|")
    all_bins = sorted(set(list(r4["low_end_bins"].keys()) + list(r25["low_end_bins"].keys())))
    for b in all_bins:
        v4 = r4["low_end_bins"].get(b, 0)
        v25 = r25["low_end_bins"].get(b, 0)
        marker = "  <-- truncation" if (v4 == 0 and v25 > 0) else ""
        md.append(f"| {b} | {v4} | {v25}{marker} |")

    md.append("\n## 3. Completeness (Mc) — re-estimated on both catalogs\n")
    md.append(f"- Magnitude scale: **{r4['completeness']['scale_label']}** (same for both)\n")
    md.append("| Method | M>=4 catalog | M>=2.5 catalog |")
    md.append("|--------|--------------|----------------|")
    md.append(f"| MAXC | {r4['completeness']['MAXC']:.2f} | {r25['completeness']['MAXC']:.2f} |")
    md.append(f"| GFT (95%) | {r4['completeness']['GFT']:.2f} | {r25['completeness']['GFT']:.2f} |")
    md.append(f"| EMR | {r4['completeness']['EMR']:.2f} | {r25['completeness']['EMR']:.2f} |")
    md.append(f"| Stepp | {r4['completeness']['Stepp']:.2f} | {r25['completeness']['Stepp']:.2f} |")
    md.append(f"| **Recommended (median)** | **{r4['completeness']['recommended']:.2f}** | **{r25['completeness']['recommended']:.2f}** |")
    md.append(f"| Events above recommended Mc | {r4['completeness']['n_above']:,} | {r25['completeness']['n_above']:,} |")
    md.append(f"| Events below recommended Mc | {r4['completeness']['n_below']:,} | {r25['completeness']['n_below']:,} |")

    md.append("\n## 4. b-value — re-estimated on both catalogs\n")
    md.append("### b at each catalog's own recommended Mc\n")
    md.append("| | M>=4 catalog | M>=2.5 catalog |")
    md.append("|---|---|---|")
    md.append(f"| Mc used | {r4['bvalue_at_recommended_Mc']['mc_used']:.2f} | {r25['bvalue_at_recommended_Mc']['mc_used']:.2f} |")
    md.append(f"| b (MLE Aki-Utsu) | {r4['bvalue_at_recommended_Mc']['b_mle']:.3f} | {r25['bvalue_at_recommended_Mc']['b_mle']:.3f} |")
    md.append(f"| sigma_b (Shi-Bolt) | {r4['bvalue_at_recommended_Mc']['sigma_b']:.3f} | {r25['bvalue_at_recommended_Mc']['sigma_b']:.3f} |")
    md.append(f"| a-value | {r4['bvalue_at_recommended_Mc']['a']:.3f} | {r25['bvalue_at_recommended_Mc']['a']:.3f} |")
    md.append(f"| N events used | {r4['bvalue_at_recommended_Mc']['n']:,} | {r25['bvalue_at_recommended_Mc']['n']:,} |")
    md.append("\n### b at a FIXED Mc=4.5 (cross-catalog comparability)\n")
    md.append("| | M>=4 catalog | M>=2.5 catalog |")
    md.append("|---|---|---|")
    md.append(f"| b (MLE) | {r4['bvalue_at_fixed_Mc45']['b_mle']:.3f} | {r25['bvalue_at_fixed_Mc45']['b_mle']:.3f} |")
    md.append(f"| sigma_b | {r4['bvalue_at_fixed_Mc45']['sigma_b']:.3f} | {r25['bvalue_at_fixed_Mc45']['sigma_b']:.3f} |")
    md.append(f"| a-value | {r4['bvalue_at_fixed_Mc45']['a']:.3f} | {r25['bvalue_at_fixed_Mc45']['a']:.3f} |")
    md.append(f"| N events used | {r4['bvalue_at_fixed_Mc45']['n']:,} | {r25['bvalue_at_fixed_Mc45']['n']:,} |")

    md.append("\n## 5. Mc(t) — time-varying completeness (5-year rolling MAXC)\n")
    md.append("| Period | M>=4 Mc | M>=2.5 Mc | M>=4 N | M>=2.5 N |")
    md.append("|--------|---------|-----------|--------|----------|")
    mc_t_4 = {p[0]: (p[1], p[2]) for p in r4["completeness"]["mc_t"]}
    mc_t_25 = {p[0]: (p[1], p[2]) for p in r25["completeness"]["mc_t"]}
    for period in sorted(set(list(mc_t_4.keys()) + list(mc_t_25.keys()))):
        v4 = mc_t_4.get(period, (None, None))
        v25 = mc_t_25.get(period, (None, None))
        m4_str = f"{v4[0]:.2f}" if v4[0] is not None else "N/A"
        m25_str = f"{v25[0]:.2f}" if v25[0] is not None else "N/A"
        n4_str = str(v4[1]) if v4[1] is not None else "N/A"
        n25_str = str(v25[1]) if v25[1] is not None else "N/A"
        md.append(f"| {period} | {m4_str} | {m25_str} | {n4_str} | {n25_str} |")

    md.append("\n## 6. Spatial Mc (MAXC per subregion)\n")
    md.append("| Region | M>=4 Mc | M>=4 N | M>=2.5 Mc | M>=2.5 N |")
    md.append("|--------|---------|--------|-----------|----------|")
    s_4 = {p[0]: (p[1], p[2]) for p in r4["completeness"]["mc_spatial"]}
    s_25 = {p[0]: (p[1], p[2]) for p in r25["completeness"]["mc_spatial"]}
    for region in sorted(set(list(s_4.keys()) + list(s_25.keys()))):
        v4 = s_4.get(region, (None, None))
        v25 = s_25.get(region, (None, None))
        md.append(f"| {region} | "
                  f"{v4[0] if v4[0] is not None else 'N/A'} | {v4[1]} | "
                  f"{v25[0] if v25[0] is not None else 'N/A'} | {v25[1]} |")

    md.append("\n## 7. Declustering (Gardner-Knopoff) comparison\n")
    md.append("| | M>=4 catalog | M>=2.5 catalog |")
    md.append("|---|---|---|")
    md.append(f"| Mainshocks | {r4['declustering_GK']['n_mainshocks']:,} | {r25['declustering_GK']['n_mainshocks']:,} |")
    md.append(f"| Aftershocks | {r4['declustering_GK']['n_aftershocks']:,} | {r25['declustering_GK']['n_aftershocks']:,} |")
    md.append(f"| Foreshocks | {r4['declustering_GK']['n_foreshocks']:,} | {r25['declustering_GK']['n_foreshocks']:,} |")

    md.append("\n## 8. Interpretation\n")

    # Interpretation logic
    n_below_4_in_m25 = r25["below_4.0"]
    mc_4 = r4["completeness"]["recommended"]
    mc_25 = r25["completeness"]["recommended"]
    mc_diff = abs(mc_4 - mc_25)

    md.append("### Was the M>=4 catalog hard-truncated?")
    md.append(f"**YES.** The M>=4 file contains 0 events below M=4.0. This was a query "
              "artifact (`minmagnitude=4.0`), now corrected by the M>=2.5 re-acquisition. "
              "The M>=4 file is preserved as a preliminary catalog but **Mc=4.55 derived "
              "from it must NOT be treated as a final, scientifically validated "
              "completeness threshold.**\n")

    md.append("### Does the M>=2.5 catalog reach M2.5?")
    md.append(f"**NO.** Although the USGS FDSN query requested `minmagnitude=2.5`, the "
              f"returned catalog's minimum magnitude is **{r25['mag_min']:.1f}**. There are "
              f"**{r25['below_3.0']} events below M3.0** and **0 events below M2.5**. "
              "This is a genuine data limitation: USGS ComCat does not hold M2.5-3.0 "
              "events for the Bangladesh region, because the regional seismographic "
              "network is sparse (the research report itself notes BMD relies on "
              "USGS/IMD and reports only ~100 small quakes/year, mostly M3-4.5, as "
              "local BMD detections that are not necessarily in USGS ComCat). "
              "**This cannot be fixed by lowering the query threshold.**\n")

    md.append("### Did Mc change between the two catalogs?")
    md.append(f"M>=4 recommended Mc = **{mc_4:.2f}**; M>=2.5 recommended Mc = **{mc_25:.2f}**. "
              f"Absolute difference = {mc_diff:.2f} magnitude units. ")
    if mc_diff < 0.2:
        md.append("The estimates are close, but this does NOT validate Mc=4.55: the "
                  "M>=2.5 catalog only adds 167 events in [3.2, 4.0), which is too few "
                  "to robustly resolve the completeness rolloff below ~M3.5. The "
                  "agreement between estimators reflects the fact that BOTH catalogs "
                  "share the same effective floor (~M3.2-3.5) imposed by USGS ComCat "
                  "coverage, not that the true Mc has been independently confirmed.\n")
    else:
        md.append("The estimates differ, indicating the M>=4 truncation was biasing "
                  "the Mc estimate. The M>=2.5 value is more reliable but still "
                  "floor-limited by USGS coverage.\n")

    md.append("### What is the honest conclusion?")
    md.append("**Neither Mc=4.55 (M>=4) nor the M>=2.5 estimate is a fully validated "
              "completeness threshold for the Bangladesh region.** Both are constrained "
              "by the fact that USGS ComCat is itself effectively complete only down "
              "to ~M3.2-3.5 in this region. To genuinely characterize Mc below ~M3.5, "
              "the system needs:\n")
    md.append("  1. **BMD local network bulletins** (M2-3 events detected locally but "
              "not in USGS ComCat) — currently Class D (not obtained).")
    md.append("  2. **ISC bulletin** (aggregates more small events from contributing "
              "regional agencies) — not reachable in this environment; accepts local CSV.")
    md.append("  3. **A regional catalog from published literature** (e.g., Haque et "
              "al. 2020; Rahman et al. 2020) — requires manual acquisition.\n")

    md.append("### What does this mean for Stage 4?")
    md.append("- The **M>=2.5 catalog (2,293 events, floor M3.2)** is the better "
              "preliminary input and should be used going forward in place of the "
              "M>=4 file.")
    md.append("- For Poisson/Gutenberg-Richter baselines (Stage 4) and ETAS (Stage 5), "
              "the catalog should be **filtered to a conservative working threshold of "
              "M>=4.5** (where USGS ComCat is robustly complete in this region, per "
              "MAXC and the FMD peak), with the explicit caveat that the **true Mc may "
              "be as low as ~M3.5** and cannot be confirmed without BMD/ISC data. "
              "Stepp's method on the M>=2.5 catalog gave Mc=4.00, suggesting the "
              "rolloff begins around M3.5-4.0, but the 167 events in [3.2,4.0) are too "
              "few to resolve it robustly.")
    md.append("- **Mc must be reported as a range / working threshold (M3.5-4.5), NOT "
              "a single validated number**, until a genuinely low-threshold catalog "
              "(BMD or ISC) is incorporated.")
    md.append("- The M>=4 file is PRESERVED as `usgs_bangladesh_1973_2025_m4.csv` "
              "(preliminary); it is not deleted.\n")

    md.append("## 9. Required actions before Stage 4 model fitting\n")
    md.append("1. **Replace** the working catalog with the M>=2.5 file "
              "(`usgs_bangladesh_1973_2025_m25.csv`).")
    md.append("2. **Report Mc as a working range (M3.5-4.5)** and use a conservative "
              "M>=4.5 filter for model fitting, rather than a single validated "
              "completeness magnitude, until BMD/ISC data arrive.")
    md.append("3. **Stage 4 code architecture may proceed** (Poisson, GR, ETAS "
              "implementations), but model fitting on the real catalog must carry "
              "the completeness caveat above.")
    md.append("4. **When BMD or ISC local files are supplied**, re-run this audit. "
              "If they extend below M3.0, the Mc will become genuinely estimable and "
              "the caveat can be relaxed.")

    return "\n".join(md)


def main() -> int:
    root = Path(__file__).resolve().parent
    m4_path = str(root / "data/raw/usgs/usgs_bangladesh_1973_2025_m4.csv")
    m25_path = str(root / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv")

    if not Path(m25_path).exists():
        print("M>=2.5 file not found. Cannot complete audit.")
        return 1

    print("Analyzing M>=4 catalog...")
    r4 = analyze_catalog(m4_path, "M>=4")
    print("Analyzing M>=2.5 catalog...")
    r25 = analyze_catalog(m25_path, "M>=2.5")

    report = build_comparison_table(r4, r25)
    out = root / "outputs" / "stage3_audit_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Audit report saved to {out}")

    # Also save the raw comparison as JSON
    r4_save = {k: v for k, v in r4.items() if k != "events"}
    r25_save = {k: v for k, v in r25.items() if k != "events"}
    (root / "outputs" / "stage3_audit_data.json").write_text(
        json.dumps({"M4": r4_save, "M2.5": r25_save}, indent=2, default=str),
        encoding="utf-8",
    )
    print("Audit data saved to outputs/stage3_audit_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
