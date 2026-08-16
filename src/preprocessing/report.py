"""Stage 3 empirical report generator.

Produces a markdown report with ACTUAL numbers from the ingested catalog.
No fabricated values. Missingness is reported explicitly.

Sections:
  1. Data sources loaded (and which were missing)
  2. Catalog overlap & duplicate rate
  3. Usable temporal coverage
  4. Spatial coverage
  5. Magnitude distributions
  6. Magnitude-type distributions
  7. Magnitude of completeness (Mc) — 4 methods + recommended
  8. Mc(t) time-varying
  9. Spatial Mc
 10. Gutenberg-Richter b-value
 11. Declustering results (GK + Reasenberg)
 12. ETAS sufficiency assessment
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..completeness.bvalue import estimate_bvalue
from ..completeness.mc import estimate_completeness, select_magnitude_series
from ..declustering import gardner_knopoff, reasenberg
from ..ingestion.schema import CanonicalEvent
from .qc import (
    CatalogStats,
    compute_catalog_stats,
    compute_overlap_stats,
    find_within_source_duplicates,
)


def _fmt(x, nd=2):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    if isinstance(x, (int,)):
        return str(x)
    return f"{x:.{nd}f}"


def generate_stage3_report(
    events: list[CanonicalEvent],
    observations_by_source: dict[str, list],
    files_loaded: dict[str, str],
    files_missing: list[str],
    spatial_subregions: Optional[list] = None,
) -> str:
    """Generate the Stage 3 empirical report as a markdown string."""

    # ---- 1. Sources loaded ----
    src_lines = []
    for src, path in files_loaded.items():
        n = len(observations_by_source.get(src, []))
        src_lines.append(f"- **{src}** — {n:,} observations — `{path}`")
    missing_lines = [f"- **{m}** — NOT LOADED (no local file supplied)" for m in files_missing]

    # ---- 2. Overlap & duplicates ----
    overlap = compute_overlap_stats(events)
    # within-source duplicates (per source)
    within_dup = {}
    for src, obs_list in observations_by_source.items():
        dups = find_within_source_duplicates(obs_list)
        within_dup[src] = len(dups)

    # ---- 3/4/5/6. Stats ----
    total_obs = sum(len(v) for v in observations_by_source.values())
    total_dup = sum(within_dup.values())
    stats = compute_catalog_stats(events, observations=None,
                                  within_source_duplicates=total_dup)

    # ---- 7/8/9. Completeness ----
    cr = estimate_completeness(events, prefer_mw=True,
                               spatial_subregions=spatial_subregions)

    # ---- 10. b-value ----
    mags, scale_label = select_magnitude_series(events, prefer_mw=True)
    bres = estimate_bvalue(mags, cr.mc_recommended, bin_width=0.1,
                           scale_label=cr.scale_label)

    # ---- 11. Declustering ----
    # Work on copies so the two methods don't interfere on the same objects.
    import copy
    events_gk = copy.deepcopy(events)
    gk = gardner_knopoff(events_gk, magnitude_field="mw")
    events_re = copy.deepcopy(events)
    re = reasenberg(events_re, magnitude_field="mw")

    # ---- 12. ETAS sufficiency ----
    # Heuristic: ETAS needs enough mainshocks above Mc over a long enough
    # period, with enough aftershock sequences.
    n_main_gk = gk.n_mainshocks
    n_aft_gk = gk.n_aftershocks
    n_clusters_with_aftershocks = sum(
        1 for cid in set(e.cluster_id for e in events_gk)
        if sum(1 for e in events_gk if e.cluster_id == cid and not e.is_mainshock) > 0
    )
    span_yr = stats.temporal_span_years
    rate_above_mc = cr.n_above_recommended / span_yr if span_yr > 0 else 0
    sufficiency_issues = []
    if cr.n_above_recommended < 100:
        sufficiency_issues.append(f"only {cr.n_above_recommended} events above Mc (need >=100 for stable ETAS)")
    if n_main_gk < 50:
        sufficiency_issues.append(f"only {n_main_gk} mainshocks after declustering (need >=50)")
    if n_clusters_with_aftershocks < 10:
        sufficiency_issues.append(f"only {n_clusters_with_aftershocks} clusters with aftershocks (ETAS α hard to estimate)")
    if span_yr < 20:
        sufficiency_issues.append(f"only {span_yr:.1f} years of coverage (need >=20 for stable long-term rate)")
    if cr.scale_label.startswith("original_magnitude"):
        sufficiency_issues.append("Mc/b on MIXED original types; Mw missing for most events (affects ETAS productivity α)")
    sufficient = len(sufficiency_issues) == 0

    # ---- Build markdown ----
    md = []
    md.append("# STAGE 3 — Catalog Preprocessing, Completeness & Declustering Report\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()} from ACTUAL ingested catalog files. "
              "No fabricated numbers. Missingness reported explicitly.\n")

    md.append("## 1. Data sources loaded\n")
    md.extend(src_lines)
    if missing_lines:
        md.append("\n### Not loaded (missingness)\n")
        md.extend(missing_lines)

    md.append("\n## 2. Catalog overlap & duplicate rate\n")
    md.append(f"- Total observations ingested: **{overlap.n_observations_total:,}**")
    md.append(f"- Canonical events after matching: **{overlap.n_canonical_events:,}**")
    md.append(f"- Distinct source catalogs: **{overlap.n_sources}**")
    md.append(f"- Multi-source events (>=2 catalogs): **{overlap.n_multi_source_events:,}** "
              f"(overlap fraction {_fmt(overlap.overlap_fraction,3)})")
    md.append(f"- Mean observations per event: **{_fmt(overlap.mean_observations_per_event,3)}**")
    md.append("- Per-source observation counts:")
    for s, c in overlap.per_source_event_counts.items():
        md.append(f"  - {s}: {c:,}")
    if overlap.pairwise_overlap:
        md.append("- Pairwise overlap (events shared between sources):")
        for k, v in overlap.pairwise_overlap.items():
            md.append(f"  - {k}: {v:,}")
    md.append("\n- Within-source duplicate candidates (tight 30s / 10km window):")
    for s, c in within_dup.items():
        md.append(f"  - {s}: {c} duplicate pairs")
    md.append(f"- Within-source duplicate rate: "
              f"**{_fmt(stats.duplicate_rate_within_source,4) if stats.duplicate_rate_within_source is not None else 'N/A'}**")
    if overlap.note:
        md.append(f"- _{overlap.note}_")

    md.append("\n## 3. Usable temporal coverage\n")
    md.append(f"- Time range: **{stats.temporal_range_utc[0].isoformat()}** → "
              f"**{stats.temporal_range_utc[1].isoformat()}**")
    md.append(f"- Span: **{_fmt(stats.temporal_span_years,1)} years**")
    md.append(f"- Distinct years with events: **{stats.n_distinct_years}**")
    # events per decade
    decades = {}
    for y, c in stats.events_per_year.items():
        d = (y // 10) * 10
        decades[d] = decades.get(d, 0) + c
    md.append("- Events per decade:")
    for d in sorted(decades):
        md.append(f"  - {d}s: {decades[d]:,}")

    md.append("\n## 4. Spatial coverage\n")
    md.append(f"- Latitude range: **{_fmt(stats.lat_range[0],2)}** → **{_fmt(stats.lat_range[1],2)}**")
    md.append(f"- Longitude range: **{_fmt(stats.lon_range[0],2)}** → **{_fmt(stats.lon_range[1],2)}**")
    md.append(f"- Depth range: **{_fmt(stats.depth_range_km[0],1)}** → **{_fmt(stats.depth_range_km[1],1)}** km")
    md.append(f"- Depth mean / median: **{_fmt(stats.depth_mean,1)}** / **{_fmt(stats.depth_median,1)}** km")

    md.append("\n## 5. Magnitude distribution (original, as reported)\n")
    md.append(f"- Original magnitude range: **{_fmt(stats.magnitude_original_range[0],2)}** → "
              f"**{_fmt(stats.magnitude_original_range[1],2)}**")
    md.append("- Histogram (0.1-unit bins, top 15):")
    top_bins = sorted(stats.magnitude_histogram.items())[:15]
    for b, c in top_bins:
        md.append(f"  - M≈{_fmt(b,1)}: {c}")
    md.append(f"\n- Mw available: **{stats.mw_available_count:,}** / {stats.n_events:,} events "
              f"({_fmt(100*stats.mw_available_count/stats.n_events,1)}%)")
    md.append(f"- Mw MISSING: **{stats.mw_missing_count:,}**")
    if stats.mw_missing_reasons:
        md.append("- Mw-missing reasons (count of events):")
        for r, c in stats.mw_missing_reasons.most_common():
            md.append(f"  - {c}× {r[:100]}")

    md.append("\n## 6. Magnitude-type distribution (original)\n")
    md.append("| Type | Count | Fraction |")
    md.append("|------|-------|----------|")
    for t, c in sorted(stats.magnitude_type_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {t} | {c:,} | {_fmt(100*c/stats.n_events,1)}% |")

    md.append("\n## 7. Magnitude of completeness (Mc)\n")
    md.append(f"- Magnitude scale used: **{cr.scale_label}**")
    md.append(f"- Events used: **{cr.n_events_used:,}**")
    md.append(f"- MAXC: **{_fmt(cr.mc_maxc.mc,2)}** ± {_fmt(cr.mc_maxc.uncertainty,2)} "
              f"{'(' + cr.mc_maxc.warning + ')' if cr.mc_maxc.warning else ''}")
    md.append(f"- GFT (95%): **{_fmt(cr.mc_gft.mc,2)}** ± {_fmt(cr.mc_gft.uncertainty,2)} "
              f"{('('+cr.mc_gft.warning+')') if cr.mc_gft.warning else ''}")
    md.append(f"- EMR: **{_fmt(cr.mc_emr.mc,2)}** "
              f"{('('+cr.mc_emr.warning+')') if cr.mc_emr.warning else ''}")
    md.append(f"- Stepp: **{_fmt(cr.mc_stepp.mc,2)}** ± {_fmt(cr.mc_stepp.uncertainty,2)} "
              f"{('('+cr.mc_stepp.warning+')') if cr.mc_stepp.warning else ''}")
    md.append(f"\n- **Recommended Mc: {_fmt(cr.mc_recommended,2)}** "
              f"(method: {cr.mc_recommended_method})")
    md.append(f"  - Rationale: {cr.mc_recommended_rationale}")
    md.append(f"  - Events above recommended Mc: **{cr.n_above_recommended:,}**")
    md.append(f"  - Events below recommended Mc: **{cr.n_below_recommended:,}** "
              "(excluded from Mw-based rate/b-value/ETAS estimation)")
    for note in cr.notes:
        md.append(f"  - _NOTE: {note}_")

    md.append("\n## 8. Mc(t) — time-varying completeness (MAXC, 5-year rolling)\n")
    md.append("| Period | Mc (MAXC) | N events |")
    md.append("|--------|-----------|----------|")
    for period, mc, n in cr.mc_t:
        md.append(f"| {period} | {_fmt(mc,2) if mc is not None else 'N/A'} | {n} |")

    md.append("\n## 9. Spatial Mc (MAXC per subregion)\n")
    md.append("| Region | Mc (MAXC) | N events |")
    md.append("|--------|-----------|----------|")
    for name, mc, n in cr.mc_spatial:
        md.append(f"| {name} | {_fmt(mc,2) if mc is not None else 'N/A (insufficient)'} | {n} |")

    md.append("\n## 10. Gutenberg-Richter b-value\n")
    md.append(f"- Magnitude scale: **{bres['mle'].scale_label}**")
    md.append(f"- Mc used: **{_fmt(bres['mle'].mc,2)}**")
    md.append(f"- **MLE (Aki-Utsu) b = {_fmt(bres['mle'].b,3)} ± {_fmt(bres['mle'].sigma_b,3)}** "
              f"(Shi-Bolt; N={bres['mle'].n_events_used})")
    md.append(f"- Cross-check b/sqrt(N): {_fmt(bres['mle'].sigma_b_simple,3)}")
    md.append(f"- LS (log10 cumulative) b = {_fmt(bres['ls'].b,3)} ± {_fmt(bres['ls'].sigma_b,3)} "
              f"{('('+bres['ls'].warning+')') if bres['ls'].warning else ''}")
    md.append(f"- a-value (MLE, at Mc): **{_fmt(bres['mle'].a,3)}**")
    if bres['mle'].warning:
        md.append(f"- _WARNING: {bres['mle'].warning}_")

    md.append("\n## 11. Declustering results\n")
    md.append("### Gardner-Knopoff (Knopoff 2000 windows; global, no Bangladesh adjustment)\n")
    md.append(f"- Total events: **{gk.n_total:,}**")
    md.append(f"- Mainshocks (independent): **{gk.n_mainshocks:,}**")
    md.append(f"- Aftershocks: **{gk.n_aftershocks:,}**")
    md.append(f"- Foreshocks: **{gk.n_foreshocks:,}**")
    md.append(f"- Clusters: **{gk.n_clusters:,}**")
    md.append(f"- Independent fraction: **{_fmt(gk.n_mainshocks/gk.n_total,3)}**")

    md.append("\n### Reasenberg (1985; Wells & Coppersmith 1994 radii)\n")
    md.append(f"- Total events: **{re.n_total:,}**")
    md.append(f"- Mainshocks: **{re.n_mainshocks:,}**")
    md.append(f"- Aftershocks: **{re.n_aftershocks:,}**")
    md.append(f"- Foreshocks: **{re.n_foreshocks:,}**")
    md.append(f"- Clusters: **{re.n_clusters:,}**")
    md.append(f"- Independent fraction: **{_fmt(re.n_mainshocks/re.n_total,3)}**")

    md.append("\n## 12. ETAS sufficiency assessment\n")
    md.append(f"- Events above recommended Mc: **{cr.n_above_recommended:,}**")
    md.append(f"- Mainshocks above Mc (GK): **{n_main_gk:,}**")
    md.append(f"- Clusters with >=1 aftershock: **{n_clusters_with_aftershocks:,}**")
    md.append(f"- Temporal span: **{_fmt(span_yr,1)} years**")
    md.append(f"- Approx. mainshock rate above Mc: **{_fmt(rate_above_mc,3)} / yr**")
    if sufficient:
        md.append("\n- **ASSESSMENT: Catalog appears SUFFICIENT for ETAS fitting (Stage 5).**")
    else:
        md.append("\n- **ASSESSMENT: Catalog has limitations for ETAS fitting. Issues:**")
        for iss in sufficiency_issues:
            md.append(f"  - {iss}")
    md.append("\n- _Note: ETAS parameter stability (especially α, the productivity "
              "exponent) requires a reasonable number of aftershock sequences. "
              "If insufficient locally, Stage 5 will use hierarchical Bayesian "
              "priors from analogous regions, clearly labeled as externally informed._")

    md.append("\n## 13. Provenance summary\n")
    md.append("Every canonical event carries a full provenance trail. Example (first event):")
    md.append("```")
    for s in events[0].provenance.steps:
        md.append(f"  {s.action}: {s.notes[:90]}")
    md.append("```")
    md.append("Each derived Mw records conversion_method, conversion_source, "
              "conversion_uncertainty, and validity_range. Events with missing Mw "
              "record the reason in a `mw_left_missing` provenance step.")

    md.append("\n## 14. Assumptions & limitations\n")
    md.append("- Magnitude scale: " + cr.scale_label)
    md.append("- GCMT focal mechanisms: " + ("loaded" if any(o.source_catalog=='gcmt' for e in events for o in e.observations) else "NOT LOADED (no local NDK supplied)"))
    md.append("- ISC-GEM historical anchor: " + ("loaded" if any(o.source_catalog=='isc-gem' for e in events for o in e.observations) else "NOT LOADED (no local CSV supplied)"))
    md.append("- Declustering window relations are GLOBAL (Knopoff 2000; Wells & Coppersmith 1994); no Bangladesh-specific adjustment is published.")
    md.append("- Mc methods are statistical; the recommended Mc is the median of 4 methods and should be sanity-checked against the catalog's instrumentation history.")
    md.append("- Numbers in this report come ONLY from the actual ingested files listed in Section 1.")

    return "\n".join(md)


def save_stage3_artifacts(
    events: list[CanonicalEvent],
    report_md: str,
    output_dir: str | Path,
) -> None:
    """Save the report and the processed catalog (with provenance) to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage3_report.md").write_text(report_md, encoding="utf-8")
    # Save catalog as JSON (with full provenance + observations)
    rows = [e.to_row() for e in events]
    (out / "stage3_catalog.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    # Save a flat CSV for quick inspection
    import csv
    flat_fields = [
        "canonical_id", "origin_time_utc", "latitude", "longitude", "depth_km",
        "original_magnitude", "original_magnitude_type", "mw", "mw_status",
        "mw_conversion_method", "mw_conversion_uncertainty", "magnitude_uncertainty",
        "origin_source_catalog", "n_sources", "source_catalogs",
        "completeness_magnitude", "above_completeness", "cluster_id", "is_mainshock",
        "declustering_method",
    ]
    with (out / "stage3_catalog.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
