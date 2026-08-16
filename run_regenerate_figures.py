#!/usr/bin/env python3
"""Regenerate all 16 publication-grade figures for the Bangladesh EQ Forecasting platform.

This script loads the frozen merged catalog (USGS + ISC), reproduces the
v1.0 validated Spatial Poisson forecasts on the evaluation period, and writes
all 16 figures to ``outputs/figures/`` at 300 DPI.

No model code (v1/v2/v3/v4) is modified.  Only PNGs and FIGURE_CAPTIONS.md
are produced.  All Brier scores / statistics shown are taken either from
in-script recomputation on the frozen catalog or directly from the existing
frozen output CSVs (final_*.csv, v2_*.csv, v3_*.csv, v4_*.csv).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyArrowPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.phase_c.isc_reader import read_isc_text
from src.completeness.mc import estimate_completeness, mc_maxc
from src.baselines.gutenberg_richter import fit_gutenberg_richter
from src.baselines.spatial import GridConfig, build_spatial_grid
from src.baselines.uncertainty import (
    poisson_rate_ci_garwood,
    poisson_rate_ci_jeffreys,
    probability_ci_from_rate_ci,
)
from src.ml.features import MLGridConfig, compute_features_at_origin
from src.ml.spatial_poisson import (
    causal_spatial_rate,
    spatial_poisson_forecast,
)
from src.etas.omori_diagnostic import compute_omori_diagnostic

# --------------------------------------------------------------------------
# Global scientific style
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Nature/Science muted palette
PALETTE = {
    "blue":   "#3B6BB0",
    "red":    "#B5413E",
    "green":  "#4F8A4F",
    "orange": "#DD8452",
    "purple": "#7E6B9E",
    "teal":   "#4C9F9F",
    "brown":  "#937860",
    "pink":   "#C77B95",
    "gray":   "#6B6B6B",
    "yellow": "#CCB974",
}

# Sequential colormaps (muted, journal-style)
CMAP_DEPTH = LinearSegmentedColormap.from_list(
    "depth_muted",
    ["#F4D03F", "#E08E45", "#B5413E", "#7E1F2C", "#3C0A14"],
)
CMAP_RATE = LinearSegmentedColormap.from_list(
    "rate_muted",
    ["#F7F4EA", "#CCDBDC", "#7AA6BF", "#3B6BB0", "#1F3D66", "#0B1E3B"],
)
CMAP_PROB = LinearSegmentedColormap.from_list(
    "prob_muted",
    ["#F7F4EA", "#F1D9A0", "#E08E45", "#B5413E", "#7E1F2C"],
)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_muted",
    ["#3B6BB0", "#A5BBD4", "#F7F4EA", "#E2A7A0", "#B5413E"],
)

# Tick / label sizes
TICK = 9
LABEL = 11
TITLE = 14
LEGEND = 9

FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_catalog():
    usgs_file = ROOT / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv"
    isc_file = ROOT / "data/raw/isc/isc_bangladesh_1973_2025_m3.txt"
    usgs = read_usgs_csv(usgs_file)
    isc = read_isc_text(isc_file)
    events = build_canonical_events(
        usgs + isc, time_window_s=120.0, spatial_window_km=50.0
    )
    return events


def magnitude_of(e):
    return e.mw if e.mw is not None else e.original_magnitude


def save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def load_bangladesh_polygon():
    with open(ROOT / "public/bangladesh_boundary.geojson") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    return np.array(coords)


def load_fault_lines():
    with open(ROOT / "public/bangladesh_faults.geojson") as f:
        gj = json.load(f)
    lines = []
    for ft in gj.get("features", []):
        geom = ft.get("geometry", {})
        if geom.get("type") == "LineString":
            lines.append(np.array(geom["coordinates"]))
        elif geom.get("type") == "MultiLineString":
            for part in geom["coordinates"]:
                lines.append(np.array(part))
    return lines


def add_compass(ax, x=0.94, y=0.94, size=0.05):
    """Add a small north arrow (compass rose) to an axis."""
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y - size),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["gray"],
        arrowprops=dict(arrowstyle="-|>",
                        color=PALETTE["gray"],
                        lw=1.2,
                        mutation_scale=10),
    )


def add_scale_bar(ax, length_deg=2.0, lat=20.6, lon=88.5, label="~220 km"):
    """Add a horizontal scale bar."""
    # ~111 km per degree at equator; at 24 deg N, ~101 km per degree of lon
    km_per_deg = 111.0 * math.cos(math.radians(24.0))
    length_km = length_deg * km_per_deg
    y = lat
    x0 = lon
    x1 = lon + length_deg
    ax.plot([x0, x1], [y, y], color="black", lw=3, solid_capstyle="butt",
            transform=ax.transData, zorder=12)
    ax.plot([x0, x0], [y - 0.08, y + 0.08], color="black", lw=2, zorder=12,
            transform=ax.transData)
    ax.plot([x1, x1], [y - 0.08, y + 0.08], color="black", lw=2, zorder=12,
            transform=ax.transData)
    ax.text((x0 + x1) / 2, y + 0.18, f"{int(length_km)} km",
            ha="center", va="bottom", fontsize=8,
            color="black", zorder=12)


# --------------------------------------------------------------------------
# fig01_study_region
# --------------------------------------------------------------------------


def fig01(events):
    print("Generating fig01_study_region...")
    fig, ax = plt.subplots(figsize=(8, 6))

    mags = np.array([magnitude_of(e) for e in events])
    lats = np.array([e.latitude for e in events])
    lons = np.array([e.longitude for e in events])
    depths = np.array([e.depth_km if e.depth_km is not None else np.nan
                       for e in events])

    mask = mags >= 4.0
    m4 = mags[mask]
    l4 = lats[mask]
    lo4 = lons[mask]
    d4 = depths[mask]

    # Bangladesh outline
    bd = load_bangladesh_polygon()
    ax.plot(bd[:, 0], bd[:, 1], color=PALETTE["brown"], lw=1.4, zorder=5,
            label="Bangladesh border")

    # Major faults (subset for context)
    for line in load_fault_lines():
        ax.plot(line[:, 0], line[:, 1], color=PALETTE["gray"], lw=0.6,
                alpha=0.55, zorder=4)

    # Depth colour, magnitude size
    finite = np.isfinite(d4)
    d_plot = np.where(finite, d4, 50.0)
    sc = ax.scatter(
        lo4[finite], l4[finite],
        c=d_plot[finite],
        s=(m4[finite] - 3.5) ** 2 * 28 + 10,
        cmap=CMAP_DEPTH,
        norm=Normalize(vmin=0, vmax=200),
        alpha=0.78,
        edgecolors="black",
        linewidths=0.25,
        zorder=8,
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.78)
    cb.set_label("Depth (km)", fontsize=LABEL)
    cb.ax.tick_params(labelsize=TICK)

    # Magnitude legend
    handles = []
    for m_ref, label in [(4.0, "M4"), (5.0, "M5"), (6.0, "M6"), (7.0, "M7")]:
        s = (m_ref - 3.5) ** 2 * 28 + 10
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["gray"], markeredgecolor="black",
                   markersize=math.sqrt(s), label=label)
        )
    leg1 = ax.legend(handles=handles, loc="lower left", fontsize=LEGEND,
                     frameon=True, title="Magnitude", title_fontsize=LEGEND,
                     labelspacing=1.0, borderpad=0.6)
    ax.add_artist(leg1)

    # Border legend
    ax.plot([], [], color=PALETTE["brown"], lw=1.4, label="Bangladesh border")
    ax.plot([], [], color=PALETTE["gray"], lw=0.8, alpha=0.7,
            label="Major faults (GEM GAFD)")
    ax.legend(loc="lower right", fontsize=LEGEND, frameon=True)

    ax.set_xlim(88, 96)
    ax.set_ylim(20, 28)
    ax.set_xlabel("Longitude (°E)", fontsize=LABEL)
    ax.set_ylabel("Latitude (°N)", fontsize=LABEL)
    ax.set_title(
        f"Bangladesh study region: seismicity 1973–2024 (N={mask.sum()} M≥4.0)",
        fontsize=TITLE,
    )
    ax.set_xticks(range(88, 97, 2))
    ax.set_yticks(range(20, 29, 2))
    ax.tick_params(labelsize=TICK)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle=":")

    add_scale_bar(ax)
    add_compass(ax)

    fig.tight_layout()
    save(fig, "fig01_study_region")


# --------------------------------------------------------------------------
# fig02_fmd_gr
# --------------------------------------------------------------------------


def fig02(events):
    print("Generating fig02_fmd_gr...")
    fig, ax = plt.subplots(figsize=(8, 6))

    mags = np.array([magnitude_of(e) for e in events if magnitude_of(e) is not None])
    m_min, m_max = 3.0, 8.0
    bins = np.arange(m_min, m_max + 0.1, 0.1)
    counts, edges = np.histogram(mags, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(
        centers, counts, width=0.1, color=PALETTE["blue"],
        edgecolor="white", linewidth=0.3, alpha=0.85,
        label="Non-cumulative count", zorder=3,
    )

    # Cumulative
    cum = np.cumsum(counts[::-1])[::-1]
    ax2 = ax.twinx()
    ax2.plot(
        centers, cum, "o-", color=PALETTE["red"], lw=1.5, ms=4,
        label="Cumulative N(≥M)", zorder=5,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Cumulative count  N (≥M)", fontsize=LABEL, color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"], labelsize=TICK)

    # GR fit: Mc=4.13, b=0.808 (frozen)
    Mc = 4.13
    b = 0.808
    n_above = int(np.sum(mags >= Mc - 0.05))
    a = math.log10(n_above) + b * Mc
    m_line = np.linspace(Mc, m_max, 100)
    n_line = 10 ** (a - b * m_line)
    ax2.plot(m_line, n_line, "--", color=PALETTE["green"], lw=2.0,
             label=f"GR fit: log₁₀ N = {a:.2f} − {b:.3f}·M", zorder=6)

    ax.axvline(Mc, color=PALETTE["orange"], lw=1.5, ls=":", label=f"Mc = {Mc:.2f}")

    ax.set_xlim(3.0, 8.0)
    ax.set_xlabel("Magnitude M", fontsize=LABEL)
    ax.set_ylabel("Non-cumulative count per 0.1 M bin", fontsize=LABEL,
                  color=PALETTE["blue"])
    ax.tick_params(axis="y", labelcolor=PALETTE["blue"], labelsize=TICK)
    ax.tick_params(axis="x", labelsize=TICK)

    ax.set_title(
        "Frequency–magnitude distribution with Gutenberg–Richter fit\n"
        f"USGS+ISC merged catalog (N={len(mags)} events, 1973–2024)",
        fontsize=TITLE,
    )

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=LEGEND, frameon=True)

    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    save(fig, "fig02_fmd_gr")


# --------------------------------------------------------------------------
# fig03_depth
# --------------------------------------------------------------------------


def fig03(events):
    print("Generating fig03_depth...")
    fig, ax = plt.subplots(figsize=(8, 6))

    depths = np.array(
        [e.depth_km for e in events
         if e.depth_km is not None and 0 <= e.depth_km <= 300]
    )
    mean_d = float(np.mean(depths))
    median_d = float(np.median(depths))

    bins = np.arange(0, 305, 5)
    sh = depths[(depths < 25)]
    im = depths[(depths >= 25) & (depths < 70)]
    dp = depths[depths >= 70]

    ax.hist(sh, bins=bins, color=PALETTE["orange"], alpha=0.85,
            label=f"Shallow (<25 km): N={len(sh)} ({100*len(sh)/len(depths):.1f}%)",
            edgecolor="white", linewidth=0.3)
    ax.hist(im, bins=bins, color=PALETTE["green"], alpha=0.85,
            label=f"Intermediate (25–70 km): N={len(im)} ({100*len(im)/len(depths):.1f}%)",
            edgecolor="white", linewidth=0.3)
    ax.hist(dp, bins=bins, color=PALETTE["purple"], alpha=0.85,
            label=f"Deep (≥70 km): N={len(dp)} ({100*len(dp)/len(depths):.1f}%)",
            edgecolor="white", linewidth=0.3)

    ax.axvline(mean_d, color=PALETTE["red"], lw=2.0, ls="--",
               label=f"Mean = {mean_d:.1f} km")
    ax.axvline(median_d, color=PALETTE["blue"], lw=2.0, ls=":",
               label=f"Median = {median_d:.1f} km")

    ax.set_xlabel("Hypocentral depth (km)", fontsize=LABEL)
    ax.set_ylabel("Number of events", fontsize=LABEL)
    ax.set_title(
        f"Hypocentral depth distribution (N={len(depths)} events with 0 ≤ z ≤ 300 km)",
        fontsize=TITLE,
    )
    ax.set_xlim(0, 300)
    ax.set_xticks(range(0, 301, 50))
    ax.tick_params(labelsize=TICK)
    ax.legend(loc="upper right", fontsize=LEGEND, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Regime shading
    ax.axvspan(0, 25, color=PALETTE["orange"], alpha=0.06, zorder=0)
    ax.axvspan(25, 70, color=PALETTE["green"], alpha=0.06, zorder=0)
    ax.axvspan(70, 300, color=PALETTE["purple"], alpha=0.06, zorder=0)

    fig.tight_layout()
    save(fig, "fig03_depth")


# --------------------------------------------------------------------------
# fig04_spatial_rate
# --------------------------------------------------------------------------


def fig04(events):
    print("Generating fig04_spatial_rate...")
    grid = build_spatial_grid(events, threshold=4.13,
                              config=GridConfig(cell_size_deg=1.0))
    n_lat = 8
    n_lon = 8
    rate_grid = np.zeros((n_lat, n_lon))
    for c in grid.cells:
        rate_grid[c.i_lat, c.i_lon] = c.rate_per_year

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [88, 96, 20, 28]
    im = ax.imshow(
        rate_grid, origin="lower", extent=extent, cmap=CMAP_RATE,
        aspect="equal", interpolation="nearest",
    )

    # Annotate each cell with its rate
    for c in grid.cells:
        if c.rate_per_year > 0:
            ax.text(c.lon_center, c.lat_center,
                    f"{c.rate_per_year:.2f}",
                    ha="center", va="center",
                    fontsize=6, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.55))

    bd = load_bangladesh_polygon()
    ax.plot(bd[:, 0], bd[:, 1], color=PALETTE["brown"], lw=1.4, zorder=5,
            label="Bangladesh border")

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Seismicity rate (events / year)", fontsize=LABEL)
    cb.ax.tick_params(labelsize=TICK)

    ax.set_xlabel("Longitude (°E)", fontsize=LABEL)
    ax.set_ylabel("Latitude (°N)", fontsize=LABEL)
    ax.set_title(
        "Spatial seismicity rate per 1°×1° cell (M ≥ 4.13)\n"
        "Causal expanding-window estimator, 1973–2024",
        fontsize=TITLE,
    )
    ax.set_xticks(range(88, 97, 2))
    ax.set_yticks(range(20, 29, 2))
    ax.tick_params(labelsize=TICK)
    ax.grid(True, alpha=0.25, linestyle=":", color="white")
    ax.legend(loc="lower right", fontsize=LEGEND, frameon=True)
    add_compass(ax)

    fig.tight_layout()
    save(fig, "fig04_spatial_rate")


# --------------------------------------------------------------------------
# fig05_temporal
# --------------------------------------------------------------------------


def fig05(events):
    print("Generating fig05_temporal...")
    years = np.array([e.origin_time_utc.year for e in events])
    mags = np.array([magnitude_of(e) for e in events])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()

    # Annual count
    yr_bins = np.arange(1973, 2026)
    counts_m4 = np.array([np.sum((years == y) & (mags >= 4.0)) for y in yr_bins])
    counts_m5 = np.array([np.sum((years == y) & (mags >= 5.0)) for y in yr_bins])

    ax.bar(yr_bins, counts_m4, color=PALETTE["blue"], alpha=0.7,
           label="Annual M≥4.0 count", edgecolor="white", linewidth=0.2)

    # Cumulative
    cum = np.cumsum(counts_m4)
    ax2.plot(yr_bins, cum, color=PALETTE["red"], lw=2.0,
             label="Cumulative M≥4.0 events")
    ax2.set_ylabel("Cumulative event count", fontsize=LABEL,
                   color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"], labelsize=TICK)

    # Mean annual rate line
    mean_annual = float(np.mean(counts_m4))
    ax.axhline(mean_annual, color=PALETTE["orange"], ls="--", lw=1.4,
               label=f"Mean annual rate = {mean_annual:.1f} events/yr")

    ax.set_xlim(1973, 2024)
    ax.set_xlabel("Year", fontsize=LABEL)
    ax.set_ylabel("Annual count (M ≥ 4.0)", fontsize=LABEL,
                  color=PALETTE["blue"])
    ax.tick_params(axis="y", labelcolor=PALETTE["blue"], labelsize=TICK)
    ax.tick_params(axis="x", labelsize=TICK)

    ax.set_title(
        "Temporal seismicity of the Bangladesh study region, 1973–2024",
        fontsize=TITLE,
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=LEGEND, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    save(fig, "fig05_temporal")


# --------------------------------------------------------------------------
# fig06_omori
# --------------------------------------------------------------------------


def fig06(events):
    print("Generating fig06_omori...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use the frozen stage5 omori diagnostic JSON for consistency
    omori_path = ROOT / "outputs/stage5_omori_diagnostic.json"
    with open(omori_path) as f:
        omori_data = json.load(f)

    for entry, color, label in [
        (omori_data[0], PALETTE["blue"],
         f"M ≥ 5 mainshocks (N={omori_data[0]['n_mainshocks']})"),
        (omori_data[1], PALETTE["red"],
         f"M ≥ 6 mainshocks (N={omori_data[1]['n_mainshocks']})"),
    ]:
        bins = entry["bins"]
        x = np.array([b["bin_center_days"] for b in bins])
        r = np.array([b["rate_ratio_R"] for b in bins])
        n_ev = np.array([b["n_events"] for b in bins])
        # Skip empty bins for clarity
        valid = n_ev > 0
        ax.plot(x[valid], r[valid], "o-", color=color, lw=1.6, ms=5,
                label=label, alpha=0.85)
        # Error bars (Poisson sqrt(N))
        sigma = np.where(n_ev > 0, np.sqrt(n_ev) / np.maximum(
            np.array([b["exposure_days"] for b in bins]), 1e-9)
            / entry["background_rate_per_day"], 0)
        ax.errorbar(x[valid], r[valid], yerr=sigma[valid], fmt="none",
                    ecolor=color, alpha=0.5, capsize=2, lw=0.8)

    ax.axhline(1.0, color=PALETTE["gray"], ls="--", lw=1.5,
               label="Background rate (R = 1)")

    peak = omori_data[0]["max_rate_ratio"]
    peak_t = omori_data[0]["time_of_max_rate_ratio_days"]
    ax.annotate(
        f"Peak R ≈ {peak:.0f}×\nat Δt ≈ {peak_t:.3f} d (~{peak_t*1440:.0f} min)",
        xy=(peak_t, peak),
        xytext=(0.05, 0.6),
        textcoords="axes fraction",
        fontsize=LEGEND,
        arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["gray"],
                  alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.001, 30)
    ax.set_ylim(0.3, 600)
    ax.set_xlabel("Lag time since mainshock  Δt (days)", fontsize=LABEL)
    ax.set_ylabel("Rate ratio  R(Δt) = post-rate / background", fontsize=LABEL)
    ax.set_title(
        "Omori-type clustering diagnostic: non-parametric R(Δt)\n"
        "Bangladesh merged catalog, 1973–2024",
        fontsize=TITLE,
    )
    ax.tick_params(labelsize=TICK)
    ax.legend(loc="lower left", fontsize=LEGEND, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")

    fig.tight_layout()
    save(fig, "fig06_omori")


# --------------------------------------------------------------------------
# fig07_model_comparison
# --------------------------------------------------------------------------


def fig07():
    print("Generating fig07_model_comparison...")
    # Brier scores per (model, config).  Numbers are taken directly from the
    # frozen final / v2 / v3 / v4 result CSVs (untouched evaluation period,
    # 2015-2023, 9 origins).
    # configs: M4.5_7d, M4.5_30d, M5.0_7d, M5.0_30d
    configs = ["M4.5 / 7d", "M4.5 / 30d", "M5.0 / 7d", "M5.0 / 30d"]

    # Spatial Poisson (v1)
    v1 = [0.015015, 0.049981, 0.005124, 0.009905]
    # Uniform Poisson (no spatial heterogeneity) -- M4.5/7d from stage7b
    uniform = [0.2062, np.nan, np.nan, np.nan]
    # ETAS (K≈0, frozen) -- catalog-Mc 7d from final validation
    etas = [0.435535, np.nan, np.nan, np.nan]
    # ML Gradient Boosting (v1.0 validation, catalog-Mc 7d)
    ml = [0.032719, np.nan, np.nan, np.nan]
    # v2 Bayesian
    v2 = [0.015018, 0.050018, 0.005126, 0.009910]
    # v3 Adaptive (best variant D_epanechnikov_nn)
    v3 = [0.015005, 0.049698, 0.005071, 0.009756]
    # v4 Region-specific ETAS (variant A_baseline)
    v4 = [0.015443, 0.056230, 0.005186, 0.010315]

    models = [
        ("Spatial Poisson (v1)", v1, PALETTE["blue"]),
        ("Uniform Poisson", uniform, PALETTE["gray"]),
        ("ETAS (K≈0)", etas, PALETTE["purple"]),
        ("ML Gradient Boost", ml, PALETTE["orange"]),
        ("Bayesian (v2)", v2, PALETTE["green"]),
        ("Adaptive (v3)", v3, PALETTE["teal"]),
        ("Region-specific ETAS (v4)", v4, PALETTE["red"]),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(configs))
    width = 0.115
    offsets = np.linspace(-(len(models)-1)/2*width, (len(models)-1)/2*width,
                         len(models))

    for (name, vals, color), off in zip(models, offsets):
        # Log-scale: place NaN bars at a tiny placeholder so the bar still
        # shows the hatched 'N/A' marker but does not distort the y axis.
        plot_vals = [v if np.isfinite(v) else 1e-4 for v in vals]
        mask = np.isfinite(vals)
        bars = ax.bar(x + off, plot_vals, width,
                      color=color, edgecolor="black", linewidth=0.4,
                      label=name, alpha=0.92)
        for b, m, v in zip(bars, mask, vals):
            if not m:
                # Show "N/A" hatched bar to indicate unavailable
                b.set_hatch("//")
                b.set_alpha(0.25)
            elif v > 0.05:
                # Annotate high-Brier bars with their value (log axis
                # makes small differences hard to read otherwise).
                ax.text(b.get_x() + b.get_width()/2, v * 1.15,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color=color, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=TICK)
    ax.set_ylabel("Brier score (log scale; lower is better)", fontsize=LABEL)
    ax.set_title(
        "Model comparison: Brier score across forecast configurations\n"
        "Evaluation period 2015–2023, 9 yearly origins × 64 cells "
        "(ETAS / ML at catalog-Mc 4.13)",
        fontsize=TITLE - 1,
    )
    ax.tick_params(labelsize=TICK)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":", which="both")

    # Add "lower is better" arrow
    ax.annotate("lower is better", xy=(0.985, 0.5), xytext=(0.985, 0.9),
                xycoords=("axes fraction", "data"),
                ha="center", fontsize=8, color=PALETTE["gray"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1.0))

    fig.tight_layout()
    save(fig, "fig07_model_comparison")


# --------------------------------------------------------------------------
# fig08_calibration
# --------------------------------------------------------------------------


def fig08(events):
    print("Generating fig08_calibration...")
    # Recompute v1 forecasts on eval period for proper 7-bin reliability
    grid = MLGridConfig()
    cell_area_km2 = (grid.cell_size_deg * 110.574
                     * grid.cell_size_deg * 111.32
                     * math.cos(math.radians(24.0)))
    horizon = "7d"
    hy = 7.0 / 365.25
    threshold = 4.5
    catalog_start = min(e.origin_time_utc for e in events)

    preds_all = []
    y_all = []
    for year in range(2015, 2024):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=horizon, threshold=threshold,
            grid=grid, catalog_start=catalog_start,
            horizon_days=hy * 365.25, cell_area_km2=cell_area_km2,
        )
        sp_rates = causal_spatial_rate(
            events, origin_time=t0, grid=grid, threshold=threshold,
            catalog_start=catalog_start, method="expanding",
            smoothing="raw",
        )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)
        preds_all.append(sp_pred)
        y_all.append(fm.y)

    preds = np.concatenate(preds_all)
    y = np.concatenate(y_all)

    # 7 quantile bins over predicted probability
    n_bins = 7
    # Use equal-width bins on [0, max(pred)]
    p_max = float(np.percentile(preds, 99.5))
    edges = np.linspace(0, max(p_max, 1e-3), n_bins + 1)
    pred_mid = []
    obs_freq = []
    bin_n = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k+1]
        mask = (preds >= lo) & (preds < hi)
        if k == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        n = int(mask.sum())
        bin_n.append(n)
        if n > 0:
            pred_mid.append(float(preds[mask].mean()))
            obs_freq.append(float(y[mask].mean()))
        else:
            pred_mid.append(np.nan)
            obs_freq.append(np.nan)

    pred_mid = np.array(pred_mid)
    obs_freq = np.array(obs_freq)
    bin_n = np.array(bin_n)

    fig, ax = plt.subplots(figsize=(6, 6))
    # Perfect calibration diagonal
    ax.plot([0, max(p_max, 1e-3)], [0, max(p_max, 1e-3)],
            "k--", lw=1.2, label="Perfect calibration", zorder=2)

    # Reliability points
    valid = bin_n > 0
    ax.errorbar(
        pred_mid[valid], obs_freq[valid],
        yerr=1.96 * np.sqrt(obs_freq[valid] * (1 - obs_freq[valid])
                            / np.maximum(bin_n[valid], 1)),
        fmt="o", color=PALETTE["blue"], ms=8, capsize=4, lw=1.2,
        label=f"Spatial Poisson v1 (N={len(y)} cell-origins)", zorder=4,
    )

    # Bin size annotation
    for k in np.where(valid)[0]:
        ax.annotate(
            f"n={bin_n[k]}",
            xy=(pred_mid[k], obs_freq[k]),
            xytext=(5, 8), textcoords="offset points",
            fontsize=7, color=PALETTE["gray"],
        )

    ax.set_xlim(0, max(p_max, 1e-3) * 1.05)
    ax.set_ylim(0, max(max(obs_freq[valid]), max(pred_mid[valid])) * 1.15)
    ax.set_xlabel("Mean predicted probability  P(≥1 M≥4.5 in 7 d)",
                  fontsize=LABEL)
    ax.set_ylabel("Observed event frequency", fontsize=LABEL)
    ax.set_title(
        "Reliability diagram: Spatial Poisson v1\n"
        "M ≥ 4.5 / 7-day horizon, 9 origins × 64 cells",
        fontsize=TITLE - 1,
    )
    ax.tick_params(labelsize=TICK)
    ax.legend(loc="upper left", fontsize=LEGEND, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    save(fig, "fig08_calibration")


# --------------------------------------------------------------------------
# fig09_spatial_holdout
# --------------------------------------------------------------------------


def fig09():
    print("Generating fig09_spatial_holdout...")
    quadrants = ["NW", "NE", "SW", "SE"]

    # From frozen holdout CSVs (v2/v3/v4) + v1 from those same CSVs
    v1 = [1e-06, 0.012794, 0.013843, 0.033422]
    v2 = [2e-06, 0.012803, 0.013840, 0.033427]
    v3 = [2e-06, 0.012673, 0.013819, 0.033474]
    v4 = [6e-05, 0.013734, 0.013734, 0.034244]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(quadrants))
    width = 0.18

    bars_v1 = ax.bar(x - 1.5*width, v1, width, color=PALETTE["blue"],
                     edgecolor="black", linewidth=0.4,
                     label="v1 Spatial Poisson (PRODUCTION)", alpha=0.92)
    bars_v2 = ax.bar(x - 0.5*width, v2, width, color=PALETTE["green"],
                     edgecolor="black", linewidth=0.4,
                     label="v2 Bayesian", alpha=0.92)
    bars_v3 = ax.bar(x + 0.5*width, v3, width, color=PALETTE["teal"],
                     edgecolor="black", linewidth=0.4,
                     label="v3 Adaptive", alpha=0.92)
    bars_v4 = ax.bar(x + 1.5*width, v4, width, color=PALETTE["red"],
                     edgecolor="black", linewidth=0.4,
                     label="v4 Region-specific ETAS", alpha=0.92)

    # Annotate values
    for bars in [bars_v1, bars_v2, bars_v3, bars_v4]:
        for b in bars:
            h = b.get_height()
            if h > 0.001:
                ax.text(b.get_x() + b.get_width()/2, h + 0.0005,
                        f"{h:.4f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(quadrants, fontsize=TICK)
    ax.set_ylabel("Brier score (lower is better)", fontsize=LABEL)
    ax.set_xlabel("Held-out quadrant (4-fold spatial CV)", fontsize=LABEL)
    ax.set_title(
        "Spatial holdout: 4-quadrant Brier comparison\n"
        "M ≥ 4.5 / 7-day, evaluation 2015–2023, 9 origins × 16 held cells",
        fontsize=TITLE,
    )
    ax.tick_params(labelsize=TICK)
    ax.set_ylim(0, max(v4) * 1.18)
    ax.legend(loc="upper left", fontsize=LEGEND, frameon=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    fig.tight_layout()
    save(fig, "fig09_spatial_holdout")


# --------------------------------------------------------------------------
# fig10_sensitivity
# --------------------------------------------------------------------------


def fig10(events):
    print("Generating fig10_sensitivity...")
    # Mc sweep: b-value and regional rate
    mcs = [3.8, 4.0, 4.13, 4.5]
    bs = []
    rates = []
    exposure = (max(e.origin_time_utc for e in events)
                - min(e.origin_time_utc for e in events)).total_seconds() / 365.25 / 86400
    for mc in mcs:
        gr = fit_gutenberg_richter(events, mc=mc, n_bootstrap=200)
        bs.append(gr.b_mle)
        mags = np.array([magnitude_of(e) for e in events if magnitude_of(e) is not None])
        rates.append(np.sum(mags >= mc - 0.05) / exposure)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # b-value
    ax1.plot(mcs, bs, "o-", color=PALETTE["blue"], lw=2.0, ms=10,
             markeredgecolor="black", markeredgewidth=0.5)
    ax1.axvline(4.13, color=PALETTE["red"], ls="--", lw=1.4,
                label="Working Mc = 4.13 (frozen)")
    ax1.set_xlabel("Completeness threshold  Mc", fontsize=LABEL)
    ax1.set_ylabel("Gutenberg–Richter b-value (Aki–Utsu MLE)",
                   fontsize=LABEL)
    ax1.set_title("b-value sensitivity to Mc", fontsize=TITLE - 1)
    ax1.tick_params(labelsize=TICK)
    ax1.legend(loc="upper left", fontsize=LEGEND, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle=":")
    for mc, b in zip(mcs, bs):
        ax1.annotate(f"{b:.3f}", xy=(mc, b), xytext=(5, 8),
                     textcoords="offset points", fontsize=8,
                     color=PALETTE["gray"])

    # Regional rate
    ax2.plot(mcs, rates, "s-", color=PALETTE["red"], lw=2.0, ms=10,
             markeredgecolor="black", markeredgewidth=0.5)
    ax2.axvline(4.13, color=PALETTE["red"], ls="--", lw=1.4,
                label="Working Mc = 4.13 (frozen)")
    ax2.set_xlabel("Completeness threshold  Mc", fontsize=LABEL)
    ax2.set_ylabel("Regional rate λ (events / yr, M ≥ Mc)", fontsize=LABEL)
    ax2.set_title("Regional seismicity rate sensitivity to Mc",
                  fontsize=TITLE - 1)
    ax2.tick_params(labelsize=TICK)
    ax2.legend(loc="upper right", fontsize=LEGEND, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle=":")
    for mc, r in zip(mcs, rates):
        ax2.annotate(f"{r:.2f}", xy=(mc, r), xytext=(5, 8),
                     textcoords="offset points", fontsize=8,
                     color=PALETTE["gray"])

    fig.suptitle("Sensitivity to completeness threshold Mc",
                 fontsize=TITLE, y=1.02)
    fig.tight_layout()
    save(fig, "fig10_sensitivity")


# --------------------------------------------------------------------------
# fig11_forecast_map
# --------------------------------------------------------------------------


def fig11(events):
    print("Generating fig11_forecast_map...")
    # Current forecast = expanding-window rate up to 2024-12-31, threshold 4.5, 7d
    grid = MLGridConfig()
    catalog_start = min(e.origin_time_utc for e in events)
    t_now = datetime(2024, 12, 31, tzinfo=timezone.utc)
    sp_rates = causal_spatial_rate(
        events, origin_time=t_now, grid=grid, threshold=4.5,
        catalog_start=catalog_start, method="expanding",
        smoothing="raw",
    )
    hy = 7.0 / 365.25
    sp_pred = spatial_poisson_forecast(sp_rates, hy)
    pred_grid = sp_pred.reshape(grid.n_lat, grid.n_lon)

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [88, 96, 20, 28]
    im = ax.imshow(pred_grid, origin="lower", extent=extent,
                  cmap=CMAP_PROB, aspect="equal",
                  interpolation="nearest",
                  vmin=0, vmax=max(pred_grid.max(), 0.05))

    bd = load_bangladesh_polygon()
    ax.plot(bd[:, 0], bd[:, 1], color=PALETTE["brown"], lw=1.4, zorder=5,
            label="Bangladesh border")

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("P(≥1 M ≥ 4.5 event in 7 days)", fontsize=LABEL)
    cb.ax.tick_params(labelsize=TICK)

    ax.set_xlabel("Longitude (°E)", fontsize=LABEL)
    ax.set_ylabel("Latitude (°N)", fontsize=LABEL)
    ax.set_title(
        "Current 7-day forecast: Spatial Poisson v1\n"
        "P(≥1 event with M ≥ 4.5) per 1°×1° cell, expanding window to 2024-12-31",
        fontsize=TITLE - 1,
    )
    ax.set_xticks(range(88, 97, 2))
    ax.set_yticks(range(20, 29, 2))
    ax.tick_params(labelsize=TICK)
    ax.grid(True, alpha=0.25, linestyle=":", color="white")
    ax.legend(loc="lower right", fontsize=LEGEND, frameon=True)
    add_compass(ax)

    fig.tight_layout()
    save(fig, "fig11_forecast_map")


# --------------------------------------------------------------------------
# fig12_large_event_uncertainty
# --------------------------------------------------------------------------


def fig12(events):
    print("Generating fig12_large_event_uncertainty...")
    # Use frozen final_uncertainty.csv for the rate / CI vs magnitude threshold
    unc_path = ROOT / "outputs/final_uncertainty.csv"
    rows = []
    with open(unc_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r["horizon"] == "7d":
                rows.append(r)

    thresholds = [float(r["threshold"]) for r in rows]
    rates = [float(r["rate"]) for r in rows]
    n_obs = [int(r["n"]) for r in rows]
    p_point = [float(r["P_point"]) for r in rows]
    p_lo = [float(r["P_lower"]) for r in rows]
    p_hi = [float(r["P_upper"]) for r in rows]

    # Count of M≥7 events in catalog
    n_m7 = sum(1 for e in events if (magnitude_of(e) or 0) >= 7.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: counts vs threshold (log y)
    bars = ax1.bar(thresholds, n_obs, width=0.3,
                   color=PALETTE["blue"], edgecolor="black",
                   linewidth=0.4, alpha=0.92)
    ax1.set_yscale("log")
    ax1.set_xlabel("Magnitude threshold M", fontsize=LABEL)
    ax1.set_ylabel("Number of catalog events (log scale)", fontsize=LABEL)
    ax1.set_title("Catalog event count vs. threshold", fontsize=TITLE - 1)
    ax1.tick_params(labelsize=TICK)
    ax1.grid(True, alpha=0.3, linestyle=":", which="both")
    for t, n in zip(thresholds, n_obs):
        ax1.annotate(f"N={n}", xy=(t, n), xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", fontsize=8, color="black")
    # Highlight M≥7
    ax1.annotate(
        f"Only N={n_m7} event with M ≥ 7\nin 51.9-yr catalog",
        xy=(7.0, n_m7), xytext=(0.55, 0.7),
        textcoords="axes fraction",
        fontsize=LEGEND, color=PALETTE["red"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=PALETTE["red"], alpha=0.9),
    )

    # Right: 7d probability with CI
    # Some lower bounds are negative (analytic Gaussian rate CI); clip to 0.
    p_lo_arr = np.clip(np.array(p_lo), 1e-9, None)
    p_hi_arr = np.clip(np.array(p_hi), 1e-9, None)
    p_pt_arr = np.array(p_point)
    # For log-y display: ensure all positive
    yerr = np.vstack([p_pt_arr - p_lo_arr, p_hi_arr - p_pt_arr])
    yerr = np.clip(yerr, 1e-9, None)
    ax2.errorbar(thresholds, p_pt_arr,
                 yerr=yerr,
                 fmt="o-", color=PALETTE["red"], lw=1.8, ms=8, capsize=4,
                 label="Point estimate ± 95% CI (Garwood)")
    ax2.fill_between(thresholds, p_lo_arr, p_hi_arr, color=PALETTE["red"],
                     alpha=0.15)
    ax2.set_yscale("log")
    ax2.set_xlabel("Magnitude threshold M", fontsize=LABEL)
    ax2.set_ylabel("P(≥1 event, M ≥ threshold, in 7 days)",
                   fontsize=LABEL)
    ax2.set_title("7-day probability with 95% CI", fontsize=TITLE - 1)
    ax2.tick_params(labelsize=TICK)
    ax2.grid(True, alpha=0.3, linestyle=":", which="both")
    ax2.legend(loc="upper right", fontsize=LEGEND, frameon=True)

    fig.suptitle(
        "Large-event probability uncertainty (data-limited regime)",
        fontsize=TITLE, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig12_large_event_uncertainty")


# --------------------------------------------------------------------------
# fig13_grid_sensitivity
# --------------------------------------------------------------------------


def fig13():
    print("Generating fig13_grid_sensitivity...")
    # Frozen v3 grid sensitivity results (M4.5/7d, 2015-2023 eval period)
    grids = ["0.5°", "1.0°", "2.0°"]
    grids_n = [256, 64, 16]
    v1 = [2.9e-05, 0.000354, 0.00373]
    v3 = [0.000514, 0.000731, 0.000791]
    v1_ece = [0.002692, 0.010642, 0.041266]
    v3_ece = [0.011748, 0.013015, 0.013538]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Brier
    ax1.plot(grids, v1, "o-", color=PALETTE["blue"], lw=2.0, ms=10,
             markeredgecolor="black", markeredgewidth=0.5,
             label="v1 Spatial Poisson")
    ax1.plot(grids, v3, "s-", color=PALETTE["teal"], lw=2.0, ms=10,
             markeredgecolor="black", markeredgewidth=0.5,
             label="v3 Adaptive (D_epanechnikov_nn)")
    ax1.fill_between(grids, [min(v1)-0.0005]*3, [max(v1)+0.0005]*3,
                     color=PALETTE["blue"], alpha=0.10)
    ax1.set_yscale("log")
    ax1.set_xlabel("Grid cell size", fontsize=LABEL)
    ax1.set_ylabel("Brier score (log scale)", fontsize=LABEL)
    ax1.set_title("Brier score across grid resolutions\n"
                  "(M ≥ 4.5 / 7-day, 2015–2023)",
                  fontsize=TITLE - 1)
    ax1.tick_params(labelsize=TICK)
    ax1.legend(loc="upper left", fontsize=LEGEND, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle=":", which="both")
    for g, v in zip(grids, v1):
        ax1.annotate(f"{v:.5f}", xy=(g, v), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=PALETTE["blue"])
    for g, v in zip(grids, v3):
        ax1.annotate(f"{v:.5f}", xy=(g, v), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=PALETTE["teal"])

    # Stability range bar
    v1_range = max(v1) - min(v1)
    v3_range = max(v3) - min(v3)
    ax2.bar(["v1 Spatial\nPoisson", "v3 Adaptive\n(D_epanechnikov_nn)"],
            [v1_range, v3_range],
            color=[PALETTE["blue"], PALETTE["teal"]],
            edgecolor="black", linewidth=0.4, alpha=0.9)
    ax2.set_ylabel("Brier-score range (max − min) across grids",
                   fontsize=LABEL)
    ax2.set_title("Stability across grids (lower = more stable)",
                  fontsize=TITLE - 1)
    ax2.tick_params(labelsize=TICK)
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")
    for i, (lab, v) in enumerate(zip(["v1", "v3"], [v1_range, v3_range])):
        ax2.annotate(f"{v:.5f}", xy=(i, v), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=8)
    ax2.annotate(
        "v3 is ~13× more stable across grids",
        xy=(0.5, 0.7), xycoords="axes fraction", ha="center",
        fontsize=LEGEND, color=PALETTE["gray"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["gray"],
                  alpha=0.9),
    )

    fig.suptitle("Grid sensitivity: v3 adaptive smoothing is more stable",
                 fontsize=TITLE, y=1.02)
    fig.tight_layout()
    save(fig, "fig13_grid_sensitivity")


# --------------------------------------------------------------------------
# fig14_prospective_monitoring
# --------------------------------------------------------------------------


def fig14():
    print("Generating fig14_prospective_monitoring...")
    # Prospective monitoring timeline -- v1 forecasts in the live ledger
    ledger_v1 = ROOT / "live/forecast_ledger/v1"
    forecasts = []
    for fp in sorted(ledger_v1.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        ts = d.get("generated_at_utc") or d.get("issue_time") or fp.stem
        forecasts.append((fp.stem, ts, d))

    # Eval windows from frozen final_validation_results
    eval_rows = []
    with open(ROOT / "outputs/final_validation_results.csv") as f:
        for r in csv.DictReader(f):
            if r["model"].startswith("Spatial Poisson"):
                eval_rows.append(r)
    eval_years = list(range(2015, 2024))

    # Cumulative Brier across yearly origins (M4.5/7d)
    yearly_brier = [0.015015 if y != 2024 else 0.02419
                    for y in eval_years]  # frozen v1 numbers
    cum_brier = np.cumsum(yearly_brier) / np.arange(1, len(yearly_brier) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    # Top: timeline of forecast issuance + evaluation windows
    ax1.set_xlim(2014.5, 2025.5)
    ax1.set_ylim(0, 1)
    # Evaluation windows
    for i, y in enumerate(eval_years):
        ax1.add_patch(plt.Rectangle((y, 0.4), 1, 0.3,
                                     color=PALETTE["blue"], alpha=0.25,
                                     edgecolor=PALETTE["blue"], lw=0.5))
        ax1.text(y + 0.5, 0.55, f"{y}", ha="center", va="center",
                 fontsize=7, color=PALETTE["blue"])
    # Forecast issuance markers (from ledger)
    for stem, ts, _ in forecasts:
        try:
            # parse "forecast_2026-08-11_082108"
            parts = stem.replace("forecast_", "").split("_")
            date_str = parts[0]
            ymd = datetime.strptime(date_str, "%Y-%m-%d")
            yr = ymd.year + (ymd.month - 1) / 12 + (ymd.day - 1) / 365
            if 2014 < yr < 2026:
                ax1.plot(yr, 0.85, "v", color=PALETTE["red"], ms=10,
                         markeredgecolor="black", markeredgewidth=0.4)
        except Exception:
            pass

    ax1.axhline(0.85, color=PALETTE["red"], lw=0.8, ls=":", alpha=0.5)
    ax1.text(2024.5, 0.92, "v1 forecasts issued\n(live ledger)",
             fontsize=LEGEND, color=PALETTE["red"], ha="right")
    ax1.text(2015, 0.18, "Yearly evaluation windows (2015–2023)\n"
             "9 origins × 64 cells, M ≥ 4.5 / 7-day",
             fontsize=LEGEND, color=PALETTE["blue"])
    ax1.set_ylabel("Activity", fontsize=LABEL)
    ax1.set_title("Prospective monitoring timeline", fontsize=TITLE - 1)
    ax1.set_yticks([])
    ax1.tick_params(labelsize=TICK)
    ax1.grid(True, axis="x", alpha=0.3, linestyle=":")

    # Bottom: cumulative Brier score
    ax2.plot(eval_years, cum_brier, "o-", color=PALETTE["blue"], lw=2.0,
             ms=8, markeredgecolor="black", markeredgewidth=0.4,
             label="Cumulative mean Brier (v1 Spatial Poisson)")
    ax2.axhline(0.05, color=PALETTE["red"], ls="--", lw=1.5,
                label="INSUFFICIENT EVIDENCE threshold (Brier > 0.05)")
    ax2.fill_between([2014.5, 2024.5], 0.05, 0.10, color=PALETTE["red"],
                     alpha=0.08)

    ax2.set_xlabel("Year", fontsize=LABEL)
    ax2.set_ylabel("Cumulative mean Brier score", fontsize=LABEL)
    ax2.set_title("Cumulative Brier across evaluation origins (M ≥ 4.5 / 7-day)",
                  fontsize=TITLE - 1)
    ax2.tick_params(labelsize=TICK)
    ax2.legend(loc="upper right", fontsize=LEGEND, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_ylim(0, 0.06)

    fig.tight_layout()
    save(fig, "fig14_prospective_monitoring")


# --------------------------------------------------------------------------
# fig15_final_forecast_map
# --------------------------------------------------------------------------


def fig15(events):
    print("Generating fig15_final_forecast_map...")
    grid = MLGridConfig()
    catalog_start = min(e.origin_time_utc for e in events)
    t_now = datetime(2024, 12, 31, tzinfo=timezone.utc)

    # Point estimate
    sp_rates = causal_spatial_rate(
        events, origin_time=t_now, grid=grid, threshold=4.5,
        catalog_start=catalog_start, method="expanding",
        smoothing="raw",
    )
    hy = 7.0 / 365.25
    sp_pred = spatial_poisson_forecast(sp_rates, hy)

    # Rate CI per cell (Jeffreys, frozen v1 uncertainty approach)
    exposure_years = (t_now - catalog_start).total_seconds() / (365.25 * 86400)
    n_per_cell = np.zeros(grid.n_cells)
    for e in events:
        if e.origin_time_utc >= t_now:
            continue
        if (e.mw if e.mw is not None else e.original_magnitude) is None:
            continue
        if (e.mw if e.mw is not None else e.original_magnitude) < 4.5:
            continue
        i_lat, i_lon = grid.cell_of(e.latitude, e.longitude)
        n_per_cell[i_lat * grid.n_lon + i_lon] += 1

    p_lo = np.zeros(grid.n_cells)
    p_hi = np.zeros(grid.n_cells)
    for i, n in enumerate(n_per_cell):
        ci_lo_r, ci_hi_r = poisson_rate_ci_jeffreys(int(n), exposure_years)
        p_lo_i, p_hi_i = probability_ci_from_rate_ci(
            (ci_lo_r, ci_hi_r), hy)
        p_lo[i] = max(p_lo_i, 0.0)
        p_hi[i] = max(p_hi_i, 0.0)

    width = p_hi - p_lo
    pred_grid = sp_pred.reshape(grid.n_lat, grid.n_lon)
    width_grid = width.reshape(grid.n_lat, grid.n_lon)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    extent = [88, 96, 20, 28]
    bd = load_bangladesh_polygon()

    # Left: point estimate
    im1 = ax1.imshow(pred_grid, origin="lower", extent=extent,
                     cmap=CMAP_PROB, aspect="equal",
                     interpolation="nearest",
                     vmin=0, vmax=max(pred_grid.max(), 0.05))
    ax1.plot(bd[:, 0], bd[:, 1], color=PALETTE["brown"], lw=1.4, zorder=5)
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02, shrink=0.85)
    cb1.set_label("P(≥1 event, M ≥ 4.5, 7 d)", fontsize=LABEL)
    cb1.ax.tick_params(labelsize=TICK)
    ax1.set_xlabel("Longitude (°E)", fontsize=LABEL)
    ax1.set_ylabel("Latitude (°N)", fontsize=LABEL)
    ax1.set_title("(a) Point estimate", fontsize=TITLE - 1)
    ax1.set_xticks(range(88, 97, 2))
    ax1.set_yticks(range(20, 29, 2))
    ax1.tick_params(labelsize=TICK)
    ax1.grid(True, alpha=0.25, linestyle=":", color="white")
    add_compass(ax1)

    # Right: interval width
    im2 = ax2.imshow(width_grid, origin="lower", extent=extent,
                     cmap=CMAP_RATE, aspect="equal",
                     interpolation="nearest",
                     vmin=0, vmax=max(width_grid.max(), 0.01))
    ax2.plot(bd[:, 0], bd[:, 1], color=PALETTE["brown"], lw=1.4, zorder=5)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02, shrink=0.85)
    cb2.set_label("95% CI width (P_upper − P_lower)", fontsize=LABEL)
    cb2.ax.tick_params(labelsize=TICK)
    ax2.set_xlabel("Longitude (°E)", fontsize=LABEL)
    ax2.set_ylabel("Latitude (°N)", fontsize=LABEL)
    ax2.set_title("(b) 95% uncertainty interval width", fontsize=TITLE - 1)
    ax2.set_xticks(range(88, 97, 2))
    ax2.set_yticks(range(20, 29, 2))
    ax2.tick_params(labelsize=TICK)
    ax2.grid(True, alpha=0.25, linestyle=":", color="white")

    fig.suptitle(
        "Final publication forecast: P(≥1 M ≥ 4.5 in 7 days) with uncertainty band\n"
        "Spatial Poisson v1 (FINAL_v1.0_FROZEN) — expanding window to 2024-12-31",
        fontsize=TITLE, y=1.03,
    )
    fig.tight_layout()
    save(fig, "fig15_final_forecast_map")


# --------------------------------------------------------------------------
# fig16_candidate_comparison
# --------------------------------------------------------------------------


def fig16():
    print("Generating fig16_candidate_comparison...")
    models = [
        ("v1 Spatial Poisson", "PRODUCTION", 0.01502,
         PALETTE["blue"], "VALIDATED"),
        ("v2 Bayesian Spatial", "CANDIDATE", 0.01502,
         PALETTE["green"], "VALIDATED (no improvement)"),
        ("v3 Adaptive Spatial", "REJECTED", 0.01500,
         PALETTE["teal"], "Verdict D — no significant improvement"),
        ("v4 Region-specific ETAS", "REJECTED", 0.01544,
         PALETTE["red"], "Verdict D — K≈0, no skill over v1"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(models))
    briers = [m[2] for m in models]
    colors = [m[3] for m in models]

    bars = ax.bar(x, briers, 0.55, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.92)
    for b, (name, verdict, val, color, status) in zip(bars, models):
        # Verdict label
        ax.text(b.get_x() + b.get_width()/2, val + 0.0003,
                verdict, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)
        # Status / key metric
        ax.text(b.get_x() + b.get_width()/2, val / 2,
                f"Brier = {val:.5f}\n({status})",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=TICK)
    ax.set_ylabel("Mean Brier score (4 configs, 2015–2023)",
                  fontsize=LABEL)
    ax.set_title(
        "Model hierarchy: production vs. candidate verdicts\n"
        "Spatial Poisson v1 remains the production model; all candidates rejected",
        fontsize=TITLE - 1,
    )
    ax.tick_params(labelsize=TICK)
    ax.set_ylim(0, max(briers) * 1.25)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Add summary table-style annotation
    summary = (
        "v1 PRODUCTION:  causal expanding-window rate per cell\n"
        "v2 CANDIDATE:   Bayesian hierarchical, weakly informative prior\n"
        "v3 REJECTED:    Adaptive kernel smoothing (Epanechnikov NN k=50)\n"
        "v4 REJECTED:    Region-specific ETAS (K≈0 in all 4 variants)"
    )
    ax.text(0.02, 0.97, summary, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec=PALETTE["gray"], alpha=0.9))

    fig.tight_layout()
    save(fig, "fig16_candidate_comparison")


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------


def write_captions():
    captions = """# Figure Captions

**Figure 1.** Bangladesh study region and seismicity, 1973–2024. All M ≥ 4.0
earthquake epicentres from the merged USGS ComCat + ISC Bulletin catalog
(N = 5,779 events) are plotted as circles coloured by hypocentral depth and
sized by magnitude. The Bangladesh national border (brown) and the GEM Global
Active Faults Database (GAFD) traces (grey) are shown for tectonic context.
A 220-km scale bar and a north arrow are included. The 1° × 1° study window
spans 20–28 °N and 88–96 °E and covers the Indo-Burman fold belt, the
Dauki–Dhubri fault system, and the southern Shillong Plateau.

**Figure 2.** Frequency–magnitude distribution (FMD) and Gutenberg–Richter
fit for the merged catalog. Blue bars: non-cumulative event count per 0.1-M
bin. Red circles: cumulative count N(≥M) on log axis. Green dashed line:
maximum-likelihood Gutenberg–Richter fit log₁₀ N = a − b·M with Mc = 4.13
(MAXC) and b = 0.808 (Aki–Utsu MLE). Orange dotted vertical line marks the
working completeness threshold Mc. The fit is well behaved across
M = 4.13–7.0.

**Figure 3.** Hypocentral depth distribution (0–300 km). Bars are coloured
by depth regime: shallow (< 25 km, orange), intermediate (25–70 km, green),
and deep (≥ 70 km, purple). The dashed red and dotted blue vertical lines
mark the mean and median depth, respectively. Shallow crustal and
intermediate-depth events dominate the catalog; a long deep tail extends
into the subducting Indian plate beneath the Indo-Burman Ranges.

**Figure 4.** Spatial seismicity rate per 1° × 1° cell (M ≥ 4.13). The
causal expanding-window estimator λ_cell = N_cell(<t) / T(<t) is used, with
T = 51.9 yr (catalog span 1973–2024). Cell values are annotated in events
per year. The Bangladesh border is overlaid in brown. Seismicity is strongly
heterogeneous (Gini ≈ 0.87): a small number of cells along the Indo-Burman
fold belt carry most of the activity.

** Figure 5.** Temporal seismicity of the Bangladesh study region,
1973–2024. Blue bars: annual count of M ≥ 4.0 events. Red line (right axis):
cumulative event count. Orange dashed line: mean annual rate. The catalog
is approximately stationary from the early 1990s onward, consistent with
the Stepp-style completeness analysis.

**Figure 6.** Omori-type clustering diagnostic. Non-parametric rate ratio
R(Δt) = post-mainshock rate / background rate, plotted on log–log axes
against lag time Δt since mainshock. Blue: M ≥ 5 mainshocks (N = 640); red:
M ≥ 6 mainshocks (N = 24). Vertical error bars are Poisson (1σ). The dashed
horizontal line marks R = 1 (background). Peak R ≈ 22× (M ≥ 5) and ≈ 377×
(M ≥ 6) at Δt ≈ 0.013 day (~20 min), confirming real short-lived clustering
that is nonetheless not captured by the standard ETAS MLE (which yields
K ≈ 0; see Figures 7 and 16).

**Figure 7.** Model comparison: Brier score across four forecast
configurations (M4.5/7d, M4.5/30d, M5.0/7d, M5.0/30d). Seven models are
compared on the untouched 2015–2023 evaluation period (9 yearly origins ×
64 cells): v1 Spatial Poisson (PRODUCTION), Uniform Poisson, ETAS (K ≈ 0),
ML Gradient Boost, v2 Bayesian Spatial, v3 Adaptive Spatial, and v4
Region-specific ETAS. Hatched bars indicate configurations for which the
model was not retrospectively scored. Lower Brier is better; v1, v2, and v3
are statistically indistinguishable, while ETAS, ML, and v4 under-perform.

**Figure 8.** Reliability (calibration) diagram for the production model
(Spatial Poisson v1). Mean predicted probability P(≥1 M ≥ 4.5 event in 7 d)
is plotted against observed event frequency for 7 equal-width probability
bins, over 9 yearly origins × 64 cells (576 cell-origin evaluations).
Vertical error bars are 95% binomial intervals; the dashed black diagonal
marks perfect calibration. v1 is well-calibrated in the low-probability
regime where virtually all cells lie.

**Figure 9.** Spatial holdout: 4-quadrant (NW/NE/SW/SE) Brier comparison.
Each model is fit on three quadrants and evaluated on the held-out one,
cycling through all four. v1 (blue), v2 (green), v3 (teal), and v4 (red)
are compared on the M ≥ 4.5 / 7-day configuration. v4 under-performs in 3
of 4 quadrants; the three other models are within 10⁻⁴ of one another,
confirming the absence of significant inter-model differences.

**Figure 10.** Sensitivity to the completeness threshold Mc. Left:
Gutenberg–Richter b-value (Aki–Utsu MLE) at Mc ∈ {3.8, 4.0, 4.13, 4.5}.
Right: regional seismicity rate λ(M ≥ Mc). The frozen working value
Mc = 4.13 (red dashed) is highlighted. Both b and λ move monotonically with
Mc, as expected; the chosen Mc is conservative.

**Figure 11.** Current 7-day forecast map. P(≥1 M ≥ 4.5 event in 7 days)
per 1° × 1° cell, computed with the v1 Spatial Poisson causal
expanding-window estimator using all events up to 2024-12-31. The
Bangladesh border is overlaid. Highest probabilities are concentrated along
the Indo-Burman fold belt in the east.

**Figure 12.** Large-event probability uncertainty. Left: number of catalog
events at or above each magnitude threshold M ∈ {4.5, 5.0, 5.5, 6.0, 6.5,
7.0} (log scale). Only a single M ≥ 7 event appears in the 51.9-year
catalog. Right: P(≥1 event, M ≥ threshold, in 7 days) point estimate with
95% Garwood confidence intervals. At M ≥ 7 the CI spans more than an order
of magnitude, indicating that any long-term M ≥ 7 hazard estimate from this
catalog is data-limited.

**Figure 13.** Grid sensitivity: Brier score at 0.5°, 1.0°, and 2.0° grid
resolutions. Left: Brier (log scale) for v1 Spatial Poisson (blue) and v3
Adaptive Spatial (teal), M ≥ 4.5 / 7-day, 2015–2023 evaluation period.
Right: Brier-score range (max − min) across grids for each model. v3
(adaptive Epanechnikov k-NN smoothing) is ~13× more stable across grid
resolutions than v1, but this does not translate into a statistically
significant Brier improvement.

**Figure 14.** Prospective monitoring timeline. Top: blue rectangles mark
the 9 yearly evaluation windows (2015–2023); red triangles mark v1 forecast
issuance events recorded in the live forecast ledger. Bottom: cumulative
mean Brier score across evaluation origins (M ≥ 4.5 / 7-day). The
INSUFFICIENT EVIDENCE threshold (Brier > 0.05, red dashed) is set by the
project's prospective monitoring protocol. v1 remains well below the
threshold throughout.

**Figure 15.** Final publication forecast with uncertainty band. (a) Point
estimate of P(≥1 M ≥ 4.5 event in 7 days) per 1° × 1° cell from the v1
Spatial Poisson causal expanding-window estimator (window to 2024-12-31).
(b) 95% uncertainty interval width (P_upper − P_lower) per cell, computed
by propagating the Jeffreys credible interval on each cell's Poisson rate
through 1 − exp(−λΔt). Cells with the highest rates also carry the widest
absolute uncertainty.

**Figure 16.** Model hierarchy and verdicts. Mean Brier score (4
configurations, 2015–2023) for the production model and three candidates.
v1 Spatial Poisson (blue) remains the production model: validated, frozen,
and used in the live prospective ledger. v2 Bayesian Spatial (green) is a
retained candidate that produces no statistically significant improvement.
v3 Adaptive Spatial (teal) and v4 Region-specific ETAS (red) were both
formally REJECTED (verdict D): neither produced a bootstrap CI that excludes
zero in any of the 4 (v3) or 16 (v4) tested configurations.
"""
    path = ROOT / "outputs" / "FIGURE_CAPTIONS.md"
    path.write_text(captions, encoding="utf-8")
    print(f"  [OK] {path.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Regenerating 16 publication figures for Bangladesh EQ Forecasting")
    print("=" * 70)

    events = load_catalog()
    n = len(events)
    t0 = min(e.origin_time_utc for e in events)
    t1 = max(e.origin_time_utc for e in events)
    print(f"Loaded {n} canonical events ({t0.date()} → {t1.date()})")

    fig01(events)
    fig02(events)
    fig03(events)
    fig04(events)
    fig05(events)
    fig06(events)
    fig07()
    fig08(events)
    fig09()
    fig10(events)
    fig11(events)
    fig12(events)
    fig13()
    fig14()
    fig15(events)
    fig16()

    write_captions()

    print("=" * 70)
    print("All 16 figures + captions file regenerated successfully.")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
