"""Direct Omori-decay diagnostic.

USER CORRECTION (Stage 5 validation):

  Before claiming ETAS failure, test whether the catalog actually exhibits
  an Omori-Utsu-like temporal signature. For qualifying M>=5 and M>=6
  events, calculate the rate of subsequent events as a function of Δt = time
  since mainshock and compare against background rate.

  Report the empirical rate ratio:
    R(Δt) = observed post-event rate / background rate
  over logarithmic time bins.

  If there is no elevated rate immediately after mainshocks, K≈0 becomes
  much more convincing. If there is a strong short-lived elevation, the
  standard ETAS formulation may simply be misspecified for this catalog.

This is a NON-PARAMETRIC diagnostic. It does not assume ETAS; it directly
measures the aftershock-decay signature in the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent


# Log-spaced time bins (in days): from 0.01 day (~15 min) to 365 days.
LOG_TIME_BINS_DAYS = np.logspace(-2, math.log10(365), 20)


@dataclass
class OmoriDiagnosticResult:
    """Result of the direct Omori-decay diagnostic for one mainshock threshold."""

    mainshock_threshold: float
    n_mainshocks: int
    # Per-bin results
    bin_centers_days: list          # geometric mean of bin edges
    bin_edges_days: list
    n_events_in_bin: list           # total post-mainshock events in each bin
    exposure_days: list             # total mainshock-exposure in each bin (sum of bin widths × n mainshocks that reached that bin)
    observed_rate_per_day: list     # n_events_in_bin / exposure_days
    background_rate_per_day: float  # overall catalog rate above the target threshold
    rate_ratio_R: list              # observed_rate / background_rate
    # Summary
    max_rate_ratio: float
    time_of_max_rate_ratio_days: float
    omori_like: bool                # True if R(Δt) > 2 for at least one bin < 7d
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mainshock_threshold": self.mainshock_threshold,
            "n_mainshocks": self.n_mainshocks,
            "background_rate_per_day": round(self.background_rate_per_day, 6),
            "max_rate_ratio": round(self.max_rate_ratio, 3),
            "time_of_max_rate_ratio_days": round(self.time_of_max_rate_ratio_days, 3),
            "omori_like": self.omori_like,
            "notes": "; ".join(self.notes),
            "bins": [
                {"bin_center_days": round(bc, 4),
                 "n_events": int(ne),
                 "exposure_days": round(ed, 2),
                 "observed_rate_per_day": round(orr, 6),
                 "rate_ratio_R": round(rr, 3)}
                for bc, ne, ed, orr, rr in zip(
                    self.bin_centers_days, self.n_events_in_bin,
                    self.exposure_days, self.observed_rate_per_day, self.rate_ratio_R
                )
            ],
        }


def compute_omori_diagnostic(
    events: list[CanonicalEvent],
    mainshock_threshold: float = 5.0,
    target_threshold: float = 4.5,
    max_lag_days: float = 365.0,
    time_bins: Optional[np.ndarray] = None,
) -> OmoriDiagnosticResult:
    """Compute the empirical rate ratio R(Δt) = post-event rate / background rate.

    For each mainshock (M >= mainshock_threshold), we count subsequent events
    (M >= target_threshold) in log-spaced time bins. The "exposure" in each
    bin is the total time the bin was observable across all mainshocks
    (censored at catalog end). The background rate is the overall catalog
    rate above target_threshold.

    Parameters
    ----------
    mainshock_threshold : M threshold defining a "mainshock" (e.g., 5.0 or 6.0)
    target_threshold : M threshold for the events being counted (e.g., 4.5)
    max_lag_days : maximum lag to consider
    time_bins : log-spaced bin edges in days (default LOG_TIME_BINS_DAYS)
    """
    if time_bins is None:
        time_bins = LOG_TIME_BINS_DAYS

    events_sorted = sorted(events, key=lambda e: e.origin_time_utc)
    catalog_end = max(e.origin_time_utc for e in events)
    catalog_start = min(e.origin_time_utc for e in events)
    catalog_span_days = (catalog_end - catalog_start).total_seconds() / 86400.0

    # Mainshocks
    mainshocks = [e for e in events_sorted
                  if (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_threshold]
    n_ms = len(mainshocks)

    # Background rate: count of target events / total catalog span
    target_events = [e for e in events_sorted
                     if (e.mw if e.mw is not None else e.original_magnitude) >= target_threshold]
    bg_rate = len(target_events) / max(catalog_span_days, 1e-6)

    # For each bin, count post-mainshock events and accumulate exposure
    n_bins = len(time_bins) - 1
    bin_centers = []
    n_events_in_bin = np.zeros(n_bins)
    exposure_days = np.zeros(n_bins)

    for k in range(n_bins):
        lo, hi = time_bins[k], time_bins[k + 1]
        bin_centers.append(math.sqrt(lo * hi))  # geometric mean

    for ms in mainshocks:
        ms_time = ms.origin_time_utc
        # For each target event after this mainshock, assign to a bin
        for te in target_events:
            if te.origin_time_utc <= ms_time:
                continue
            lag_days = (te.origin_time_utc - ms_time).total_seconds() / 86400.0
            if lag_days > max_lag_days:
                continue
            # Find bin
            for k in range(n_bins):
                if time_bins[k] <= lag_days < time_bins[k + 1]:
                    n_events_in_bin[k] += 1
                    break
        # Exposure: for each bin, the mainshock contributed (min(hi, max_lag) - lo)
        # days of exposure IF the mainshock + bin doesn't extend past catalog_end
        for k in range(n_bins):
            lo, hi = time_bins[k], time_bins[k + 1]
            bin_hi = min(hi, max_lag_days)
            if bin_hi <= lo:
                continue
            # Censor at catalog end
            ms_plus_hi = ms_time + timedelta_days(bin_hi)
            if ms_plus_hi > catalog_end:
                bin_hi = (catalog_end - ms_time).total_seconds() / 86400.0
                if bin_hi <= lo:
                    continue
            exposure_days[k] += (bin_hi - lo)

    # Observed rate and rate ratio
    observed_rate = np.where(exposure_days > 0, n_events_in_bin / exposure_days, 0.0)
    rate_ratio = np.where(bg_rate > 0, observed_rate / bg_rate, 0.0)

    # Find max rate ratio and whether it's Omori-like
    valid = exposure_days > 0
    if valid.any():
        max_idx = int(np.argmax(np.where(valid, rate_ratio, -1)))
        max_R = float(rate_ratio[max_idx])
        t_max = float(bin_centers[max_idx])
    else:
        max_idx = -1
        max_R = 0.0
        t_max = float("nan")

    # Omori-like: R > 2 in any bin with center < 7 days
    omori_like = False
    for k in range(n_bins):
        if valid[k] and bin_centers[k] < 7.0 and rate_ratio[k] > 2.0:
            omori_like = True
            break

    notes = []
    if n_ms < 5:
        notes.append(f"Only {n_ms} mainshocks; high variance in R(Δt).")
    if omori_like:
        notes.append(
            f"Omori-like signature DETECTED: R(Δt) > 2 in at least one bin < 7d. "
            f"Peak R={max_R:.2f} at Δt={t_max:.2f} days. The catalog DOES exhibit "
            f"short-lived aftershock-like elevation; standard ETAS may be misspecified, "
            f"not wrong about triggering existence."
        )
    else:
        notes.append(
            f"No Omori-like signature: R(Δt) never exceeds 2 in bins < 7d. "
            f"Peak R={max_R:.2f} at Δt={t_max:.2f} days. This SUPPORTS the K≈0 "
            f"finding: the catalog does not exhibit the temporal aftershock decay "
            f"that ETAS is designed to capture."
        )

    return OmoriDiagnosticResult(
        mainshock_threshold=mainshock_threshold,
        n_mainshocks=n_ms,
        bin_centers_days=bin_centers,
        bin_edges_days=time_bins.tolist(),
        n_events_in_bin=n_events_in_bin.tolist(),
        exposure_days=exposure_days.tolist(),
        observed_rate_per_day=observed_rate.tolist(),
        background_rate_per_day=float(bg_rate),
        rate_ratio_R=rate_ratio.tolist(),
        max_rate_ratio=max_R,
        time_of_max_rate_ratio_days=t_max,
        omori_like=omori_like,
        notes=notes,
    )


def timedelta_days(d: float):
    from datetime import timedelta
    return timedelta(days=d)
