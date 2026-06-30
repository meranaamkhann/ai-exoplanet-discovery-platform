"""
preprocessing.py
=================
Light curve cleaning & preparation pipeline. Mirrors the steps a real
astronomy pipeline (e.g. Kepler PDC-MAP / TESS SPOC) performs before
transit search:

  1. Remove NaNs / non-finite points (corrupted observations)
  2. Sigma-clip outliers (cosmic rays, flares) iteratively
  3. Detrend long-term stellar/instrumental trends via robust Savitzky-Golay
     or median-filter approach (preserves transit shape, removes slow drift)
  4. Normalize to median = 1.0 (fractional flux)
  5. Estimate a per-point noise model (rolling MAD) for SNR calculations
  6. Interpolate over small gaps (for CNN input only — raw data keeps gaps marked)
  7. Phase-fold given a period + epoch, producing global & local (zoomed) views
     for the CNN, like the Astronet local/global view representation.

Everything here is dependency-light (numpy/scipy/pandas only) and works for
irregularly-sampled, gappy, multi-mission data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
from dataclasses import dataclass


@dataclass
class CleanedLightCurve:
    time: np.ndarray
    flux: np.ndarray          # detrended, normalized (median ~1.0)
    flux_err: np.ndarray
    raw_flux: np.ndarray      # normalized but NOT detrended (for plotting/inspection)
    trend: np.ndarray         # the trend that was subtracted
    quality_flags: np.ndarray  # bool, True = good/kept point
    n_removed_outliers: int
    n_input_points: int
    n_output_points: int
    gap_fraction: float
    noise_ppm: float           # robust per-point noise estimate (MAD-based)


class LightCurvePreprocessor:
    def __init__(self, sigma_clip_threshold: float = 5.0, detrend_window_days: float = 0.75):
        self.sigma_clip_threshold = sigma_clip_threshold
        self.detrend_window_days = detrend_window_days

    # --------------------------------------------------------------- main
    def clean(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None) -> CleanedLightCurve:
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        n_input = len(time)

        # 1. drop non-finite / corrupted points
        finite = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            flux_err = np.asarray(flux_err, dtype=float)
            finite &= np.isfinite(flux_err) & (flux_err > 0)
        else:
            flux_err = np.full_like(flux, np.nan)

        time, flux, flux_err = time[finite], flux[finite], flux_err[finite]

        # sort by time (handle any out-of-order cadences) and drop exact duplicate timestamps
        order = np.argsort(time)
        time, flux, flux_err = time[order], flux[order], flux_err[order]
        _, uniq_idx = np.unique(time, return_index=True)
        time, flux, flux_err = time[uniq_idx], flux[uniq_idx], flux_err[uniq_idx]

        if len(time) < 10:
            raise ValueError("Too few valid points after cleaning (<10). Check input data quality.")

        # 2. normalize roughly first (median = 1) so outlier clipping thresholds are scale-free
        med = np.nanmedian(flux)
        if med == 0 or not np.isfinite(med):
            raise ValueError("Invalid flux normalization (median is zero/non-finite).")
        flux_norm = flux / med
        raw_flux_norm = flux_norm.copy()

        # 3. iterative sigma clipping on residuals from a NARROW rolling median.
        #    Critically, this window is sized in CADENCES (not days) and kept narrow — it only
        #    targets single-point/few-point spikes (cosmic rays, sensor glitches). A wide window
        #    here would track real transit dips and clip them as "outliers", destroying signal.
        keep_mask = np.ones(len(flux_norm), dtype=bool)
        cadence = np.median(np.diff(time)) if len(time) > 1 else 0.02
        outlier_win_points = 7  # ~3.4 hours at Kepler long cadence; narrow & fixed by design
        outlier_win_points = min(outlier_win_points, max(5, len(flux_norm) - (1 - len(flux_norm) % 2)))
        outlier_win_points += (1 - outlier_win_points % 2)

        for _ in range(3):
            local_med = median_filter(flux_norm, size=min(outlier_win_points, len(flux_norm) - 1 or 1))
            resid = flux_norm - local_med
            mad = np.median(np.abs(resid[keep_mask] - np.median(resid[keep_mask]))) * 1.4826
            mad = max(mad, 1e-6)
            candidate_outlier = np.abs(resid) >= self.sigma_clip_threshold * mad
            # Cosmic rays / glitches are ISOLATED single (or 2-point) spikes. Real transits and
            # eclipses are multi-point CONSECUTIVE dips. Only clip candidates that are isolated —
            # i.e. not part of a run of >=3 consecutive flagged points — so we never remove a
            # genuine (even very deep) transit/eclipse, only true point-glitches.
            isolated_outlier = candidate_outlier & ~self._in_run_of_at_least(candidate_outlier, 3)
            new_keep = keep_mask & ~isolated_outlier
            # safety valve: never clip more than 3% of points in one pass
            if (keep_mask & ~new_keep).sum() > 0.03 * len(flux_norm):
                worst = np.argsort(np.where(isolated_outlier, np.abs(resid), -np.inf))[::-1]
                allow = int(0.03 * len(flux_norm))
                new_keep = keep_mask.copy()
                new_keep[worst[:allow]] = False
            keep_mask = new_keep

        n_removed = int((~keep_mask).sum())
        time_c, flux_c, err_c = time[keep_mask], flux_norm[keep_mask], flux_err[keep_mask]

        # 4. detrend: subtract a smooth long-term trend (Savitzky-Golay), preserving short dips
        trend = self._robust_trend(time_c, flux_c)
        flux_detrended = flux_c - trend + 1.0  # re-center at 1.0

        # 5. noise estimate via MAD of high-pass residual (robust to outliers & transits)
        hp_resid = flux_detrended - median_filter(flux_detrended, size=min(7, len(flux_detrended) - 1 or 1))
        noise_ppm = float(np.median(np.abs(hp_resid - np.median(hp_resid))) * 1.4826 * 1e6)

        # fill missing flux_err if it wasn't provided
        if np.all(~np.isfinite(err_c)):
            err_c = np.full_like(flux_detrended, max(noise_ppm, 50) / 1e6)

        # gap fraction: based on expected vs actual cadence count over the baseline
        baseline = time_c[-1] - time_c[0] if len(time_c) > 1 else 0
        expected_n = baseline / max(cadence, 1e-6)
        gap_fraction = float(max(0.0, 1.0 - len(time_c) / max(expected_n, 1)))

        return CleanedLightCurve(
            time=time_c, flux=flux_detrended, flux_err=err_c,
            raw_flux=raw_flux_norm[keep_mask], trend=trend, quality_flags=keep_mask,
            n_removed_outliers=n_removed, n_input_points=n_input,
            n_output_points=len(time_c), gap_fraction=gap_fraction, noise_ppm=noise_ppm,
        )

    @staticmethod
    def _in_run_of_at_least(mask: np.ndarray, min_run: int) -> np.ndarray:
        """Return a boolean array marking points that belong to a consecutive run of True
        values in `mask` of length >= min_run. Used to distinguish multi-point astrophysical
        dips (transits/eclipses — keep) from isolated single-point glitches (cosmic rays — clip).
        """
        n = len(mask)
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        run_start = None
        for i in range(n + 1):
            val = mask[i] if i < n else False
            if val and run_start is None:
                run_start = i
            elif not val and run_start is not None:
                if i - run_start >= min_run:
                    out[run_start:i] = True
                run_start = None
        return out

    # ------------------------------------------------------------- helpers
    def _robust_trend(self, time, flux):
        """Estimate slow stellar/instrumental trend while staying robust to short, narrow
        transit-like dips.

        Key idea: a plain rolling MEDIAN with a window comparable to (or wider than) the
        orbital period will itself dip during transits, since transits can occupy a large
        fraction of points inside the window when the period is short. Instead we use a
        rolling HIGH PERCENTILE (e.g. 80th) as the trend estimate — transits are one-sided
        dips, so the upper envelope of the local distribution is dominated by genuine
        out-of-transit flux and is far less biased by transit width/period coincidences.
        The window is also capped at a modest absolute size (independent of any assumed
        period) so it tracks slow trends only.
        """
        if len(time) < 15:
            return np.full_like(flux, np.median(flux))
        cadence = np.median(np.diff(time))
        win_days = max(self.detrend_window_days, 0.3)
        win_points = max(7, int(win_days / max(cadence, 1e-6)))
        win_points += (1 - win_points % 2)
        win_points = min(win_points, len(flux) - (1 - len(flux) % 2))
        win_points = max(win_points, 5)

        trend = self._rolling_percentile(flux, win_points, percentile=75)
        print("\n===== ROLLING PERCENTILE =====")
        print("Trend min   :", np.min(trend))
        print("Trend max   :", np.max(trend))
        print("Trend median:", np.median(trend))
        print("==============================\n")

        # light smoothing of the trend itself to avoid step artifacts from the percentile filter
        try:
            poly = min(2, win_points - 1)
            smooth_win = win_points if win_points % 2 == 1 else win_points + 1
            smooth_win = min(smooth_win, len(trend) - (1 - len(trend) % 2))
            smooth_win = max(smooth_win, 5)
            trend = signal.savgol_filter(trend, window_length=smooth_win, polyorder=poly)
        except Exception:
            pass
        return trend

    def _rolling_percentile(self, flux: np.ndarray, win_points: int, percentile: float = 75):
        """Vectorized rolling percentile via a sliding-window view (fast, no Python loop)."""
        n = len(flux)
        half = win_points // 2
        padded = np.pad(flux, (half, half), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, win_points)
        return np.percentile(windows, percentile, axis=1)[:n]

    # ------------------------------------------------------- phase folding
    @staticmethod
    def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float):
        """Return phase in [-0.5, 0.5) sorted, and the corresponding flux."""
        phase = (((time - epoch) / period + 0.5) % 1.0) - 0.5
        order = np.argsort(phase)
        return phase[order], flux[order]

    @staticmethod
    def binned_view(phase: np.ndarray, flux: np.ndarray, n_bins: int, phase_range=(-0.5, 0.5)):
        """Produce a fixed-length binned representation (for CNN input)."""
        bins = np.linspace(phase_range[0], phase_range[1], n_bins + 1)
        bin_idx = np.digitize(phase, bins) - 1
        binned = np.full(n_bins, np.nan)
        for i in range(n_bins):
            sel = bin_idx == i
            if sel.any():
                binned[i] = np.median(flux[sel])
        # fill empty bins via linear interpolation, edges via nearest
        nan_mask = np.isnan(binned)
        if nan_mask.any() and not nan_mask.all():
            idxs = np.arange(n_bins)
            binned[nan_mask] = np.interp(idxs[nan_mask], idxs[~nan_mask], binned[~nan_mask])
        elif nan_mask.all():
            binned[:] = 1.0
        return binned

    @classmethod
    def make_global_local_views(cls, time, flux, period, epoch, duration_hours,
                                  n_global=201, n_local=61, local_window_durations=4.0):
        """
        Astronet-style dual representation:
          - global view: full phase-folded curve, n_global bins across [-0.5, 0.5)
          - local view: zoomed around transit, +/- local_window_durations * duration
        """
        phase, f = cls.phase_fold(time, flux, period, epoch)
        global_view = cls.binned_view(phase, f, n_global, (-0.5, 0.5))

        dur_phase = (duration_hours / 24.0) / period
        half_window = min(0.49, max(dur_phase * local_window_durations, 0.01))
        local_view = cls.binned_view(phase, f, n_local, (-half_window, half_window))

        # normalize both views: median-subtract using out-of-transit (edges) as baseline
        def _norm(view):
            edge = np.concatenate([view[: max(1, len(view) // 8)], view[-max(1, len(view) // 8):]])
            baseline = np.median(edge) if len(edge) else 1.0
            return view - baseline

        return _norm(global_view).astype(np.float32), _norm(local_view).astype(np.float32)
