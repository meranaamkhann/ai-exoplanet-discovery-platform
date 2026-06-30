"""
features.py
============
Classical signal-analysis layer: Box Least Squares (BLS) period search and
hand-engineered features that feed both (a) the false-positive rule engine
and (b) the ensemble classifier alongside the CNN.

These mirror real vetting metrics used by the Kepler/TESS pipelines
(odd-even depth test, secondary eclipse search, V-shape/grazing test,
SNR, etc.) — this is what makes the system's verdicts explainable.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from astropy.timeseries import BoxLeastSquares


@dataclass
class BLSResult:
    best_period: float
    best_t0: float
    best_duration: float
    depth: float
    depth_err: float
    snr: float
    power: float
    periods_tested: np.ndarray
    power_spectrum: np.ndarray


@dataclass
class TransitFeatures:
    # core BLS-derived
    period_days: float
    epoch_days: float
    duration_hours: float
    depth_ppm: float
    depth_err_ppm: float
    snr: float
    bls_power: float
    n_transits_observed: int

    # vetting / false-positive diagnostics
    odd_even_depth_diff_sigma: float   # large -> likely eclipsing binary (alternating depths)
    secondary_eclipse_depth_ppm: float  # significant -> likely EB
    secondary_eclipse_sig_sigma: float
    transit_shape_score: float          # 0 (V-shaped/grazing) .. 1 (flat-bottomed/planet-like)
    duration_period_ratio: float        # unusually large -> EB-like
    in_transit_std_ratio: float         # scatter in-transit vs out-of-transit (>1.5 hints contamination/blend)
    periodicity_strength: float         # primary BLS peak power vs next-best peak (phase coherence)

    # derived physical estimates (require a stellar radius assumption if not provided)
    planet_radius_re_estimate: float
    estimated_snr_per_transit: float

    def to_dict(self):
        d = asdict(self)
        d.pop("periods_tested", None)
        return d


class FeatureExtractor:
    def __init__(self, min_period=0.5, max_period_frac=0.5, n_periods=6000, oversample_factor=3.0, n_durations=14):
        self.min_period = min_period
        self.max_period_frac = max_period_frac  # fraction of baseline
        self.n_periods = n_periods
        self.oversample_factor = oversample_factor
        self.n_durations = n_durations

    # ------------------------------------------------------------- BLS
    def run_bls(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> BLSResult:
        baseline = time[-1] - time[0]
        max_period = max(self.min_period * 2, baseline * self.max_period_frac)

        # Frequency-spaced grid (standard BLS practice): linear-in-period grids are far too
        # coarse at short periods (where BLS peaks are narrow), causing aliasing/missed signals.
        # Spacing follows the standard BLS frequency resolution: df ~ q / baseline, where q is
        # the expected transit duty cycle. We use a conservative oversampling factor.
        min_duration_guess = 30 / 1440  # 30 min, conservative minimum transit duration
        duty_cycle = min_duration_guess / self.min_period
        df = duty_cycle / baseline / self.oversample_factor
        f_min, f_max = 1.0 / max_period, 1.0 / self.min_period
        n_freq_needed = int(np.ceil((f_max - f_min) / max(df, 1e-12)))
        # Respect the resolution requirement; only cap at a high ceiling to bound runtime on
        # pathological inputs (very long baseline + very short min_period).
        n_freq = int(np.clip(n_freq_needed, 500, max(self.n_periods, 20000)))
        freqs = np.linspace(f_max, f_min, n_freq)
        periods = 1.0 / freqs  # descending freq -> ascending period

        model = BoxLeastSquares(time, flux, dy=flux_err)
        # Duration grid: realistic transit durations range ~20 min to ~14 hours regardless of
        # period (duration scales with sqrt(a) not period directly, and a/Rstar varies a lot).
        # astropy requires max(duration) < min(period) strictly.
        max_dur = min(14 / 24, self.min_period * 0.95)
        durations = np.geomspace(20 / 1440, max_dur, self.n_durations)
        result = model.power(periods, durations, objective="snr")

        best_idx = np.argmax(result.power)
        best_period = result.period[best_idx]
        best_t0 = result.transit_time[best_idx]
        best_duration = result.duration[best_idx]
        depth = result.depth[best_idx]
        depth_err = result.depth_err[best_idx] if hasattr(result, "depth_err") else depth * 0.1
        power = result.power[best_idx]

        print("\n========== BLS DEBUG ==========")
        print("Flux min :", np.min(flux))
        print("Flux max :", np.max(flux))
        print("Flux median :", np.median(flux))
        print("BLS depth :", depth)
        print("BLS power :", power)
        print("BLS duration :", best_duration)
        print("===============================\n")

        # NOTE on period aliasing: BLS can lock onto either P or a sub-multiple of P (commonly
        # P/2) depending on noise/gaps/duty cycle, and a simple "prefer the shorter period if its
        # power is comparably strong" heuristic is NOT reliable — it cannot distinguish "true
        # period is P, P/2 is a noise alias" from "true period is P/2, BLS's main peak at P is
        # itself the alias" using power ratio alone (we verified both cases occur in practice).
        # We therefore keep BLS's own best answer here and handle period-aliasing diagnosis in
        # the vetting layer (odd-even depth test, secondary-eclipse test), which look at the
        # actual per-transit depth PATTERN rather than just aggregate power.

        # SNR from BLS stats if available
        try:
            stats = model.compute_stats(best_period, best_duration, best_t0)
            snr = float(np.atleast_1d(stats.get("snr", power))[0]) if isinstance(stats, dict) else float(power)
        except Exception:
            snr = float(power)

        return BLSResult(
            best_period=float(best_period), best_t0=float(best_t0),
            best_duration=float(best_duration), depth=float(depth),
            depth_err=float(depth_err), snr=float(snr), power=float(power),
            periods_tested=result.period, power_spectrum=result.power,
        )

    # --------------------------------------------------------- vetting
    def extract(self, time, flux, flux_err, bls: BLSResult, stellar_radius_rsun: float = 1.0) -> TransitFeatures:
        period, t0, dur = bls.best_period, bls.best_t0, bls.best_duration
        phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
        dur_phase = dur / period
        in_transit = np.abs(phase) < dur_phase / 2
        out_transit = ~in_transit

        n_transits_observed = self._count_observed_transits(time, period, t0, dur)
        depth_ppm = max(bls.depth, 1e-8) * 1e6

        odd_even_sigma = self._odd_even_test(time, flux, flux_err, period, t0, dur)
        sec_depth_ppm, sec_sigma = self._secondary_eclipse_test(time, flux, flux_err, period, t0, dur, primary_depth_ppm=depth_ppm)
        shape_score = self._transit_shape_score(phase, flux, dur_phase)
        dur_period_ratio = dur / period
        in_std_ratio = self._in_out_scatter_ratio(flux, in_transit, out_transit)
        periodicity_strength = self._periodicity_strength(bls)

        rp_re = self._radius_from_depth(depth_ppm, stellar_radius_rsun)


        noise = np.std(flux[out_transit]) if out_transit.sum() > 5 else np.std(flux)
        snr_per_transit = (depth_ppm / 1e6) / max(noise, 1e-8)

        return TransitFeatures(
            period_days=period, epoch_days=t0, duration_hours=dur * 24,
            depth_ppm=depth_ppm, depth_err_ppm=bls.depth_err * 1e6,
            snr=bls.snr, bls_power=bls.power, n_transits_observed=n_transits_observed,
            odd_even_depth_diff_sigma=odd_even_sigma,
            secondary_eclipse_depth_ppm=sec_depth_ppm, secondary_eclipse_sig_sigma=sec_sigma,
            transit_shape_score=shape_score, duration_period_ratio=dur_period_ratio,
            in_transit_std_ratio=in_std_ratio, periodicity_strength=periodicity_strength,
            planet_radius_re_estimate=rp_re, estimated_snr_per_transit=snr_per_transit,
        )

    # ------------------------------------------------------------ tests
    def _count_observed_transits(self, time, period, t0, duration):
        cycle = np.floor((time - t0) / period + 0.5)
        return int(len(np.unique(cycle[np.abs(((time - t0) / period + 0.5) % 1 - 0.5) < (duration / period)])))

    def _odd_even_test(self, time, flux, flux_err, period, t0, duration):
        """Compare depth of odd- vs even-numbered transits. Large diff => likely eclipsing binary
        at half the true period (primary/secondary masquerading as one periodic dip).

        This is also our primary defense against the P/2 BLS alias: when BLS locks onto half the
        true period of an eclipsing binary, primary and secondary eclipses interleave as
        "odd"/"even" transits in the folded series at period P/2 — so this test directly catches
        that case even when the secondary-eclipse-at-phase-0.5 test (run at the wrong period) can't.
        """
        phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
        dur_phase = duration / period
        in_transit = np.abs(phase) < dur_phase / 2
        out_transit = ~in_transit
        cycle = np.floor((time - t0) / period + 0.5)
        if in_transit.sum() < 6:
            return 0.0
        odd_mask = in_transit & (np.mod(cycle, 2) == 1)
        even_mask = in_transit & (np.mod(cycle, 2) == 0)
        if odd_mask.sum() < 2 or even_mask.sum() < 2:
            return 0.0
        odd_depth = 1 - np.mean(flux[odd_mask])
        even_depth = 1 - np.mean(flux[even_mask])
        odd_err = np.std(flux[odd_mask]) / np.sqrt(odd_mask.sum())
        even_err = np.std(flux[even_mask]) / np.sqrt(even_mask.sum())
        combined_err = np.sqrt(odd_err ** 2 + even_err ** 2) + 1e-10

        # Critical safeguard against the P/2-alias-of-a-single-planet false positive: a TRUE
        # eclipsing binary has BOTH odd and even transits as real, individually-significant dips
        # (primary AND secondary eclipses both physically exist). A spurious P/2 alias of a single
        # real transit instead has ONE real dip and one near-baseline/noisy "transit" (the
        # interleaved empty cycle) — so we require both depths to be statistically significant
        # relative to the global out-of-transit noise floor before trusting the odd/even diff.
        global_noise = np.std(flux[out_transit]) if out_transit.sum() > 5 else combined_err
        odd_significant = abs(odd_depth) > 2.0 * (global_noise / np.sqrt(max(odd_mask.sum(), 1)))
        even_significant = abs(even_depth) > 2.0 * (global_noise / np.sqrt(max(even_mask.sum(), 1)))
        if not (odd_significant and even_significant):
            return 0.0  # one side isn't a real dip -> this is a P/2 alias artifact, not an EB

        return float(abs(odd_depth - even_depth) / combined_err)

    def _secondary_eclipse_test(self, time, flux, flux_err, period, t0, duration, primary_depth_ppm=None):
        """Search near phase 0.5 for a secondary dip — strong evidence of an eclipsing binary
        rather than a planet (planets' secondary eclipses are ~undetectably shallow).

        Robustness note: light curves have correlated (red) noise on top of white photon noise,
        so a naive std/sqrt(n) standard error UNDERESTIMATES the true uncertainty and can falsely
        flag noise fluctuations as "significant" secondary eclipses (look-elsewhere effect across
        many candidates makes this worse). We therefore (a) estimate the baseline scatter from
        BINNED out-of-transit points (averaging down red noise less aggressively than sqrt(n) on
        raw points would suggest, giving a more conservative/realistic error), and (b) additionally
        require the secondary depth to be a non-trivial fraction of the primary transit depth
        (real EB secondaries are typically >5-10% of the primary depth; sub-percent "secondaries"
        are almost always noise even if formally several-sigma under an idealized white-noise model).
        """
        phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
        dur_phase = duration / period
        sec_mask = np.abs(np.abs(phase) - 0.5) < dur_phase / 2
        out_mask = (np.abs(phase) > dur_phase) & (np.abs(np.abs(phase) - 0.5) > dur_phase)
        if sec_mask.sum() < 5 or out_mask.sum() < 5:
            return 0.0, 0.0
        sec_depth = 1 - np.mean(flux[sec_mask])

        # Conservative error estimate: bin out-of-transit flux into ~20 bins and use the scatter
        # of BIN MEANS (captures red-noise correlation better than raw-point std/sqrt(n)).
        out_flux = flux[out_mask]
        n_bins = max(5, min(20, len(out_flux) // 10))
        bin_means = np.array([np.mean(chunk) for chunk in np.array_split(out_flux, n_bins) if len(chunk) > 0])
        binned_scatter = np.std(bin_means) if len(bin_means) > 1 else np.std(out_flux) / np.sqrt(len(out_flux))
        sec_err = max(binned_scatter, np.std(out_flux) / np.sqrt(sec_mask.sum()))
        sig = sec_depth / max(sec_err, 1e-10)

        # Require non-trivial depth ratio relative to primary transit to avoid flagging tiny,
        # statistically-fragile dips as a "secondary eclipse". Real eclipsing-binary secondary
        # eclipses are typically >=10-15% of the primary depth (see calibration in dataset:
        # synthetic EB secondary/primary ratios range ~0.14-0.87, median ~0.38); planet-host
        # "secondaries" are noise/systematics in essentially all realistic cases below that.
        if primary_depth_ppm is not None and primary_depth_ppm > 0:
            depth_ratio = (sec_depth * 1e6) / primary_depth_ppm
            if depth_ratio < 0.12:
                sig *= 0.15  # heavily discount: far too shallow relative to primary for a credible EB secondary
            elif depth_ratio < 0.25:
                sig *= 0.45  # borderline: discount but don't fully dismiss

            # Additional gate: for very shallow primary transits (<300 ppm), the photometric
            # noise floor means the secondary test window is dominated by noise at the ~ppm level,
            # making any "detection" statistically fragile regardless of sigma. Real eclipsing-binary
            # secondaries at such tiny depths would need space-grade photometry to confirm — at
            # typical Kepler SNR, sub-300ppm signals (like Kepler-10b at 152ppm) can't reliably
            # distinguish a true secondary from correlated noise.
            if primary_depth_ppm < 300:
                sig *= 0.4  # discount significantly for very shallow-transit systems

        return float(max(0, sec_depth) * 1e6), float(sig)

    def _transit_shape_score(self, phase, flux, dur_phase):
        """0 = sharp V-shape (grazing/EB-like), 1 = flat-bottomed (planet-like).

        Compares the depth in the innermost CORE of the transit (center) against the depth
        near the ingress/egress EDGE (just inside the transit boundary). For a flat-bottomed
        (low-impact-parameter, fully-transiting) signal, the bottom is flat near the center and
        only falls off sharply right at ingress/egress — so the edge region (still mostly outside
        the flat core) is much shallower than the core, giving a LOW edge/core ratio.
        For a grazing/V-shaped eclipse (high impact parameter), depth increases smoothly and
        roughly uniformly across the whole transit width with no flat plateau, so edge and core
        depths are comparable, giving a HIGH edge/core ratio. We therefore invert the raw
        edge/core ratio to map onto the documented 0 (V-shape) .. 1 (flat-bottom) convention.
        """
        half = dur_phase / 2
        if half <= 0:
            return 0.5
        core = np.abs(phase) < half * 0.25
        edge = (np.abs(phase) > half * 0.6) & (np.abs(phase) < half * 0.95)
        if core.sum() < 3 or edge.sum() < 3:
            return 0.5
        core_depth = 1 - np.median(flux[core])
        edge_depth = 1 - np.median(flux[edge])
        if core_depth <= 1e-10:
            return 0.5
        ratio = edge_depth / core_depth
        return float(np.clip(1.0 - ratio, 0, 1))

    def _in_out_scatter_ratio(self, flux, in_transit, out_transit):
        if in_transit.sum() < 5 or out_transit.sum() < 5:
            return 1.0
        in_std = np.std(flux[in_transit])
        out_std = np.std(flux[out_transit])
        return float(in_std / max(out_std, 1e-10))

    def _periodicity_strength(self, bls: BLSResult):
        """Ratio of best peak to the median of the power spectrum, excluding a window around
        the best period (and its harmonics) — high value = sharp, confident periodicity."""
        power = bls.power_spectrum
        periods = bls.periods_tested
        best_p = bls.best_period
        mask = np.ones_like(power, dtype=bool)
        for harmonic in [0.5, 1.0, 2.0]:
            target = best_p * harmonic
            mask &= np.abs(periods - target) > 0.02 * target
        background = np.median(power[mask]) if mask.any() else np.median(power)
        return float(power.max() / max(background, 1e-10))

    def _radius_from_depth(self, depth_ppm, stellar_radius_rsun):
        rp_rstar = np.sqrt(max(depth_ppm, 0) / 1e6)
        rp_rsun = rp_rstar * stellar_radius_rsun
        return float(rp_rsun / 0.009158)  # convert to Earth radii
