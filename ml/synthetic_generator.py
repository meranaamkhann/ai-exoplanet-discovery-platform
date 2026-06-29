"""
synthetic_generator.py
=======================
Physics-grounded synthetic light curve generator for training and validating
the exoplanet transit detection pipeline.

Generates labeled light curves for 5 classes:
  0 - PLANET_TRANSIT       : Mandel-Agol-style limb-darkened transit, periodic
  1 - ECLIPSING_BINARY      : Deep V/U-shaped eclipses, often with secondary eclipse
  2 - STELLAR_VARIABILITY   : Pulsating / rotating star, sinusoidal-ish, no flat bottom
  3 - STARSPOT_ACTIVITY     : Quasi-periodic spot modulation, evolving amplitude
  4 - INSTRUMENT_NOISE      : Pure noise / systematics, jumps, no real periodic signal

Design notes:
- Uses a simplified analytic transit model (trapezoid + smoothed limb-darkening
  approximation) rather than requiring `batman` (kept dependency-free & fast).
- Injects realistic systematics: red (1/f) noise, point-to-point white noise,
  flux discontinuities (jumps), sparse/missing cadences, occasional cosmic-ray
  outliers - mirroring real Kepler/TESS data quality issues.
- Every generated curve carries full ground-truth metadata for supervised
  training AND for evaluating the pipeline's regression outputs (depth,
  duration, period, etc.) against truth.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

LABELS = {
    0: "PLANET_TRANSIT",
    1: "ECLIPSING_BINARY",
    2: "STELLAR_VARIABILITY",
    3: "STARSPOT_ACTIVITY",
    4: "INSTRUMENT_NOISE",
}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}


@dataclass
class LightCurveSample:
    time: np.ndarray                 # days
    flux: np.ndarray                 # normalized flux (~1.0 baseline)
    flux_err: np.ndarray             # per-point uncertainty
    label_id: int
    label_name: str
    # ground truth physical parameters (NaN if not applicable to this class)
    period_days: float = float("nan")
    epoch_days: float = float("nan")
    duration_hours: float = float("nan")
    depth_ppm: float = float("nan")
    planet_radius_re: float = float("nan")     # Earth radii
    stellar_radius_rsun: float = float("nan")
    stellar_teff_k: float = float("nan")
    impact_param: float = float("nan")
    snr_true: float = float("nan")
    notes: str = ""

    def to_meta_dict(self) -> dict:
        d = asdict(self)
        d.pop("time"); d.pop("flux"); d.pop("flux_err")
        return d


class SyntheticLightCurveGenerator:
    """
    Generates light curves mimicking Kepler/TESS long-cadence photometry.

    rng: numpy.random.Generator for full reproducibility (seed it!)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, cadence_min: float = 29.4):
        self.rng = rng or np.random.default_rng()
        self.cadence_days = cadence_min / (60 * 24)  # Kepler long cadence default

    # ---------------------------------------------------------------- utils
    def _time_grid(self, baseline_days: float) -> np.ndarray:
        n = int(baseline_days / self.cadence_days)
        t = np.arange(n) * self.cadence_days
        return t

    def _apply_data_gaps(self, time, flux, flux_err, gap_prob=0.15, max_gap_frac=0.08):
        """Simulate downlink gaps / safe-mode events -> missing cadences."""
        if self.rng.random() < gap_prob:
            n = len(time)
            n_gaps = self.rng.integers(1, 4)
            mask = np.ones(n, dtype=bool)
            for _ in range(n_gaps):
                gap_len = int(n * self.rng.uniform(0.005, max_gap_frac))
                start = self.rng.integers(0, max(1, n - gap_len))
                mask[start:start + gap_len] = False
            return time[mask], flux[mask], flux_err[mask]
        return time, flux, flux_err

    def _sparse_sample(self, time, flux, flux_err, keep_frac=None):
        """Randomly drop cadences to simulate irregular/sparse sampling."""
        if keep_frac is None:
            return time, flux, flux_err
        n = len(time)
        keep = self.rng.random(n) < keep_frac
        keep[0] = keep[-1] = True
        return time[keep], flux[keep], flux_err[keep]

    def _add_red_noise(self, n, sigma, tau_cadences=50):
        """1/f-like correlated (red) noise via AR(1) process — mimics instrumental drift."""
        phi = np.exp(-1.0 / max(tau_cadences, 1))
        eps = self.rng.normal(0, sigma, n)
        red = np.zeros(n)
        for i in range(1, n):
            red[i] = phi * red[i - 1] + eps[i]
        red -= red.mean()
        return red

    def _add_jumps(self, flux, n_jumps_range=(0, 2), amp_range=(0.0005, 0.004)):
        """Simulate instrumental discontinuities (e.g., safe-mode recovery steps)."""
        n = len(flux)
        n_jumps = self.rng.integers(*n_jumps_range)
        out = flux.copy()
        for _ in range(n_jumps):
            idx = self.rng.integers(int(0.1 * n), int(0.9 * n))
            amp = self.rng.uniform(*amp_range) * self.rng.choice([-1, 1])
            out[idx:] += amp
        return out

    def _add_cosmic_rays(self, flux, rate=0.002, amp_range=(0.003, 0.02)):
        n = len(flux)
        out = flux.copy()
        n_hits = self.rng.binomial(n, rate)
        idxs = self.rng.choice(n, size=n_hits, replace=False) if n_hits > 0 else []
        for idx in idxs:
            out[idx] += self.rng.uniform(*amp_range) * self.rng.choice([-1, 1])
        return out

    def _quadratic_limb_darkened_transit(self, phase, rp_rstar, duration_phase, impact_param, u1=0.3, u2=0.2):
        """
        Smooth analytic approximation of a limb-darkened transit using a
        smoothstep ingress/egress instead of a full Mandel-Agol integral.
        Captures: flat bottom, smooth ingress/egress, depth ~ (Rp/Rstar)^2,
        and impact-parameter-dependent V-shape (grazing vs central transits).
        """
        half_dur = duration_phase / 2.0
        depth = rp_rstar ** 2
        # limb darkening makes the bottom slightly curved & ingress/egress softer
        ld_factor = 1.0 - 0.3 * (u1 + u2)
        x = np.abs(phase) / half_dur
        # grazing transits (high impact param) -> more V-shaped, shallower flat bottom
        flat_frac = max(0.05, 1.0 - impact_param)
        flux = np.ones_like(phase)
        in_transit = x < 1.0
        # smoothstep profile creates rounded ingress/egress and a flat-ish bottom
        xt = np.clip((x[in_transit] - (1 - flat_frac)) / max(flat_frac, 1e-3), 0, 1)
        smooth = 1 - (3 * xt ** 2 - 2 * xt ** 3)  # smoothstep descent
        flux[in_transit] -= depth * ld_factor * smooth
        return flux

    # ---------------------------------------------------------- class makers
    def make_planet_transit(self, baseline_days=80.0, snr_level="random") -> LightCurveSample:
        t = self._time_grid(baseline_days)
        n = len(t)

        stellar_teff = self.rng.uniform(4500, 6500)
        stellar_radius = self.rng.uniform(0.7, 1.3)
        period = self.rng.uniform(1.0, 40.0)
        epoch = self.rng.uniform(0, period)
        rp_re = self.rng.uniform(0.8, 12.0)  # Earth radii
        rp_rstar = (rp_re * 0.009158) / stellar_radius  # R_earth/R_sun ≈ 0.009158
        impact = self.rng.uniform(0.0, 0.85)
        duration_hours = self._estimate_duration_hours(period, stellar_radius, impact)
        duration_phase = (duration_hours / 24.0) / period

        phase = (((t - epoch) / period + 0.5) % 1.0) - 0.5
        flux = self._quadratic_limb_darkened_transit(phase, rp_rstar, duration_phase, impact)

        depth_ppm = (rp_rstar ** 2) * 1e6

        if snr_level == "random":
            snr_level = self.rng.choice(["high", "medium", "low", "very_low"], p=[0.25, 0.35, 0.25, 0.15])
        sigma = self._sigma_for_snr(depth_ppm, snr_level)

        white = self.rng.normal(0, sigma, n)
        red = self._add_red_noise(n, sigma * 0.6, tau_cadences=self.rng.integers(20, 150))
        flux = flux + white + red
        flux = self._add_jumps(flux)
        flux = self._add_cosmic_rays(flux)
        flux_err = np.full(n, sigma) * self.rng.uniform(0.9, 1.1, n)

        t, flux, flux_err = self._apply_data_gaps(t, flux, flux_err)
        if self.rng.random() < 0.3:
            t, flux, flux_err = self._sparse_sample(t, flux, flux_err, keep_frac=self.rng.uniform(0.4, 0.8))

        n_transits = max(1, int(baseline_days / period))
        snr_true = depth_ppm / (sigma * 1e6) * np.sqrt(max(1, n_transits))

        return LightCurveSample(
            time=t, flux=flux, flux_err=flux_err,
            label_id=0, label_name="PLANET_TRANSIT",
            period_days=period, epoch_days=epoch, duration_hours=duration_hours,
            depth_ppm=depth_ppm, planet_radius_re=rp_re, stellar_radius_rsun=stellar_radius,
            stellar_teff_k=stellar_teff, impact_param=impact, snr_true=snr_true,
            notes=f"snr_level={snr_level}, n_transits_in_baseline={n_transits}",
        )

    def make_eclipsing_binary(self, baseline_days=80.0) -> LightCurveSample:
        t = self._time_grid(baseline_days)
        n = len(t)
        period = self.rng.uniform(0.5, 20.0)
        epoch = self.rng.uniform(0, period)
        depth_primary = self.rng.uniform(0.01, 0.5)   # 1-50% — much deeper than planets
        depth_secondary = depth_primary * self.rng.uniform(0.1, 0.9)
        impact = self.rng.uniform(0.0, 0.9)
        duration_hours = self.rng.uniform(1.5, 14.0)
        duration_phase = (duration_hours / 24.0) / period

        phase = (((t - epoch) / period + 0.5) % 1.0) - 0.5
        flux = self._quadratic_limb_darkened_transit(phase, np.sqrt(depth_primary), duration_phase, impact)
        # secondary eclipse near phase 0.5 (allow slight eccentricity offset)
        secondary_offset = self.rng.uniform(0.45, 0.55)
        phase2 = (((t - epoch) / period + 0.5 - secondary_offset) % 1.0) - 0.5
        flux2 = self._quadratic_limb_darkened_transit(phase2, np.sqrt(depth_secondary), duration_phase * 1.1, impact)
        flux = flux + (flux2 - 1.0)

        sigma = self.rng.uniform(0.0003, 0.002)
        white = self.rng.normal(0, sigma, n)
        red = self._add_red_noise(n, sigma * 0.5, tau_cadences=self.rng.integers(20, 150))
        flux = flux + white + red
        flux = self._add_jumps(flux)
        flux_err = np.full(n, sigma) * self.rng.uniform(0.9, 1.1, n)

        t, flux, flux_err = self._apply_data_gaps(t, flux, flux_err)

        return LightCurveSample(
            time=t, flux=flux, flux_err=flux_err,
            label_id=1, label_name="ECLIPSING_BINARY",
            period_days=period, epoch_days=epoch, duration_hours=duration_hours,
            depth_ppm=depth_primary * 1e6, stellar_radius_rsun=self.rng.uniform(0.8, 1.5),
            stellar_teff_k=self.rng.uniform(4500, 7000), impact_param=impact,
            snr_true=depth_primary * 1e6 / (sigma * 1e6),
            notes=f"secondary_depth_ppm={depth_secondary*1e6:.1f}, secondary_offset={secondary_offset:.3f}",
        )

    def make_stellar_variability(self, baseline_days=80.0) -> LightCurveSample:
        """Pulsators / non-radial oscillators: smooth sinusoidal variability, no flat-bottom dips."""
        t = self._time_grid(baseline_days)
        n = len(t)
        n_modes = self.rng.integers(1, 4)
        flux = np.ones(n)
        primary_period = self.rng.uniform(0.2, 15.0)
        for k in range(n_modes):
            p = primary_period / (k + 1) * self.rng.uniform(0.9, 1.1)
            amp = self.rng.uniform(0.0005, 0.01) / (k + 1)
            phase0 = self.rng.uniform(0, 2 * np.pi)
            flux += amp * np.sin(2 * np.pi * t / p + phase0)

        sigma = self.rng.uniform(0.0003, 0.0015)
        flux += self.rng.normal(0, sigma, n)
        flux += self._add_red_noise(n, sigma * 0.4, tau_cadences=self.rng.integers(30, 200))
        flux_err = np.full(n, sigma) * self.rng.uniform(0.9, 1.1, n)
        t, flux, flux_err = self._apply_data_gaps(t, flux, flux_err)

        return LightCurveSample(
            time=t, flux=flux, flux_err=flux_err,
            label_id=2, label_name="STELLAR_VARIABILITY",
            period_days=primary_period, stellar_teff_k=self.rng.uniform(5000, 9000),
            notes=f"n_pulsation_modes={n_modes}",
        )

    def make_starspot_activity(self, baseline_days=80.0) -> LightCurveSample:
        """Rotational modulation from starspots: quasi-periodic, evolving amplitude/shape (spot evolution)."""
        t = self._time_grid(baseline_days)
        n = len(t)
        rot_period = self.rng.uniform(0.5, 30.0)
        base_amp = self.rng.uniform(0.001, 0.03)
        # amplitude envelope evolves slowly (spots grow/decay over ~weeks)
        envelope_period = self.rng.uniform(20, 100)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * t / envelope_period + self.rng.uniform(0, 2 * np.pi))
        # quasi-periodicity: period itself drifts slightly (differential rotation)
        drift = self.rng.uniform(-0.05, 0.05)
        instantaneous_period = rot_period * (1 + drift * t / baseline_days)
        phase_accum = np.cumsum(2 * np.pi / instantaneous_period * np.gradient(t))
        flux = 1.0 - base_amp * envelope * (0.5 + 0.5 * np.sin(phase_accum))
        # second harmonic for non-sinusoidal spot shape
        flux -= base_amp * 0.3 * envelope * np.sin(2 * phase_accum + 1.0)

        sigma = self.rng.uniform(0.0003, 0.0012)
        flux += self.rng.normal(0, sigma, n)
        flux_err = np.full(n, sigma) * self.rng.uniform(0.9, 1.1, n)
        t, flux, flux_err = self._apply_data_gaps(t, flux, flux_err)

        return LightCurveSample(
            time=t, flux=flux, flux_err=flux_err,
            label_id=3, label_name="STARSPOT_ACTIVITY",
            period_days=rot_period, stellar_teff_k=self.rng.uniform(3500, 5800),
            notes=f"envelope_period={envelope_period:.1f}, drift={drift:.3f}",
        )

    def make_instrument_noise(self, baseline_days=80.0) -> LightCurveSample:
        """Pure systematics / noise: no real astrophysical periodicity."""
        t = self._time_grid(baseline_days)
        n = len(t)
        sigma = self.rng.uniform(0.0005, 0.003)
        flux = np.ones(n) + self.rng.normal(0, sigma, n)
        flux += self._add_red_noise(n, sigma * self.rng.uniform(0.5, 1.5), tau_cadences=self.rng.integers(10, 300))
        flux = self._add_jumps(flux, n_jumps_range=(1, 4), amp_range=(0.001, 0.01))
        flux = self._add_cosmic_rays(flux, rate=self.rng.uniform(0.002, 0.01))
        # occasionally inject a quasi-periodic *non-astrophysical* artifact (e.g. thermal/momentum-dump cycle)
        if self.rng.random() < 0.4:
            art_period = self.rng.uniform(0.5, 6.0)
            art_amp = self.rng.uniform(0.0005, 0.003)
            flux += art_amp * (np.mod(t, art_period) < art_period * 0.05)
        flux_err = np.full(n, sigma) * self.rng.uniform(0.9, 1.1, n)
        t, flux, flux_err = self._apply_data_gaps(t, flux, flux_err, gap_prob=0.3)

        return LightCurveSample(
            time=t, flux=flux, flux_err=flux_err,
            label_id=4, label_name="INSTRUMENT_NOISE",
            notes="pure_systematics",
        )

    # --------------------------------------------------------------- helpers
    def _estimate_duration_hours(self, period_days, stellar_radius_rsun, impact_param):
        """Approximate transit duration via T ~ (P/pi) * asin(Rstar/a * sqrt(1-b^2)), with a from Kepler's 3rd law (assume Mstar~1 Msun)."""
        a_au = (period_days / 365.25) ** (2 / 3)  # assumes M_star ~ 1 Msun
        rstar_au = stellar_radius_rsun * 0.00465047
        ratio = np.clip(rstar_au / a_au * np.sqrt(max(1e-6, 1 - impact_param ** 2)), 0, 1)
        dur_days = (period_days / np.pi) * np.arcsin(ratio)
        return max(0.3, dur_days * 24)

    def _sigma_for_snr(self, depth_ppm, level):
        # choose photometric noise sigma to target a desired SNR regime
        table = {
            "high": (depth_ppm / 1e6) / self.rng.uniform(15, 30),
            "medium": (depth_ppm / 1e6) / self.rng.uniform(8, 15),
            "low": (depth_ppm / 1e6) / self.rng.uniform(4, 8),
            "very_low": (depth_ppm / 1e6) / self.rng.uniform(1.5, 4),
        }
        return max(1e-5, table[level])

    # ------------------------------------------------------------- dataset
    def generate_dataset(self, n_per_class: int, baseline_days_range=(40, 120)) -> list[LightCurveSample]:
        makers = [
            self.make_planet_transit,
            self.make_eclipsing_binary,
            self.make_stellar_variability,
            self.make_starspot_activity,
            self.make_instrument_noise,
        ]
        samples = []
        for maker in makers:
            for _ in range(n_per_class):
                bdays = self.rng.uniform(*baseline_days_range)
                samples.append(maker(baseline_days=bdays))
        self.rng.shuffle(samples)
        return samples


def save_dataset_npz(samples: list[LightCurveSample], path: str):
    """Save as compressed npz (ragged arrays via object dtype) + sidecar JSON metadata."""
    times = np.array([s.time for s in samples], dtype=object)
    fluxes = np.array([s.flux for s in samples], dtype=object)
    errs = np.array([s.flux_err for s in samples], dtype=object)
    labels = np.array([s.label_id for s in samples])
    np.savez_compressed(path, time=times, flux=fluxes, flux_err=errs, label=labels, allow_pickle=True)
    meta = [s.to_meta_dict() for s in samples]
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    gen = SyntheticLightCurveGenerator(rng=np.random.default_rng(42))
    samples = gen.generate_dataset(n_per_class=5)
    for s in samples[:5]:
        print(s.label_name, len(s.time), s.to_meta_dict())
