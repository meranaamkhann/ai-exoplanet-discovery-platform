"""
dataset_builder.py — expanded 14-feature aux vector for better PLANET/NOISE/STELLAR_VAR discrimination
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from synthetic_generator import SyntheticLightCurveGenerator, LightCurveSample
from preprocessing import LightCurvePreprocessor
from features import FeatureExtractor, TransitFeatures

AUX_FEATURE_NAMES = [
    # Core BLS
    "snr", "bls_power", "log_depth_ppm", "n_transits_observed",
    # Transit geometry
    "transit_shape_score", "duration_period_ratio", "in_transit_std_ratio",
    # FP discriminators
    "odd_even_depth_diff_sigma", "secondary_eclipse_sig_sigma",
    # Signal coherence
    "periodicity_strength",
    # Phase-fold quality (key for PLANET vs NOISE and STELLAR_VAR vs PLANET)
    "phase_fold_snr",
    "dip_symmetry",
    "flat_bottom_fraction",
    "log_snr_per_transit",
]


def aux_vector_from_features(feats: TransitFeatures, global_view: np.ndarray | None = None) -> np.ndarray:
    phase_fold_snr = 0.0
    dip_symmetry = 0.5
    flat_bottom_fraction = 0.0

    if global_view is not None and len(global_view) > 10:
        gv = np.asarray(global_view, dtype=float)
        n = len(gv)
        # Out-of-transit scatter: outer 40% of phase
        outer = np.concatenate([gv[:int(0.2*n)], gv[int(0.8*n):]])
        outer_rms = float(np.std(outer)) if len(outer) > 2 else 1e-6
        dip_depth = float(-np.min(gv))
        phase_fold_snr = float(np.clip(dip_depth / max(outer_rms, 1e-8), 0, 50))

        # Dip symmetry: left vs right half of central 30%
        center = gv[int(0.35*n):int(0.65*n)]
        if len(center) > 4:
            mid = len(center) // 2
            ld = -np.min(center[:mid]) if mid > 0 else 0.0
            rd = -np.min(center[mid:]) if mid < len(center) else 0.0
            total = ld + rd
            dip_symmetry = float(np.clip(1.0 - abs(ld - rd) / max(total, 1e-8), 0, 1))
            # Flat-bottom: fraction within 10% of minimum
            min_v = np.min(center)
            flat_bottom_fraction = float(np.clip(np.mean(center <= min_v * 0.9), 0, 1))

    log_snr_per_transit = float(np.log10(max(feats.estimated_snr_per_transit, 1e-3)))

    return np.array([
        np.clip(feats.snr, 0, 200),
        np.clip(feats.bls_power, 0, 200),
        np.log10(max(feats.depth_ppm, 1e-3)),
        np.clip(feats.n_transits_observed, 0, 100),
        float(np.clip(feats.transit_shape_score, 0, 1)),
        np.clip(feats.duration_period_ratio, 0, 1),
        np.clip(feats.in_transit_std_ratio, 0, 5),
        np.clip(feats.odd_even_depth_diff_sigma, 0, 50),
        np.clip(feats.secondary_eclipse_sig_sigma, -10, 100),
        np.clip(feats.periodicity_strength, 0, 50),
        phase_fold_snr,
        dip_symmetry,
        flat_bottom_fraction,
        np.clip(log_snr_per_transit, -3, 3),
    ], dtype=np.float32)


@dataclass
class ModelSample:
    global_view: np.ndarray
    local_view: np.ndarray
    aux_features: np.ndarray
    label: int
    meta: dict


class DatasetBuilder:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.generator = SyntheticLightCurveGenerator(rng=self.rng)
        self.preprocessor = LightCurvePreprocessor()
        self.feature_extractor = FeatureExtractor(
            min_period=0.5, n_periods=1500, oversample_factor=0.6, n_durations=6
        )

    def _build_one(self, sample: LightCurveSample) -> ModelSample | None:
        try:
            cleaned = self.preprocessor.clean(sample.time, sample.flux, sample.flux_err)
        except ValueError:
            return None
        if len(cleaned.time) < 20:
            return None

        has_known = sample.label_name in ("PLANET_TRANSIT", "ECLIPSING_BINARY") and np.isfinite(sample.period_days)

        try:
            bls = self.feature_extractor.run_bls(cleaned.time, cleaned.flux, cleaned.flux_err)
        except Exception:
            return None

        if has_known:
            period_err = abs(bls.best_period - sample.period_days) / sample.period_days
            if period_err < 0.05:
                period, epoch, dur = sample.period_days, sample.epoch_days, sample.duration_hours / 24
            else:
                period, epoch, dur = bls.best_period, bls.best_t0, bls.best_duration
        else:
            period, epoch, dur = bls.best_period, bls.best_t0, bls.best_duration

        try:
            feats = self.feature_extractor.extract(
                cleaned.time, cleaned.flux, cleaned.flux_err, bls,
                stellar_radius_rsun=sample.stellar_radius_rsun if np.isfinite(sample.stellar_radius_rsun) else 1.0
            )
        except Exception:
            return None

        global_view, local_view = self.preprocessor.make_global_local_views(
            cleaned.time, cleaned.flux, period, epoch, dur * 24
        )
        # Pass global_view into aux for the new phase-fold quality features
        aux = aux_vector_from_features(feats, global_view=global_view)

        return ModelSample(
            global_view=global_view, local_view=local_view, aux_features=aux,
            label=sample.label_id,
            meta={"true_period": sample.period_days, "recovered_period": bls.best_period,
                  "label_name": sample.label_name},
        )

    def build_dataset(self, n_per_class: int, baseline_days_range=(40, 120), verbose=True) -> list[ModelSample]:
        out = []
        makers = [
            self.generator.make_planet_transit,
            self.generator.make_eclipsing_binary,
            self.generator.make_stellar_variability,
            self.generator.make_starspot_activity,
            self.generator.make_instrument_noise,
        ]
        for maker in makers:
            n_ok, attempts = 0, 0
            while n_ok < n_per_class and attempts < n_per_class * 3:
                attempts += 1
                bdays = self.rng.uniform(*baseline_days_range)
                ms = self._build_one(maker(baseline_days=bdays))
                if ms is not None:
                    out.append(ms)
                    n_ok += 1
            if verbose:
                raw = maker(baseline_days=60)
                print(f"  {raw.label_name}: built {n_ok}/{n_per_class}")
        self.rng.shuffle(out)
        return out


def to_tensors(samples: list[ModelSample]):
    g   = torch.tensor(np.stack([s.global_view  for s in samples]), dtype=torch.float32)
    l   = torch.tensor(np.stack([s.local_view   for s in samples]), dtype=torch.float32)
    aux = torch.tensor(np.stack([s.aux_features for s in samples]), dtype=torch.float32)
    y   = torch.tensor([s.label for s in samples], dtype=torch.long)
    return g, l, aux, y
