"""
pipeline.py
============
Orchestrates the full detection pipeline for backend API use:
  ingest -> clean -> BLS candidate search -> CNN classify -> ensemble fuse ->
  explain -> rank candidates

Loads the most recently trained model (via ml/checkpoints/latest.json) once
at startup and reuses it across requests (model versioning support).
"""

from __future__ import annotations
import sys
import os
import json
import time as time_module
import uuid
import numpy as np
import torch

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))

from preprocessing import LightCurvePreprocessor, CleanedLightCurve  # noqa: E402
from features import FeatureExtractor, TransitFeatures, BLSResult  # noqa: E402
from dataset_builder import aux_vector_from_features, AUX_FEATURE_NAMES  # noqa: E402
from model_cnn import ExoplanetCNN, CLASS_NAMES  # noqa: E402
from ensemble import EnsembleScorer, CandidateVerdict  # noqa: E402
from explain import compute_saliency, generate_scientific_explanation, generate_observation_summary  # noqa: E402


class ModelRegistry:
    """Loads & caches the latest trained model + its calibration/scaler artifacts."""

    def __init__(self, checkpoints_dir: str | None = None):
        self.checkpoints_dir = checkpoints_dir or os.path.join(ML_DIR, "checkpoints")
        self.model: ExoplanetCNN | None = None
        self.aux_mean = None
        self.aux_std = None
        self.temperature = 1.0
        self.run_id = None
        self.metrics = {}
        self._loaded = False

    def load_latest(self):
        latest_path = os.path.join(self.checkpoints_dir, "latest.json")
        if not os.path.exists(latest_path):
            raise FileNotFoundError(
                f"No trained model found at {latest_path}. Run `python ml/train.py` first."
            )
        with open(latest_path) as f:
            latest = json.load(f)
        run_dir = latest["run_dir"]
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(ML_DIR, run_dir)

        ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        model = ExoplanetCNN(global_len=ckpt["global_len"], local_len=ckpt["local_len"],
                              n_aux_features=ckpt["n_aux_features"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model

        with open(os.path.join(run_dir, "scaler.json")) as f:
            scaler = json.load(f)
        self.aux_mean = torch.tensor(scaler["aux_mean"], dtype=torch.float32)
        self.aux_std = torch.tensor(scaler["aux_std"], dtype=torch.float32)

        with open(os.path.join(run_dir, "temperature.json")) as f:
            self.temperature = json.load(f)["temperature"]

        with open(os.path.join(run_dir, "metrics.json")) as f:
            self.metrics = json.load(f)

        self.run_id = latest["run_id"]
        self._loaded = True
        return self

    @property
    def is_loaded(self):
        return self._loaded


# Module-level singleton, loaded once at FastAPI startup
registry = ModelRegistry()


class DetectionPipeline:
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.preprocessor = LightCurvePreprocessor()
        self.ensemble_scorer = EnsembleScorer()

    def clean(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None) -> CleanedLightCurve:
        return self.preprocessor.clean(time, flux, flux_err)

    def find_candidates(self, cleaned: CleanedLightCurve, min_period_days=0.5,
                          max_period_frac=0.5, n_top_candidates=3, fast_mode=True,
                          stellar_radius_rsun=1.0):
        """
        Multi-stage candidate search: run BLS to find the single best period,
        then ALSO search for additional periodic signals after masking out the
        primary candidate's transits (handles multi-planet / multiple transit
        events in one light curve).
        """
        time, flux, flux_err = cleaned.time.copy(), cleaned.flux.copy(), cleaned.flux_err.copy()
        fe = FeatureExtractor(
            min_period=min_period_days, max_period_frac=max_period_frac,
            n_periods=4000 if fast_mode else 12000,   # more resolution than training (training used 1500)
            oversample_factor=1.5 if fast_mode else 3.5,  # better period recovery at inference time
            n_durations=10 if fast_mode else 16,
        )

        candidates = []
        working_flux = flux.copy()
        for i in range(n_top_candidates):
            if len(time) < 30:
                break
            try:
                bls = fe.run_bls(time, working_flux, flux_err)
            except Exception:
                break
            if bls.snr < 3.0 and i > 0:
                break  # stop searching once signal is indistinguishable from noise

            feats = fe.extract(time, working_flux, flux_err, bls, stellar_radius_rsun=stellar_radius_rsun)
            candidates.append((bls, feats))

            # mask out this candidate's in-transit points before searching for the next signal
            phase = (((time - bls.best_t0) / bls.best_period + 0.5) % 1.0) - 0.5
            dur_phase = bls.best_duration / bls.best_period
            in_transit = np.abs(phase) < dur_phase * 0.6
            working_flux = working_flux.copy()
            working_flux[in_transit] = np.median(working_flux[~in_transit]) if (~in_transit).any() else working_flux[in_transit]

        return candidates

    def classify(self, cleaned: CleanedLightCurve, bls: BLSResult, feats: TransitFeatures):
        if not self.registry.is_loaded:
            raise RuntimeError("Model not loaded. Call registry.load_latest() at startup.")

        global_view, local_view = self.preprocessor.make_global_local_views(
            cleaned.time, cleaned.flux, bls.best_period, bls.best_t0, bls.best_duration * 24
        )
        # Pass global_view for enhanced phase-fold quality features (phase_fold_snr, dip_symmetry, flat_bottom_fraction)
        aux = aux_vector_from_features(feats, global_view=global_view)
        aux_t = torch.tensor(aux, dtype=torch.float32)

        # Handle models trained with old 10-feature aux vector gracefully (backward compat)
        if len(aux_t) != self.registry.aux_mean.shape[0]:
            # Truncate or pad to match trained model's expected input size
            expected = self.registry.aux_mean.shape[0]
            if len(aux_t) > expected:
                aux_t = aux_t[:expected]
            else:
                aux_t = torch.nn.functional.pad(aux_t, (0, expected - len(aux_t)))

        aux_std = (aux_t - self.registry.aux_mean) / self.registry.aux_std.clamp_min(1e-6)

        g_t = torch.tensor(global_view, dtype=torch.float32)
        l_t = torch.tensor(local_view, dtype=torch.float32)

        with torch.no_grad():
            logits = self.registry.model(g_t.unsqueeze(0), l_t.unsqueeze(0), aux_std.unsqueeze(0))
            calibrated_logits = logits / self.registry.temperature
            probs = torch.softmax(calibrated_logits, dim=1).squeeze(0).numpy()

        target_class = int(np.argmax(probs))
        saliency = compute_saliency(self.registry.model, g_t, l_t, aux_std, target_class, AUX_FEATURE_NAMES)

        return probs, global_view, local_view, saliency

    def analyze(self, time, flux, flux_err=None, min_period_days=0.5, max_period_frac=0.5,
                stellar_radius_rsun=1.0, n_top_candidates=5, fast_mode=True, target_name="Uploaded target"):
        t0 = time_module.time()
        cleaned = self.clean(time, flux, flux_err)
        raw_candidates = self.find_candidates(
            cleaned, min_period_days=min_period_days, max_period_frac=max_period_frac,
            n_top_candidates=n_top_candidates, fast_mode=fast_mode, stellar_radius_rsun=stellar_radius_rsun,
        )

        results = []
        for rank, (bls, feats) in enumerate(raw_candidates, start=1):
            probs, global_view, local_view, saliency = self.classify(cleaned, bls, feats)
            verdict = self.ensemble_scorer.score(probs, feats)
            explanation = generate_scientific_explanation(verdict, feats, target_name=target_name)
            results.append({
                "candidate_id": f"cand_{uuid.uuid4().hex[:10]}",
                "rank": rank,
                "verdict": verdict,
                "features": feats,
                "explanation": explanation,
                "global_view": global_view,
                "local_view": local_view,
                "saliency": saliency,
            })

        # rank by: planet-like confidence first, then overall confidence, deprioritize FPs
        results.sort(key=lambda r: (
            r["verdict"].final_label != "PLANET_TRANSIT",
            r["verdict"].is_likely_false_positive,
            -r["verdict"].final_confidence,
        ))
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        obs_summary = generate_observation_summary(
            [(target_name, r["verdict"], r["features"]) for r in results]
        ) if results else "No periodic signal candidates were found above the detection threshold."

        elapsed = time_module.time() - t0
        return cleaned, results, obs_summary, elapsed
