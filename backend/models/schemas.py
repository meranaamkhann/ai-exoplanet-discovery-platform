"""
schemas.py
===========
Pydantic models defining the FastAPI request/response contract.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


# ---------------------------------------------------------------- Upload
class UploadResponse(BaseModel):
    dataset_id: str
    n_points_raw: int
    n_points_after_cleaning: int
    n_removed_outliers: int
    gap_fraction: float
    noise_ppm: float
    baseline_days: float
    cadence_minutes: float
    mission_hint: Optional[str] = None
    warnings: list[str] = []


class TimeSeriesPoint(BaseModel):
    t: float
    f: float


class LightCurvePreview(BaseModel):
    dataset_id: str
    raw: list[TimeSeriesPoint]
    cleaned: list[TimeSeriesPoint]
    trend: list[TimeSeriesPoint]


# --------------------------------------------------------------- Analysis
class AnalyzeRequest(BaseModel):
    dataset_id: str
    min_period_days: float = Field(default=0.5, ge=0.1, le=10.0, description="Minimum period to search (days)")
    max_period_frac_of_baseline: float = Field(default=0.5, ge=0.1, le=0.8, description="Max period as fraction of baseline")
    stellar_radius_rsun: float = Field(default=1.0, ge=0.1, le=10.0, description="Stellar radius in solar radii")
    n_top_candidates: int = Field(default=5, ge=1, le=10, description="Number of top candidates to return")
    fast_mode: bool = Field(default=True, description="Use coarser BLS grid for faster turnaround in the UI.")


class EvidenceItemSchema(BaseModel):
    direction: str
    label: str
    reason: str
    strength: float


class TransitFeaturesSchema(BaseModel):
    period_days: float
    epoch_days: float
    duration_hours: float
    depth_ppm: float
    depth_err_ppm: float
    snr: float
    bls_power: float
    n_transits_observed: int
    odd_even_depth_diff_sigma: float
    secondary_eclipse_depth_ppm: float
    secondary_eclipse_sig_sigma: float
    transit_shape_score: float
    duration_period_ratio: float
    in_transit_std_ratio: float
    periodicity_strength: float
    planet_radius_re_estimate: float
    estimated_snr_per_transit: float


class CandidateResult(BaseModel):
    candidate_id: str
    rank: int
    final_label: str
    final_confidence: float
    cnn_probabilities: dict[str, float]
    rule_adjusted_probabilities: dict[str, float]
    evidence: list[EvidenceItemSchema]
    false_positive_flags: list[str]
    is_likely_false_positive: bool
    data_quality_warning: Optional[str] = None
    features: TransitFeaturesSchema
    explanation: str
    phase_folded_global: list[float]
    phase_folded_local: list[float]
    global_saliency: list[float]
    local_saliency: list[float]


class AnalyzeResponse(BaseModel):
    dataset_id: str
    analysis_id: str
    candidates: list[CandidateResult]
    observation_summary: str
    processing_time_seconds: float
    model_version: str


# -------------------------------------------------------------- History
class HistoryItem(BaseModel):
    analysis_id: str
    dataset_id: str
    dataset_name: str
    created_at: datetime
    top_label: str
    top_confidence: float
    n_candidates: int
    is_likely_false_positive: bool


class SummaryStats(BaseModel):
    total_analyses: int
    total_candidates: int
    total_datasets: int
    by_label: dict[str, int]
    by_status: dict[str, int]
    avg_confidence: float
    likely_false_positives: int


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int


# ------------------------------------------------------------ Candidate mgmt
class CandidateStatusUpdate(BaseModel):
    candidate_id: str
    status: str  # "confirmed" | "rejected" | "needs_review" | "pending"
    reviewer_notes: Optional[str] = None


# ----------------------------------------------------------- Real / sample data
class RealTargetInfo(BaseModel):
    kepoi_name: str
    kepler_name: Optional[str]
    kic_id: int
    disposition: str
    period_days: float
    depth_ppm: float
    planet_radius_re: Optional[float]
    disposition_reason: str


class ModelInfo(BaseModel):
    run_id: str
    test_accuracy: float
    macro_f1: float
    n_train: int
    n_val: int
    n_test: int
    calibration_temperature: float
    class_names: list[str]
    trained_at: Optional[str] = None
