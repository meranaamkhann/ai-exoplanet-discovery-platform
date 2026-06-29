// Types mirroring backend/models/schemas.py exactly — keep in sync.

export interface UploadResponse {
  dataset_id: string;
  n_points_raw: number;
  n_points_after_cleaning: number;
  n_removed_outliers: number;
  gap_fraction: number;
  noise_ppm: number;
  baseline_days: number;
  cadence_minutes: number;
  mission_hint?: string | null;
  warnings: string[];
}

export interface TimeSeriesPoint {
  t: number;
  f: number;
}

export interface LightCurvePreview {
  dataset_id: string;
  raw: TimeSeriesPoint[];
  cleaned: TimeSeriesPoint[];
  trend: TimeSeriesPoint[];
}

export interface AnalyzeRequest {
  dataset_id: string;
  min_period_days?: number;
  max_period_frac_of_baseline?: number;
  stellar_radius_rsun?: number;
  n_top_candidates?: number;
  fast_mode?: boolean;
}

export interface EvidenceItem {
  direction: 'supports' | 'against';
  label: string;
  reason: string;
  strength: number;
}

export interface TransitFeatures {
  period_days: number;
  epoch_days: number;
  duration_hours: number;
  depth_ppm: number;
  depth_err_ppm: number;
  snr: number;
  bls_power: number;
  n_transits_observed: number;
  odd_even_depth_diff_sigma: number;
  secondary_eclipse_depth_ppm: number;
  secondary_eclipse_sig_sigma: number;
  transit_shape_score: number;
  duration_period_ratio: number;
  in_transit_std_ratio: number;
  periodicity_strength: number;
  planet_radius_re_estimate: number;
  estimated_snr_per_transit: number;
}

export type SignalClass =
  | 'PLANET_TRANSIT'
  | 'ECLIPSING_BINARY'
  | 'STELLAR_VARIABILITY'
  | 'STARSPOT_ACTIVITY'
  | 'INSTRUMENT_NOISE';

export interface CandidateResult {
  candidate_id: string;
  rank: number;
  final_label: SignalClass;
  final_confidence: number;
  cnn_probabilities: Record<string, number>;
  rule_adjusted_probabilities: Record<string, number>;
  evidence: EvidenceItem[];
  false_positive_flags: string[];
  is_likely_false_positive: boolean;
  data_quality_warning?: string | null;
  features: TransitFeatures;
  explanation: string;
  phase_folded_global: number[];
  phase_folded_local: number[];
  global_saliency: number[];
  local_saliency: number[];
  status?: string;
  reviewer_notes?: string | null;
}

export interface AnalyzeResponse {
  dataset_id: string;
  analysis_id: string;
  candidates: CandidateResult[];
  observation_summary: string;
  processing_time_seconds: number;
  model_version: string;
}

export interface HistoryItem {
  analysis_id: string;
  dataset_id: string;
  dataset_name: string;
  created_at: string;
  top_label: string;
  top_confidence: number;
  n_candidates: number;
  is_likely_false_positive: boolean;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
}

export interface RealTargetInfo {
  kepoi_name: string;
  kepler_name: string | null;
  kic_id: number;
  disposition: string;
  period_days: number;
  depth_ppm: number;
  planet_radius_re: number | null;
  disposition_reason: string;
}

export interface ModelInfo {
  run_id: string;
  test_accuracy: number;
  macro_f1: number;
  n_train: number;
  n_val: number;
  n_test: number;
  calibration_temperature: number;
  class_names: string[];
  trained_at?: string | null;
}

export interface SummaryStats {
  total_analyses: number;
  total_candidates: number;
  total_datasets: number;
  by_label: Record<string, number>;
  by_status: Record<string, number>;
  avg_confidence: number;
  likely_false_positives: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_run_id: string | null;
}

export interface TessFetchResponse {
  dataset_id: string;
  tic_id: number;
  sector: number;
  n_points: number;
  baseline_days: number;
  noise_ppm: number;
  source: string;
}

export interface BLSPowerSpectrum {
  periods: number[];
  power: number[];
  best_period: number;
  best_power: number;
}
