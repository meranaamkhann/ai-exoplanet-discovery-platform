import axios, { AxiosError } from 'axios';
import type {
  UploadResponse, LightCurvePreview, AnalyzeRequest, AnalyzeResponse,
  HistoryResponse, RealTargetInfo, ModelInfo, HealthResponse, CandidateResult,
} from '../types/api';

const client = axios.create({
  baseURL:
    import.meta.env.VITE_API_URL ||
    "https://ai-exoplanet-discovery-platform.onrender.com",
  timeout: 180000,
});

// Structured error messages
client.interceptors.response.use(
  (res) => res,
  (err: AxiosError<{ detail?: string }>) => {
    const msg = err.response?.data?.detail ?? err.message ?? 'Unknown error';
    return Promise.reject(new Error(msg));
  }
);

export const api = {
  health: () => client.get<HealthResponse>('/health').then((r) => r.data),
  modelInfo: () => client.get<ModelInfo>('/model-info').then((r) => r.data),
  uploadFile: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return client.post<UploadResponse>('/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data);
  },
  preview: (datasetId: string, maxPoints = 3000) =>
    client.get<LightCurvePreview>(`/datasets/${datasetId}/preview`, { params: { max_points: maxPoints } }).then((r) => r.data),
  analyze: (req: AnalyzeRequest) => client.post<AnalyzeResponse>('/analyze', req).then((r) => r.data),
  getAnalysis: (analysisId: string) => client.get(`/analyses/${analysisId}`).then((r) => r.data),
  history: (limit = 50, offset = 0) =>
    client.get<HistoryResponse>('/history', { params: { limit, offset } }).then((r) => r.data),
  candidates: (status?: string, label?: string) =>
    client.get<CandidateResult[]>('/candidates', { params: { status, label } }).then((r) => r.data),
  updateCandidateStatus: (candidateId: string, status: string, reviewerNotes?: string) =>
    client.post('/candidates/status', { candidate_id: candidateId, status, reviewer_notes: reviewerNotes }).then((r) => r.data),
  realTargets: () => client.get<RealTargetInfo[]>('/real-targets').then((r) => r.data),
  loadRealTarget: (kepoiName: string, baselineDays = 90, seed = 0) =>
    client.post(`/real-targets/${kepoiName}/load`, null, { params: { baseline_days: baselineDays, seed } }).then((r) => r.data),
  generateDemo: (kind: string, seed = 0, baselineDays = 90) =>
    client.post('/demo/generate', null, { params: { kind, seed, baseline_days: baselineDays } }).then((r) => r.data),
  fetchTess: (ticId: number, sector?: number) =>
    client.get(`/fetch-tess/${ticId}`, { params: sector ? { sector } : {} }).then((r) => r.data),
  fetchKepler: (kicId: number, quarter?: number) =>
    client.get(`/fetch-kepler/${kicId}`, { params: quarter ? { quarter } : {} }).then((r) => r.data),
};

export default api;
