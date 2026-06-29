import { useState, useCallback, useRef } from 'react';
import {
  Upload, Sparkles, Database, Play, Loader2,
  ChevronDown, Satellite, Globe, AlertTriangle, CheckCircle2, X
} from 'lucide-react';
import api from '../lib/api';
import type { UploadResponse, LightCurvePreview, RealTargetInfo } from '../types/api';
import LightCurveViewer from '../components/LightCurveViewer';
import { Panel } from '../components/ui';
import { useNavigate } from 'react-router-dom';

const DEMO_KINDS = [
  { key: 'planet',            label: 'Planet Transit',      desc: 'Periodic flat-bottomed dip from exoplanet',     color: 'var(--color-class-planet)' },
  { key: 'eclipsing_binary',  label: 'Eclipsing Binary',    desc: 'Deep stellar companion eclipse, secondary dip', color: 'var(--color-class-eb)' },
  { key: 'stellar_variability', label: 'Stellar Variability', desc: 'Sinusoidal pulsation, no flat bottom',        color: 'var(--color-class-variability)' },
  { key: 'starspot',          label: 'Starspot Activity',   desc: 'Quasi-periodic rotational modulation',          color: 'var(--color-class-starspot)' },
  { key: 'noise',             label: 'Instrument Noise',    desc: 'Pure systematics, no astrophysical signal',     color: 'var(--color-class-noise)' },
];

type LoadingStage = 'uploading' | 'cleaning' | 'previewing' | 'analyzing' | null;

const STAGE_LABELS: Record<string, string> = {
  uploading:  'Uploading & parsing...',
  cleaning:   'Cleaning & detrending...',
  previewing: 'Generating preview...',
  analyzing:  'Running BLS + CNN pipeline...',
};

export default function AnalyzePage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [preview, setPreview] = useState<LightCurvePreview | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [stage, setStage] = useState<LoadingStage>(null);
  const [error, setError] = useState<string | null>(null);
  const [realTargets, setRealTargets] = useState<RealTargetInfo[] | null>(null);
  const [showRealTargets, setShowRealTargets] = useState(false);
  
  const [ticInput, setTicInput] = useState('307210830');
  const [kicInput, setKicInput] = useState('11442793');
  const [dragOver, setDragOver] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [analysisParams, setAnalysisParams] = useState({
    min_period_days: 0.5,
    stellar_radius_rsun: 1.0,
    n_top_candidates: 5,
    fast_mode: true,
  });

  const setDataset = useCallback(async (dsId: string, name: string) => {
    setDatasetName(name);
    setStage('previewing');
    const prev = await api.preview(dsId);
    setPreview(prev);
    setUploadResult((u) => u ? { ...u, dataset_id: dsId } : {
      dataset_id: dsId, n_points_raw: prev.raw.length,
      n_points_after_cleaning: prev.cleaned.length,
      n_removed_outliers: 0, gap_fraction: 0, noise_ppm: 0,
      baseline_days: 0, cadence_minutes: 0, warnings: [],
    });
    setStage(null);
  }, []);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setError(null); setStage('uploading');
    try {
      const res = await api.uploadFile(file);
      setUploadResult(res);
      setStage('cleaning');
      await new Promise((r) => setTimeout(r, 300)); // visual pause
      await setDataset(res.dataset_id, file.name);
    } catch (e: any) {
      setError(e.message ?? 'Upload failed');
      setStage(null);
    }
  };

  const handleDemoGenerate = async (kind: string) => {
    setError(null); setStage('uploading');
    try {
      const seed = Math.floor(Math.random() * 99999);
      const res = await api.generateDemo(kind, seed, 90);
      await setDataset(res.dataset_id, `Demo: ${kind.replace('_',' ')} (seed ${seed})`);
    } catch (e: any) {
      setError(e.message ?? 'Demo generation failed');
      setStage(null);
    }
  };

  const handleRealTarget = async (kepoiName: string, displayName: string) => {
    setError(null); setStage('uploading');
    try {
      const res = await api.loadRealTarget(kepoiName, 90, Math.floor(Math.random() * 999));
      await setDataset(res.dataset_id, `${displayName} (real KOI parameters)`);
      setShowRealTargets(false);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load target');
      setStage(null);
    }
  };

  const handleTessFetch = async () => {
    const ticId = parseInt(ticInput, 10);
    if (isNaN(ticId)) { setError('Enter a valid TIC ID'); return; }
    setError(null); setStage('uploading');
    try {
      const res = await api.fetchTess(ticId);
      await setDataset(res.dataset_id, `TIC ${ticId} (TESS live)`);
    } catch (e: any) {
      setError(e.message ?? 'TESS fetch failed');
      setStage(null);
    }
  };

  const handleKeplerFetch = async () => {
    const kicId = parseInt(kicInput, 10);
    if (isNaN(kicId)) { setError('Enter a valid KIC ID'); return; }
    setError(null); setStage('uploading');
    try {
      const res = await api.fetchKepler(kicId);
      await setDataset(res.dataset_id, `KIC ${kicId} (Kepler live)`);
    } catch (e: any) {
      setError(e.message ?? 'Kepler fetch failed');
      setStage(null);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadResult) return;
    setError(null); setStage('analyzing');
    try {
      const res = await api.analyze({ dataset_id: uploadResult.dataset_id, ...analysisParams });
      navigate(`/results/${res.analysis_id}`);
    } catch (e: any) {
      setError(e.message ?? 'Analysis failed');
      setStage(null);
    }
  };

  const isLoading = stage !== null && stage !== 'analyzing';

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="h-16 flex items-center px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur sticky top-0 z-10">
        <h1 className="font-[var(--font-display)] text-[18px] font-semibold">New Analysis</h1>
        {uploadResult && (
          <button onClick={() => { setUploadResult(null); setPreview(null); setError(null); }}
            className="ml-auto flex items-center gap-1.5 text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-class-eb)] transition-colors">
            <X size={13}/> Clear dataset
          </button>
        )}
      </div>

      <div className="max-w-5xl mx-auto p-6 space-y-5">
        {/* Error banner */}
        {error && (
          <div className="flex items-start gap-3 px-4 py-3 rounded-lg bg-[var(--color-class-eb)]/10 border border-[var(--color-class-eb)]/30 text-[13px] text-[var(--color-class-eb)]">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <div className="flex-1">{error}</div>
            <button onClick={() => setError(null)} className="shrink-0 opacity-60 hover:opacity-100"><X size={14}/></button>
          </div>
        )}

        {/* Loading state */}
        {stage && (
          <div className="flex items-center justify-center gap-3 py-10 text-[var(--color-text-secondary)]">
            <Loader2 size={20} className="animate-spin text-[var(--color-accent)]" />
            <div>
              <div className="text-[14px] font-medium">{STAGE_LABELS[stage] ?? 'Processing...'}</div>
              {stage === 'analyzing' && (
                <div className="text-[11px] text-[var(--color-text-tertiary)] mt-1">
                  BLS period search → CNN classification → ensemble vetting → Integrated Gradients
                </div>
              )}
            </div>
          </div>
        )}

        {/* Dataset not yet loaded */}
        {!uploadResult && !stage && (
          <>
            {/* Upload drop zone */}
            <label
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
              className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl py-12 cursor-pointer transition-all ${
                dragOver
                  ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/[0.04]'
                  : 'border-[var(--color-border-default)] bg-[var(--color-bg-inset)] hover:border-[var(--color-accent)]/50'
              }`}>
              <div className="w-12 h-12 rounded-xl bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/25 flex items-center justify-center">
                <Upload size={22} className="text-[var(--color-accent)]" />
              </div>
              <div className="text-center">
                <div className="text-[14px] font-medium text-[var(--color-text-primary)]">
                  Drop a light curve file here, or click to browse
                </div>
                <div className="text-[11.5px] text-[var(--color-text-tertiary)] mt-1">
                  CSV/TSV/TXT — columns: <span className="font-mono-nums">time, flux, [flux_err]</span> &nbsp;·&nbsp; Kepler, TESS, or generic formats
                </div>
              </div>
              <input ref={fileInputRef} type="file" accept=".csv,.tsv,.txt,.dat" className="hidden"
                onChange={(e) => handleFiles(e.target.files)} />
            </label>

            <div className="grid grid-cols-3 gap-4">
              {/* Synthetic demos */}
              <Panel title="Synthetic Demos" icon={<Sparkles size={13} className="text-[var(--color-accent)]"/>}>
                <div className="space-y-1.5">
                  {DEMO_KINDS.map((d) => (
                    <button key={d.key} onClick={() => handleDemoGenerate(d.key)} disabled={isLoading}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] hover:border-[var(--color-accent)]/35 transition-colors text-left disabled:opacity-40">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: d.color }} />
                      <div>
                        <div className="text-[12px] font-medium text-[var(--color-text-primary)]">{d.label}</div>
                        <div className="text-[10px] text-[var(--color-text-tertiary)] leading-tight">{d.desc}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </Panel>

              {/* Real KOIs */}
              <Panel title="Real Kepler KOIs" icon={<Database size={13} className="text-[var(--color-accent)]"/>}>
                {!showRealTargets ? (
                  <button onClick={async () => { setShowRealTargets(true); if (!realTargets) setRealTargets(await api.realTargets()); }}
                    className="w-full flex flex-col items-center gap-2 border border-[var(--color-border-default)] rounded-lg py-8 hover:border-[var(--color-accent)]/40 transition-colors">
                    <Database size={22} className="text-[var(--color-text-tertiary)]" />
                    <span className="text-[12px] text-[var(--color-text-secondary)]">Browse confirmed KOIs</span>
                    <span className="text-[10px] text-[var(--color-text-tertiary)]">Confirmed planets + false positives</span>
                  </button>
                ) : (
                  <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                    {realTargets?.map((t) => {
                      const dColor = t.disposition === 'CONFIRMED' ? 'var(--color-class-planet)' : t.disposition === 'FALSE POSITIVE' ? 'var(--color-class-eb)' : 'var(--color-class-starspot)';
                      return (
                        <button key={t.kepoi_name} onClick={() => handleRealTarget(t.kepoi_name, t.kepler_name ?? t.kepoi_name)} disabled={isLoading}
                          className="w-full flex items-center justify-between px-2.5 py-2 rounded-lg bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] hover:border-[var(--color-accent)]/35 transition-colors text-left disabled:opacity-40">
                          <div>
                            <div className="text-[11.5px] font-semibold text-[var(--color-text-primary)]">{t.kepler_name ?? t.kepoi_name}</div>
                            <div className="text-[10px] font-mono-nums text-[var(--color-text-tertiary)]">P={t.period_days.toFixed(2)}d · {t.depth_ppm.toFixed(0)}ppm</div>
                          </div>
                          <span className="text-[9.5px] font-medium px-1.5 py-0.5 rounded" style={{ color: dColor, background: `color-mix(in srgb,${dColor} 12%,transparent)` }}>
                            {t.disposition.split(' ')[0]}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </Panel>

              {/* Live MAST fetch */}
              <Panel title="Live MAST Fetch" icon={<Satellite size={13} className="text-[var(--color-accent)]"/>}>
                <div className="space-y-2.5">
                  <div className="text-[10.5px] text-[var(--color-text-tertiary)] leading-relaxed">
                    Fetch real photometry directly from NASA MAST. Requires internet access to archive.stsci.edu.
                  </div>
                  {/* TESS */}
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wide">TESS (TIC ID)</div>
                    <div className="flex gap-1.5">
                      <input value={ticInput} onChange={(e) => setTicInput(e.target.value)}
                        placeholder="e.g. 307210830"
                        className="flex-1 bg-[var(--color-bg-inset)] border border-[var(--color-border-default)] rounded-md text-[11.5px] px-2 py-1.5 text-[var(--color-text-primary)] font-mono-nums focus:outline-none focus:border-[var(--color-accent)]/50" />
                      <button onClick={handleTessFetch} disabled={isLoading}
                        className="px-2.5 py-1.5 rounded-md bg-[var(--color-accent)]/15 border border-[var(--color-accent)]/25 text-[var(--color-accent)] text-[11px] font-medium hover:bg-[var(--color-accent)]/25 transition-colors disabled:opacity-40">
                        <Globe size={13}/>
                      </button>
                    </div>
                  </div>
                  {/* Kepler */}
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wide">Kepler (KIC ID)</div>
                    <div className="flex gap-1.5">
                      <input value={kicInput} onChange={(e) => setKicInput(e.target.value)}
                        placeholder="e.g. 11442793"
                        className="flex-1 bg-[var(--color-bg-inset)] border border-[var(--color-border-default)] rounded-md text-[11.5px] px-2 py-1.5 text-[var(--color-text-primary)] font-mono-nums focus:outline-none focus:border-[var(--color-accent)]/50" />
                      <button onClick={handleKeplerFetch} disabled={isLoading}
                        className="px-2.5 py-1.5 rounded-md bg-[var(--color-accent)]/15 border border-[var(--color-accent)]/25 text-[var(--color-accent)] text-[11px] font-medium hover:bg-[var(--color-accent)]/25 transition-colors disabled:opacity-40">
                        <Globe size={13}/>
                      </button>
                    </div>
                  </div>
                </div>
              </Panel>
            </div>
          </>
        )}

        {/* Dataset loaded — show preview + run button */}
        {uploadResult && preview && !stage && (
          <div className="space-y-4">
            {/* Dataset info bar */}
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-[var(--color-bg-panel)] border border-[var(--color-border-default)]">
              <CheckCircle2 size={16} className="text-[var(--color-class-planet)] shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-medium text-[var(--color-text-primary)] truncate">{datasetName}</div>
                <div className="text-[10.5px] text-[var(--color-text-tertiary)] mt-0.5">
                  {uploadResult.n_points_raw.toLocaleString()} raw pts → {uploadResult.n_points_after_cleaning.toLocaleString()} clean &nbsp;·&nbsp;
                  noise {uploadResult.noise_ppm.toFixed(0)} ppm &nbsp;·&nbsp;
                  {uploadResult.baseline_days.toFixed(1)} d baseline &nbsp;·&nbsp;
                  {uploadResult.cadence_minutes.toFixed(1)} min cadence
                </div>
              </div>
              {uploadResult.n_removed_outliers > 0 && (
                <span className="text-[10px] bg-[var(--color-class-starspot)]/10 text-[var(--color-class-starspot)] border border-[var(--color-class-starspot)]/20 px-2 py-0.5 rounded-full shrink-0">
                  {uploadResult.n_removed_outliers} outliers removed
                </span>
              )}
            </div>

            {/* Warnings */}
            {uploadResult.warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-2 px-3 py-2.5 rounded-lg text-[11.5px] text-[var(--color-class-starspot)] bg-[var(--color-class-starspot)]/8 border border-[var(--color-class-starspot)]/20">
                <AlertTriangle size={13} className="mt-0.5 shrink-0" />{w}
              </div>
            ))}

            {/* Light curve preview */}
            <LightCurveViewer raw={preview.raw} cleaned={preview.cleaned} trend={preview.trend} height={320} />

            {/* Analysis params */}
            <div>
              <button onClick={() => setAdvancedOpen((v) => !v)}
                className="flex items-center gap-1.5 text-[12px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors mb-2">
                <ChevronDown size={14} className={`transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                Analysis parameters
              </button>
              {advancedOpen && (
                <div className="grid grid-cols-4 gap-3 p-4 rounded-lg bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)]">
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1 block">Min period (days)</label>
                    <input type="number" step="0.1" min="0.1" max="10" value={analysisParams.min_period_days}
                      onChange={(e) => setAnalysisParams((p) => ({ ...p, min_period_days: +e.target.value }))}
                      className="w-full bg-[var(--color-bg-panel)] border border-[var(--color-border-default)] rounded px-2 py-1.5 text-[12px] font-mono-nums text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]/50" />
                  </div>
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1 block">Stellar radius (R☉)</label>
                    <input type="number" step="0.1" min="0.1" max="10" value={analysisParams.stellar_radius_rsun}
                      onChange={(e) => setAnalysisParams((p) => ({ ...p, stellar_radius_rsun: +e.target.value }))}
                      className="w-full bg-[var(--color-bg-panel)] border border-[var(--color-border-default)] rounded px-2 py-1.5 text-[12px] font-mono-nums text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]/50" />
                  </div>
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1 block">Max candidates</label>
                    <select value={analysisParams.n_top_candidates}
                      onChange={(e) => setAnalysisParams((p) => ({ ...p, n_top_candidates: +e.target.value }))}
                      className="w-full bg-[var(--color-bg-panel)] border border-[var(--color-border-default)] rounded px-2 py-1.5 text-[12px] text-[var(--color-text-primary)] focus:outline-none">
                      {[1,2,3,5,8,10].map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1 block">BLS precision</label>
                    <select value={analysisParams.fast_mode ? 'fast' : 'precise'}
                      onChange={(e) => setAnalysisParams((p) => ({ ...p, fast_mode: e.target.value === 'fast' }))}
                      className="w-full bg-[var(--color-bg-panel)] border border-[var(--color-border-default)] rounded px-2 py-1.5 text-[12px] text-[var(--color-text-primary)] focus:outline-none">
                      <option value="fast">Fast (~3s)</option>
                      <option value="precise">Precise (~12s)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Run button */}
            <button onClick={handleAnalyze} disabled={stage === 'analyzing'}
              className="w-full flex items-center justify-center gap-2.5 py-4 rounded-xl font-semibold text-[15px] transition-all glow-accent disabled:opacity-60"
              style={{ background: 'var(--color-accent)', color: '#1a1206' }}>
              {stage === 'analyzing' ? (
                <><Loader2 size={17} className="animate-spin" /> Running detection pipeline...</>
              ) : (
                <><Play size={17} fill="currentColor" /> Run Detection &amp; Classification</>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
