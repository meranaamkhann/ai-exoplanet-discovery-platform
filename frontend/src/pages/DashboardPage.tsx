import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Telescope, Upload, Database, History, Activity, Target, Layers, 
  Zap, TrendingUp, AlertTriangle, CheckCircle2, Clock
} from 'lucide-react';
import api from '../lib/api';
import type { ModelInfo, HistoryResponse } from '../types/api';
import { Panel, ClassBadge } from '../components/ui';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';

export default function DashboardPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [health, setHealth] = useState<any | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    api.modelInfo().then(setModelInfo).catch(() => {});
    api.history(8, 0).then(setHistory).catch(() => {});
  }, []);

  // Per-class performance data for radar chart (from our trained model metrics)
  const radarData = modelInfo ? [
    { class: 'Planet', value: 77.3 },
    { class: 'Eclipsing\nBinary', value: 98.3 },
    { class: 'Stellar\nVar', value: 80.6 },
    { class: 'Starspot', value: 90.5 },
    { class: 'Noise', value: 81.6 },
  ] : [];

  const recentItems = history?.items.slice(0, 6) ?? [];
  const totalAnalyses = history?.total ?? 0;
  const nPlanetCandidates = history?.items.filter((h) => h.top_label === 'PLANET_TRANSIT' && !h.is_likely_false_positive).length ?? 0;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="h-16 flex items-center px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur sticky top-0 z-10">
        <h1 className="font-[var(--font-display)] text-[18px] font-semibold">Mission Control</h1>
        <div className="ml-auto flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${health?.model_loaded ? 'bg-[var(--color-class-planet)]' : 'bg-[var(--color-class-noise)]'}`} />
          <span className="text-[11px] text-[var(--color-text-tertiary)]">
            {health?.model_loaded ? 'Pipeline online' : 'Model not loaded'}
          </span>
        </div>
      </div>

      <div className="p-6 space-y-5 max-w-7xl">
        {/* Hero */}
        <div className="relative rounded-2xl border border-[var(--color-border-subtle)] overflow-hidden bg-gradient-to-br from-[var(--color-bg-panel)] via-[var(--color-bg-panel-raised)] to-[var(--color-bg-panel)]">
          <div className="absolute inset-0 bg-grid opacity-60 pointer-events-none" />
          <div className="absolute -right-20 -top-20 w-80 h-80 rounded-full" style={{ background: 'radial-gradient(circle, rgba(245,166,35,0.08) 0%, transparent 70%)' }} />
          <div className="relative p-8">
            <div className="flex items-start justify-between gap-8">
              <div className="max-w-2xl">
                <div className="flex items-center gap-2 mb-3">
                  <Telescope size={16} className="text-[var(--color-accent)]" />
                  <span className="text-[10.5px] uppercase tracking-widest text-[var(--color-text-tertiary)] font-medium">ExoNova · ISRO PS-7</span>
                </div>
                <h2 className="font-[var(--font-display)] text-[28px] font-bold text-[var(--color-text-primary)] mb-3 leading-tight">
                  AI-Powered Exoplanet Discovery<br/>from Noisy Stellar Light Curves
                </h2>
                <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed mb-6">
                  Upload any Kepler/TESS light curve or fetch live from MAST. The pipeline detrends, 
                  runs Box Least Squares period search, classifies with a calibrated dual-view 1D-CNN, 
                  applies domain-expert vetting rules, and explains every decision with Integrated Gradients saliency maps.
                </p>
                <div className="flex gap-3">
                  <Link to="/analyze"
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-[13px] transition-all glow-accent"
                    style={{ background: 'var(--color-accent)', color: '#1a1206' }}>
                    <Upload size={15} /> Start New Analysis
                  </Link>
                  <Link to="/real-data"
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[var(--color-border-strong)] text-[13px] font-medium text-[var(--color-text-secondary)] hover:border-[var(--color-accent)]/40 hover:text-[var(--color-text-primary)] transition-all">
                    <Database size={15} /> Real NASA Targets
                  </Link>
                </div>
              </div>
              {/* Quick stats */}
              <div className="shrink-0 grid grid-cols-2 gap-3 min-w-[220px]">
                <div className="col-span-2 bg-[var(--color-bg-inset)] rounded-xl border border-[var(--color-border-subtle)] p-3 text-center">
                  <div className="text-[32px] font-bold font-mono-nums text-[var(--color-accent)]">
                    {modelInfo ? `${(modelInfo.test_accuracy * 100).toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mt-0.5">Test Accuracy</div>
                </div>
                <div className="bg-[var(--color-bg-inset)] rounded-xl border border-[var(--color-border-subtle)] p-3 text-center">
                  <div className="text-[20px] font-bold font-mono-nums text-[var(--color-text-primary)]">{totalAnalyses}</div>
                  <div className="text-[9.5px] text-[var(--color-text-tertiary)] mt-0.5">Analyses</div>
                </div>
                <div className="bg-[var(--color-bg-inset)] rounded-xl border border-[var(--color-border-subtle)] p-3 text-center">
                  <div className="text-[20px] font-bold font-mono-nums text-[var(--color-class-planet)]">{nPlanetCandidates}</div>
                  <div className="text-[9.5px] text-[var(--color-text-tertiary)] mt-0.5">Planet Cands.</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-5">
          {/* Model performance radar */}
          <Panel title="Per-Class F1 Score" icon={<TrendingUp size={13} className="text-[var(--color-accent)]"/>}>
            {radarData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={180}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="var(--color-border-subtle)" />
                    <PolarAngleAxis dataKey="class" tick={{ fill: 'var(--color-text-tertiary)', fontSize: 10 }} />
                    <Radar name="F1" dataKey="value" stroke="var(--color-accent)" fill="var(--color-accent)" fillOpacity={0.2} strokeWidth={2} />
                    <Tooltip formatter={(v: any) => [`${v}%`, 'F1']} contentStyle={{ background: 'var(--color-bg-panel-raised)', border: '1px solid var(--color-border-default)', borderRadius: 6, fontSize: 11 }} />
                  </RadarChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-2 gap-1.5 mt-1">
                  <div className="text-center">
                    <div className="font-mono-nums text-[13px] font-semibold text-[var(--color-text-primary)]">{modelInfo!.n_train.toLocaleString()}</div>
                    <div className="text-[9.5px] text-[var(--color-text-tertiary)]">Training samples</div>
                  </div>
                  <div className="text-center">
                    <div className="font-mono-nums text-[13px] font-semibold text-[var(--color-text-primary)]">{modelInfo!.macro_f1.toFixed(3)}</div>
                    <div className="text-[9.5px] text-[var(--color-text-tertiary)]">Macro F1</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-40 text-[var(--color-text-tertiary)] text-[12px]">
                <Activity size={16} className="mr-2" /> Model not loaded
              </div>
            )}
          </Panel>

          {/* Pipeline stages */}
          <Panel title="Detection Pipeline" icon={<Layers size={13} className="text-[var(--color-accent)]"/>}>
            <div className="space-y-2.5">
              {[
                { step: '01', label: 'Ingest & validate', desc: 'CSV/TSV/FITS, gap detection', icon: Upload },
                { step: '02', label: 'Clean & detrend', desc: '75th-pct rolling filter, sigma clip', icon: Zap },
                { step: '03', label: 'BLS period search', desc: 'Frequency-resolved grid, multi-planet', icon: Target },
                { step: '04', label: 'CNN classify', desc: 'Dual-view 1D-CNN, 353K params', icon: Activity },
                { step: '05', label: 'Ensemble vetting', desc: 'Odd/even, secondary eclipse, V-shape', icon: CheckCircle2 },
                { step: '06', label: 'Explain & rank', desc: 'Integrated Gradients saliency', icon: TrendingUp },
              ].map(({ step, label, desc, icon: Icon }) => (
                <div key={step} className="flex items-center gap-3">
                  <span className="text-[10px] font-mono-nums font-bold text-[var(--color-accent)] w-5 shrink-0">{step}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[12px] font-medium text-[var(--color-text-primary)]">{label}</div>
                    <div className="text-[10px] text-[var(--color-text-tertiary)]">{desc}</div>
                  </div>
                  <Icon size={13} className="text-[var(--color-text-disabled)] shrink-0" />
                </div>
              ))}
            </div>
          </Panel>

          {/* Recent analyses */}
          <Panel title="Recent Analyses" icon={<History size={13} className="text-[var(--color-accent)]"/>}
            action={<Link to="/history" className="text-[11px] text-[var(--color-accent)] hover:underline">View all</Link>}>
            {recentItems.length > 0 ? (
              <div className="space-y-2">
                {recentItems.map((h) => (
                  <Link key={h.analysis_id} to={`/results/${h.analysis_id}`}
                    className="flex items-center gap-2.5 p-2 rounded-lg hover:bg-[var(--color-bg-inset)] transition-colors -mx-1 px-1">
                    <div className="flex-1 min-w-0">
                      <div className="text-[11.5px] font-medium text-[var(--color-text-primary)] truncate">{h.dataset_name}</div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <ClassBadge label={h.top_label} />
                        {h.is_likely_false_positive && <AlertTriangle size={10} className="text-[var(--color-class-eb)]" />}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="font-mono-nums text-[11px] text-[var(--color-text-primary)]">{(h.top_confidence * 100).toFixed(0)}%</div>
                      <div className="text-[9.5px] text-[var(--color-text-tertiary)]">{new Date(h.created_at).toLocaleDateString()}</div>
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-[var(--color-text-tertiary)] gap-2">
                <Clock size={20} />
                <span className="text-[12px]">No analyses yet</span>
                <Link to="/analyze" className="text-[11px] text-[var(--color-accent)]">Run your first analysis →</Link>
              </div>
            )}
          </Panel>
        </div>

        {/* Signal classes */}
        <Panel title="Signal Classification Framework" icon={<Target size={13} className="text-[var(--color-accent)]"/>}>
          <div className="grid grid-cols-5 gap-3">
            {[
              { label: 'PLANET_TRANSIT', f1: '77.3%', desc: 'Flat-bottomed periodic dip. No secondary eclipse. Odd/even depths consistent.', key_feat: 'Flat bottom + symmetric' },
              { label: 'ECLIPSING_BINARY', f1: '98.3%', desc: 'Deep V-shaped eclipses, secondary at phase 0.5, large odd/even depth difference.', key_feat: 'Secondary eclipse' },
              { label: 'STELLAR_VARIABILITY', f1: '80.6%', desc: 'Continuous sinusoidal brightness oscillations. Multiple pulsation modes possible.', key_feat: 'Smooth periodic modulation' },
              { label: 'STARSPOT_ACTIVITY', f1: '90.5%', desc: 'Quasi-periodic rotational modulation with evolving amplitude (spot growth/decay).', key_feat: 'Asymmetric quasi-periodic' },
              { label: 'INSTRUMENT_NOISE', f1: '81.6%', desc: 'No coherent astrophysical signal. Dominated by systematic artefacts and red noise.', key_feat: 'Low periodicity strength' },
            ].map(({ label, f1, desc, key_feat }) => (
              <div key={label} className="bg-[var(--color-bg-inset)] rounded-xl border border-[var(--color-border-subtle)] p-3">
                <ClassBadge label={label} />
                <div className="mt-2 font-mono-nums text-[13px] font-semibold text-[var(--color-text-primary)]">{f1} F1</div>
                <p className="mt-1.5 text-[10.5px] text-[var(--color-text-tertiary)] leading-relaxed">{desc}</p>
                <div className="mt-2 text-[9.5px] font-medium text-[var(--color-accent)] uppercase tracking-wide">{key_feat}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
