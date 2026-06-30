import { useEffect, useState } from 'react';
import { Database, Loader2, Play, ChevronRight, Info } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import api from '../lib/api';
import type { RealTargetInfo } from '../types/api';
import { Panel } from '../components/ui';

const DISPOSITION_COLORS: Record<string, string> = {
  CONFIRMED: 'var(--color-class-planet)',
  'FALSE POSITIVE': 'var(--color-class-eb)',
  CANDIDATE: 'var(--color-class-starspot)',
};

export default function RealDataPage() {
  const navigate = useNavigate();
  const [targets, setTargets] = useState<RealTargetInfo[]>([]);
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [selected, setSelected] = useState<RealTargetInfo | null>(null);

  useEffect(() => { api.realTargets().then(setTargets); }, []);

  const handleLoad = async (t: RealTargetInfo) => {
    setAnalyzing(t.kepoi_name);
    try {
      const res = await api.loadRealTarget(t.kepoi_name, 90, 4);
      const analysis = await api.analyze({ dataset_id: res.dataset_id, fast_mode: false, n_top_candidates: 3 });
      navigate(`/results/${analysis.analysis_id}`);
    } catch (e) {
      console.error(e);
    } finally {
      setAnalyzing(null);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="h-16 flex items-center px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur sticky top-0 z-10">
        <div>
          <h1 className="font-[var(--font-display)] text-[18px] font-semibold">
            Validation Dataset
          </h1>

          <p className="text-[11px] text-[var(--color-text-tertiary)] mt-0.5">
            Curated NASA Kepler Objects of Interest for quantitative AI pipeline validation
          </p>
        </div>
      </div>

      <div className="p-6 max-w-7xl space-y-5">
              <div className="grid grid-cols-4 gap-4">

        <Panel>

          <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)]">
            Validated Targets
          </div>

          <div className="mt-2 text-3xl font-bold font-mono-nums text-[var(--color-accent)]">
            {targets.length}
          </div>

        </Panel>

        <Panel>

          <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)]">
            Confirmed Planets
          </div>

          <div className="mt-2 text-3xl font-bold font-mono-nums text-[var(--color-class-planet)]">
            {targets.filter(t => t.disposition==="CONFIRMED").length}
          </div>

        </Panel>

        <Panel>

          <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)]">
            False Positives
          </div>

          <div className="mt-2 text-3xl font-bold font-mono-nums text-[var(--color-class-eb)]">
            {targets.filter(t=>t.disposition==="FALSE POSITIVE").length}
          </div>

        </Panel>

        <Panel>

          <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)]">
            Candidates
          </div>

          <div className="mt-2 text-3xl font-bold font-mono-nums text-[var(--color-class-starspot)]">
            {targets.filter(t=>t.disposition==="CANDIDATE").length}
          </div>

        </Panel>

      </div>
        <div className="grid grid-cols-[1fr_320px] gap-5">
          {/* Target list */}
          <Panel title="Curated Kepler KOI Validation Set"
            icon={<Database size={13} className="text-[var(--color-accent)]"/>}>
            <p className="text-[11.5px] text-[var(--color-text-tertiary)] mb-4 leading-relaxed">
              Real Kepler Objects of Interest from the NASA Exoplanet Archive — confirmed exoplanets,
              certified false positives, and edge cases. Each target is synthesized at its actual
              published orbital parameters for quantitative pipeline validation.
            </p>
            <div className="space-y-2">
              {targets.map((t) => {
                const color = DISPOSITION_COLORS[t.disposition] ?? 'var(--color-text-tertiary)';
                const isAnalyzing = analyzing === t.kepoi_name;
                return (
                  <div key={t.kepoi_name}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all cursor-pointer ${
                      selected?.kepoi_name === t.kepoi_name
                        ? 'border-[var(--color-accent)]/50 bg-[var(--color-accent)]/[0.04]'
                        : 'border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] hover:border-[var(--color-border-strong)]'
                    }`}
                    onClick={() => setSelected(t)}>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-[13px] font-semibold text-[var(--color-text-primary)]">
                          {t.kepler_name ?? t.kepoi_name}
                        </span>
                        <span className="text-[9.5px] text-[var(--color-text-tertiary)] font-mono-nums">{t.kepoi_name}</span>
                        <span className="text-[9.5px] font-medium px-1.5 py-0.5 rounded border"
                          style={{ color, borderColor: `color-mix(in srgb,${color} 35%,transparent)`, background: `color-mix(in srgb,${color} 10%,transparent)` }}>
                          {t.disposition}
                        </span>
                      </div>
                      <div className="flex gap-4 mt-1 text-[10.5px] font-mono-nums text-[var(--color-text-tertiary)]">
                        <span>P = {t.period_days.toFixed(4)} d</span>
                        <span>depth = {t.depth_ppm.toFixed(0)} ppm</span>
                        {t.planet_radius_re && <span>Rₚ = {t.planet_radius_re.toFixed(2)} R⊕</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button onClick={(e) => { e.stopPropagation(); setSelected(t); }}
                        className="p-1.5 rounded-lg hover:bg-[var(--color-bg-panel)] text-[var(--color-text-tertiary)] transition-colors">
                        <Info size={13} />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleLoad(t); }}
                        disabled={!!analyzing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11.5px] font-medium border transition-colors disabled:opacity-40"
                        style={{ color: 'var(--color-accent)', borderColor: 'color-mix(in srgb,var(--color-accent) 35%,transparent)', background: 'color-mix(in srgb,var(--color-accent) 8%,transparent)' }}>
                        {isAnalyzing ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                        {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                        {!isAnalyzing && <ChevronRight size={11} />}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </Panel>

          {/* Detail panel */}
          <div className="space-y-4">
            {selected ? (
              <Panel title={selected.kepler_name ?? selected.kepoi_name}>
               <div
                className="mb-4 px-3 py-2 rounded-lg border text-[11px] font-medium"
                style={{
                background:"color-mix(in srgb,var(--color-class-planet) 10%,transparent)",
                borderColor:"color-mix(in srgb,var(--color-class-planet) 30%,transparent)",
                color:"var(--color-class-planet)"
                }}
                >

                ✓ Official NASA Validation Target

                </div>
                <div className="space-y-3">
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1">Disposition</div>
                    <div className="text-[13px] font-semibold" style={{ color: DISPOSITION_COLORS[selected.disposition] ?? 'inherit' }}>
                      {selected.disposition}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1">Vetting Notes</div>
                    <p className="text-[11.5px] text-[var(--color-text-secondary)] leading-relaxed">{selected.disposition_reason}</p>
                  </div>
                  <div>
                      <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1">
                        Scientific Importance
                      </div>

                      <p className="text-[11.5px] text-[var(--color-text-secondary)] leading-relaxed">
                        This object is part of the curated NASA Kepler Object of Interest (KOI)
                        validation dataset used to benchmark the AI detection pipeline against
                        published astronomical catalogues. Its known classification provides an
                        objective reference for evaluating model performance.
                      </p>
                    </div>
                    <hr className="border-[var(--color-border-subtle)]" />
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    {[
                      ['Orbital period', `${selected.period_days.toFixed(6)} d`],
                      ['Transit depth', `${selected.depth_ppm.toFixed(1)} ppm`],
                      ...(selected.planet_radius_re ? [['Planet radius', `${selected.planet_radius_re.toFixed(3)} R⊕`]] : []),
                      ['KIC ID', `${selected.kic_id}`],
                    ].map(([l, v]) => (
                      <div key={l} className="bg-[var(--color-bg-inset)] rounded-lg px-2.5 py-2 border border-[var(--color-border-subtle)]">
                        <div className="text-[9.5px] text-[var(--color-text-tertiary)]">{l}</div>
                        <div className="font-mono-nums font-semibold text-[var(--color-text-primary)]">{v}</div>
                      </div>
                    ))}
                  </div>
                  <button onClick={() => handleLoad(selected)}
                    disabled={!!analyzing}
                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl font-semibold text-[13px] transition-all disabled:opacity-50"
                    style={{ background: 'var(--color-accent)', color: '#1a1206' }}>
                    {analyzing === selected.kepoi_name ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                    Run Full Analysis
                  </button>
                </div>
              </Panel>
            ) : (
              <div className="rounded-xl border border-[var(--color-border-subtle)] p-6 text-center text-[var(--color-text-tertiary)]">
                <Database size={24} className="mx-auto mb-2 opacity-40" />
                <p className="text-[12px]">Select a target to see details</p>
              </div>
            )}

            {/* Method note */}
            <Panel>
              <p className="text-[11px] text-[var(--color-text-tertiary)] leading-relaxed">
                <strong className="text-[var(--color-text-secondary)]">Validation method:</strong>{' '}
                Each target is synthesized using its actual published ephemeris (period, depth, duration,
                stellar parameters) from the NASA Exoplanet Archive cumulative KOI table, then injected
                with realistic Kepler-cadence noise for quantitative pipeline benchmarking.
                Live MAST fetch is available via <span className="font-mono-nums text-[var(--color-accent)]">/api/fetch-kepler/{'{'}kic_id{'}'}</span> when internet is available.
              </p>
            </Panel>
              <Panel title="Validation Workflow"
              icon={<Database size={13} className="text-[var(--color-accent)]" />}
            >
              <div className="space-y-2 text-[11px]">

                <div>NASA KOI Archive</div>

                <div className="opacity-50">↓</div>

                <div>Synthetic Reconstruction</div>

                <div className="opacity-50">↓</div>

                <div>Light Curve Preprocessing</div>

                <div className="opacity-50">↓</div>

                <div>Box Least Squares Transit Search</div>

                <div className="opacity-50">↓</div>

                <div>Dual-View CNN Classification</div>

                <div className="opacity-50">↓</div>

                <div>Explainable AI</div>

                <div className="opacity-50">↓</div>

                <div>Scientific Report</div>

              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
