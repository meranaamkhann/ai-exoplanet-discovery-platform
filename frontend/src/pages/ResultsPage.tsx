import { useEffect, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Loader2,
  Orbit,
  Gauge,
  Ruler,
  Globe2,
  AlertTriangle,
  Download,
  ChevronLeft,
  CheckCircle2,
  XCircle,
  Eye,
  Star,
  Database,
} from 'lucide-react';
import api from '../lib/api';
import type { CandidateResult, ModelInfo } from '../types/api';
import CandidateCard from '../components/CandidateCard';
import PhaseFoldChart from '../components/PhaseFoldChart';
import { Panel, StatTile, ClassBadge, ConfidenceBar, EvidenceTag } from '../components/ui';

// Per-class description copy shown in the explanation panel
const CLASS_DESCRIPTIONS: Record<string, string> = {
  PLANET_TRANSIT:
    'A flat-bottomed periodic dimming consistent with an opaque disk transiting the stellar face. ' +
    'The signal shows no significant secondary eclipse at phase 0.5, and odd/even transit depths ' +
    'are statistically consistent — both ruling against an eclipsing stellar companion.',
  ECLIPSING_BINARY:
    'Periodic brightness variations caused by two stars mutually eclipsing. ' +
    'Key diagnostics: a secondary eclipse is detected near phase 0.5, odd/even transit depths ' +
    'differ significantly (primary/secondary alternating), and/or the eclipse is V-shaped (grazing geometry).',
  STELLAR_VARIABILITY:
    'Smooth, continuous brightness oscillations intrinsic to the star — pulsation modes or ' +
    'non-radial oscillations. Unlike transits, the light curve shows no flat-bottomed dip; ' +
    'flux changes continuously throughout the "transit" window.',
  STARSPOT_ACTIVITY:
    'Quasi-periodic rotational modulation from magnetically active surface regions rotating in/out of view. ' +
    'The signal amplitude evolves over time (spot growth/decay), and the profile is typically ' +
    'asymmetric rather than symmetric around the minimum.',
  INSTRUMENT_NOISE:
    'No coherent astrophysical periodic signal detected above the noise floor. ' +
    'The dominant signal is likely instrumental systematics, momentum-dump artefacts, ' +
    'or thermally-driven spacecraft trends rather than a stellar or planetary phenomenon.',
};

export default function ResultsPage() {
  const { analysisId } = useParams();
  const [data, setData] = useState<any | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusUpdating, setStatusUpdating] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo| null>(null);
 useEffect(() => {
    if (!analysisId) return;

      Promise.all([
      api.getAnalysis(analysisId),
      api.modelInfo(),
      ])
      .then(([analysis, model]) => {
      setData(analysis);
      setModelInfo(model);

      if (analysis.candidates?.length) {
        setSelectedId(analysis.candidates[0].candidate_id);
      }
      })
      .catch((err) => {
      console.error(err);
      })
      .finally(() => {
      setLoading(false);
      });

     }, [analysisId]);

  const handleStatusUpdate = useCallback(async (candidateId: string, status: string) => {
    setStatusUpdating(true);
    await api.updateCandidateStatus(candidateId, status);
    setData((prev: any) => ({
      ...prev,
      candidates: prev.candidates.map((c: any) =>
        c.candidate_id === candidateId ? { ...c, status } : c
      ),
    }));
    setStatusUpdating(false);
  }, []);

  const handleExport = useCallback(() => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exonova_analysis_${analysisId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data, analysisId]);

  const handleDownloadPDF = useCallback(() => {
    // Uses server-side PDF generation endpoint
    window.open(`/api/analyses/${analysisId}/report.pdf`, '_blank');
  }, [analysisId]);

  const handleExportCSV = useCallback(() => {
    if (!data?.candidates) return;
    const headers = ['rank','label','confidence','period_d','duration_h','depth_ppm','snr','planet_radius_Re','false_positive_flags','status'];
    const rows = data.candidates.map((c: CandidateResult) => [
      c.rank, c.final_label, c.final_confidence.toFixed(4),
      c.features.period_days.toFixed(6), c.features.duration_hours.toFixed(4),
      c.features.depth_ppm.toFixed(2), c.features.snr.toFixed(2),
      c.features.planet_radius_re_estimate.toFixed(3),
      c.false_positive_flags.join('|'), (c as any).status ?? 'pending',
    ]);
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exonova_candidates_${analysisId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data, analysisId]);

  if (loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-[var(--color-text-tertiary)]">
        <Loader2 size={28} className="animate-spin text-[var(--color-accent)]" />
        <span className="text-[13px]">Loading analysis results...</span>
      </div>
    );
  }

  if (!data || !data.candidates?.length) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8 text-center">
        <AlertTriangle size={32} className="text-[var(--color-class-starspot)]" />
        <p className="text-[15px] font-medium text-[var(--color-text-primary)]">No candidates detected</p>
        <p className="text-[13px] text-[var(--color-text-tertiary)] max-w-sm">
          No periodic signal was found above the detection threshold. Try uploading a longer baseline or adjusting the min period parameter.
        </p>
        <Link to="/analyze" className="mt-2 px-4 py-2 rounded-lg bg-[var(--color-accent)]/15 border border-[var(--color-accent)]/25 text-[var(--color-accent)] text-[13px] font-medium hover:bg-[var(--color-accent)]/25 transition-colors">
          ← New analysis
        </Link>
      </div>
    );
  }

  const candidates: CandidateResult[] = data.candidates;
  const selected = candidates.find((c) => c.candidate_id === selectedId) ?? candidates[0];
  const f = selected.features;
  const currentStatus: string = (selected as any).status ?? 'pending';

  function getConfidenceLabel(confidence: number) {
  if (confidence >= 0.9) return "Very High";
  if (confidence >= 0.75) return "High";
  if (confidence >= 0.6) return "Medium";
  if (confidence >= 0.4) return "Low";
  return "Very Low";
}

  const STATUS_ACTIONS = [
    { status: 'confirmed', label: 'Confirm', icon: CheckCircle2, color: 'var(--color-class-planet)' },
    { status: 'needs_review', label: 'Review', icon: Eye, color: 'var(--color-class-starspot)' },
    { status: 'rejected', label: 'Reject', icon: XCircle, color: 'var(--color-class-eb)' },
  ];

  return (
    <div className="flex-1 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <Link to="/analyze" className="flex items-center gap-1.5 text-[12px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors">
            <ChevronLeft size={14} /> New Analysis
          </Link>
          <span className="text-[var(--color-border-default)]">/</span>
          <h1 className="font-[var(--font-display)] text-[16px] font-semibold">Results</h1>
          <span className="font-mono-nums text-[10px] text-[var(--color-text-disabled)] hidden md:block">{analysisId}</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium text-[var(--color-text-secondary)] border border-[var(--color-border-default)] hover:border-[var(--color-accent)]/40 hover:text-[var(--color-text-primary)] transition-colors">
            <Download size={13} /> CSV
          </button>
          <button onClick={handleDownloadPDF}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium text-[var(--color-accent)] border border-[var(--color-accent)]/30 hover:bg-[var(--color-accent)]/10 transition-colors">
            <Download size={13} /> Report PDF
          </button>
          <button onClick={handleExport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium text-[var(--color-text-secondary)] border border-[var(--color-border-default)] hover:border-[var(--color-accent)]/40 hover:text-[var(--color-text-primary)] transition-colors">
            <Download size={13} /> JSON
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden flex">
        {/* Left: candidate list */}
        <div className="w-72 shrink-0 border-r border-[var(--color-border-subtle)] flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-[var(--color-border-subtle)]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)] font-medium">
                {candidates.length} candidate{candidates.length !== 1 ? 's' : ''} found
              </span>
              <span className="text-[10px] font-mono-nums text-[var(--color-text-disabled)]">
                {data.processing_time_seconds?.toFixed(1)}s
              </span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {candidates.map((c) => (
              <CandidateCard
                key={c.candidate_id} candidate={c}
                selected={c.candidate_id === selected.candidate_id}
                onClick={() => setSelectedId(c.candidate_id)}
              />
            ))}
          </div>
          {data.observation_summary && (
            <div className="border-t border-[var(--color-border-subtle)] p-3">
              <p className="text-[10.5px] text-[var(--color-text-tertiary)] leading-relaxed">{data.observation_summary}</p>
            </div>
          )}
        </div>

        {/* Right: detail */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Executive Summary */}

<Panel title="Final AI Assessment">
  <div className="grid grid-cols-3 gap-6">

    {/* Left */}

    <div>

        <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-2">
            Prediction
        </div>

        <div className="text-3xl font-bold text-[var(--color-text-primary)]">
            {selected.final_label.replace(/_/g, ' ')}
        </div>

        <div className="mt-5">

            <div className="text-[11px] text-[var(--color-text-tertiary)] mb-1">
                Confidence
            </div>

            <ConfidenceBar
                value={selected.final_confidence}
                height={12}
            />

            <div className="mt-2 text-xl font-bold font-mono-nums">
                {(selected.final_confidence * 100).toFixed(1)}%
            </div>

            <div className="text-[12px] text-[var(--color-text-secondary)]">
                {getConfidenceLabel(selected.final_confidence)}
            </div>

        </div>

    </div>

    {/* Middle */}

    <div>

        <StatTile
            label="Processing Time"
            value={`${data.processing_time_seconds.toFixed(2)} s`}
        />

        <div className="mt-3"/>

        <StatTile
            label="Candidate Rank"
            value={`#${selected.rank}`}
        />

        <div className="mt-3"/>

        <StatTile
            label="False Positive Risk"
            value={
                selected.is_likely_false_positive
                    ? "Elevated"
                    : "Low"
            }
        />

    </div>

    {/* Right */}

    <div>

        <h3 className="font-semibold text-[15px] mb-3">

            Scientific Recommendation

        </h3>

        <p className="text-[13px] leading-relaxed text-[var(--color-text-secondary)]">

            {selected.final_confidence >= 0.90

                ? "This candidate exhibits a highly significant transit signature. Recommend immediate follow-up photometric observations and independent confirmation."

                : selected.final_confidence >= 0.75

                ? "Promising planetary transit candidate. Additional observations are recommended to confirm the detection."

                : selected.final_confidence >= 0.60

                ? "Moderate confidence detection. Additional data and further validation are required."

                : "Low-confidence detection. More observations are required before classification."}

        </p>

    </div>

</div>
</Panel>

{/* Header card */}
          {/* Header card */}
          <Panel>
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-[11px] font-mono-nums text-[var(--color-text-tertiary)] bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] rounded px-1.5 py-0.5">
                    #{selected.rank}
                  </span>
                  <ClassBadge label={selected.final_label} />
                  {selected.is_likely_false_positive && (
                    <span className="flex items-center gap-1 text-[10.5px] text-[var(--color-class-eb)] bg-[var(--color-class-eb)]/10 border border-[var(--color-class-eb)]/25 px-2 py-0.5 rounded-full">
                      <AlertTriangle size={11} /> FP risk
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex-1 max-w-xs">
                    <ConfidenceBar value={selected.final_confidence} height={10} />
                  </div>
                  <span className="text-[22px] font-bold font-mono-nums text-[var(--color-text-primary)]">
                    {(selected.final_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="grid grid-cols-5 gap-2">
                  <StatTile label="Period" value={f.period_days.toFixed(4)} unit="d" />
                  <StatTile label="Duration" value={f.duration_hours.toFixed(2)} unit="h" />
                  <StatTile label="Depth" value={f.depth_ppm.toFixed(0)} unit="ppm" />
                  <StatTile label="SNR" value={f.snr.toFixed(1)} />
                  <StatTile label="Rₚ est." value={f.planet_radius_re_estimate.toFixed(2)} unit="R⊕" />
                </div>
              </div>
              {/* Vetting actions */}
              <div className="shrink-0">
                <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-2">Vetting status</div>
                <div className="flex flex-col gap-1.5">
                  {STATUS_ACTIONS.map(({ status, label, icon: Icon, color }) => (
                    <button key={status}
                      onClick={() => handleStatusUpdate(selected.candidate_id, status)}
                      disabled={statusUpdating}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11.5px] font-medium border transition-all disabled:opacity-50"
                      style={currentStatus === status ? {
                        color, background: `color-mix(in srgb,${color} 12%,transparent)`,
                        borderColor: `color-mix(in srgb,${color} 40%,transparent)`,
                      } : {
                        color: 'var(--color-text-secondary)',
                        background: 'transparent',
                        borderColor: 'var(--color-border-subtle)',
                      }}>
                      <Icon size={13} />
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {selected.data_quality_warning && (
              <div className="mt-3 flex items-start gap-2 px-3 py-2.5 rounded-lg text-[11.5px] text-[var(--color-class-starspot)] bg-[var(--color-class-starspot)]/8 border border-[var(--color-class-starspot)]/20">
                <AlertTriangle size={13} className="mt-0.5 shrink-0" /> {selected.data_quality_warning}
              </div>
            )}
          </Panel>

          {/* Phase-fold views */}
          <Panel title="Phase-Folded Light Curve" icon={<Orbit size={13} className="text-[var(--color-accent)]"/>}>
            <div className="grid grid-cols-2 gap-3 mb-2">
              <PhaseFoldChart
                values={selected.phase_folded_global} saliency={selected.global_saliency}
                title="Global View" subtitle="Full orbital phase [-0.5, 0.5)"
                phaseRange={[-0.5, 0.5]} height={200}
              />
              <PhaseFoldChart
                values={selected.phase_folded_local} saliency={selected.local_saliency}
                title="Local View" subtitle="Zoomed on transit egress"
                phaseRange={[-0.5, 0.5]} height={200}
              />
            </div>
            <div className="flex items-center gap-3 text-[10px] text-[var(--color-text-disabled)]">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[var(--color-data-cyan)] inline-block"/>Out-of-transit</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[var(--color-accent)] inline-block"/>High AI saliency (drives classification)</span>
            </div>
          </Panel>

          {/* Probability breakdown */}
          <Panel title="Classification Probabilities" icon={<Gauge size={13} className="text-[var(--color-accent)]"/>}>
            <div className="space-y-2">
              {Object.entries(selected.rule_adjusted_probabilities)
                .sort(([, a], [, b]) => (b as number) - (a as number))
                .map(([cls, prob]) => {
                  const p = prob as number;
                  const cnnP = selected.cnn_probabilities[cls] as number;
                  return (
                    <div key={cls} className="flex items-center gap-3">
                      <div className="w-36 text-[11px] text-[var(--color-text-secondary)] truncate">{cls.replace(/_/g,' ')}</div>
                      <div className="flex-1 flex items-center gap-2">
                        <div className="flex-1 h-4 rounded bg-[var(--color-bg-inset)] relative overflow-hidden">
                          <div className="h-full rounded transition-all duration-500"
                            style={{ width: `${Math.max(2, p*100)}%`, background: 'var(--color-accent)', opacity: 0.7 }} />
                          <div className="absolute top-0 left-0 h-full rounded border-r-2"
                            style={{ width: `${Math.max(2, cnnP*100)}%`, borderColor: 'var(--color-data-cyan)' }}
                            title={`CNN raw: ${(cnnP*100).toFixed(1)}%`} />
                        </div>
                        <span className="font-mono-nums text-[11px] text-[var(--color-text-primary)] w-10 text-right">
                          {(p*100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
            <p className="text-[10px] text-[var(--color-text-disabled)] mt-2">
              Amber bar = ensemble-adjusted confidence · Cyan line = raw CNN output before vetting rules
            </p>
          </Panel>

          {/* Scientific explanation */}
          <Panel title="Scientific Interpretation" icon={<Globe2 size={13} className="text-[var(--color-accent)]"/>}>
            <p className="text-[12.5px] text-[var(--color-text-secondary)] leading-relaxed mb-3">{selected.explanation}</p>
            <div className="px-3 py-2.5 rounded-lg bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] text-[11.5px] text-[var(--color-text-tertiary)] leading-relaxed">
              {CLASS_DESCRIPTIONS[selected.final_label]}
            </div>
          </Panel>

          {/* Evidence */}
          <Panel title="Vetting Evidence" icon={<Star size={13} className="text-[var(--color-accent)]"/>}>
            <div className="space-y-2">
              {selected.evidence.map((e, i) => (
                <EvidenceTag key={i} direction={e.direction}>{e.reason}</EvidenceTag>
              ))}
            </div>
            {selected.false_positive_flags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {selected.false_positive_flags.map((flag) => (
                  <span key={flag}
                    className="text-[10px] px-2 py-0.5 rounded-full border font-medium"
                    style={{ color: 'var(--color-class-eb)', borderColor: 'color-mix(in srgb,var(--color-class-eb) 35%,transparent)', background: 'color-mix(in srgb,var(--color-class-eb) 10%,transparent)' }}>
                    {flag.replace(/_/g,' ')}
                  </span>
                ))}
              </div>
            )}
          </Panel>

          {/* Full metrics table */}
          <Panel title="Signal Metrics" icon={<Ruler size={13} className="text-[var(--color-accent)]"/>}>
            <div className="grid grid-cols-3 gap-2">
              {[
                ['Orbital period', `${f.period_days.toFixed(6)} d`],
                ['Transit duration', `${f.duration_hours.toFixed(4)} h`],
                ['Transit depth', `${f.depth_ppm.toFixed(1)} ppm`],
                ['Depth uncertainty', `±${f.depth_err_ppm.toFixed(1)} ppm`],
                ['Detection SNR', f.snr.toFixed(2)],
                ['BLS power', f.bls_power.toFixed(2)],
                ['Transits observed', `${f.n_transits_observed}`],
                ['Epoch (d)', f.epoch_days.toFixed(4)],
                ['Planet radius (est.)', `${f.planet_radius_re_estimate.toFixed(3)} R⊕`],
                ['Transit shape score', `${f.transit_shape_score.toFixed(3)} (0=V, 1=flat)`],
                ['Odd/even σ', `${f.odd_even_depth_diff_sigma.toFixed(2)} σ`],
                ['Secondary eclipse σ', `${f.secondary_eclipse_sig_sigma.toFixed(2)} σ`],
                ['Dur/period ratio', f.duration_period_ratio.toFixed(5)],
                ['In/out scatter', f.in_transit_std_ratio.toFixed(3)],
                ['Periodicity strength', f.periodicity_strength.toFixed(2)],
              ].map(([label, val]) => (
                <div key={label} className="bg-[var(--color-bg-inset)] rounded-lg px-3 py-2.5 border border-[var(--color-border-subtle)]">
                  <div className="text-[9.5px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-0.5">{label}</div>
                  <div className="font-mono-nums text-[12px] text-[var(--color-text-primary)] font-semibold">{val}</div>
                </div>
              ))}
            </div>
          </Panel>
          <Panel
    title="Model Information"
    icon={<Database size={13} className="text-[var(--color-accent)]" />}
>

    {modelInfo ? (

        <div className="grid grid-cols-4 gap-3">

            <StatTile
                label="Run ID"
                value={modelInfo.run_id}
            />

            <StatTile
                label="Accuracy"
                value={`${(modelInfo.test_accuracy * 100).toFixed(1)}%`}
            />

            <StatTile
                label="Macro F1"
                value={modelInfo.macro_f1.toFixed(3)}
            />
              
              <StatTile
                label="Calibration"
              value={modelInfo.calibration_temperature.toFixed(2)}
            />
            <StatTile
                label="Training"
                value={modelInfo.n_train.toLocaleString()}
            />

            <StatTile
                label="Validation"
                value={modelInfo.n_val.toLocaleString()}
            />

            <StatTile
                label="Test"
                value={modelInfo.n_test.toLocaleString()}
            />
              <StatTile
                 label="Classes"
                 value={modelInfo.class_names.length.toString()}
            />
        </div>

    ) : (

        <div className="text-sm text-[var(--color-text-tertiary)]">
            Model information unavailable.
        </div>

    )}

</Panel>
        </div>
      </div>
    </div>
  );
}
