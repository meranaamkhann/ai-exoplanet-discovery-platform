import { ChevronRight, AlertTriangle, Orbit } from 'lucide-react';
import type { CandidateResult } from '../types/api';
import { ClassBadge, ConfidenceBar } from './ui';

interface Props {
  candidate: CandidateResult;
  onClick?: () => void;
  selected?: boolean;
}

export default function CandidateCard({ candidate, onClick, selected }: Props) {
  const f = candidate.features;
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border p-3.5 transition-all ${
        selected
          ? 'border-[var(--color-accent)]/50 bg-[var(--color-accent)]/[0.06]'
          : 'border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] hover:border-[var(--color-border-strong)]'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-5 h-5 rounded bg-[var(--color-bg-panel-raised)] text-[10px] font-mono-nums text-[var(--color-text-tertiary)] border border-[var(--color-border-subtle)]">
            {candidate.rank}
          </span>
          <ClassBadge label={candidate.final_label} />
          {candidate.is_likely_false_positive && (
            <span className="flex items-center gap-1 text-[10px] text-[var(--color-class-eb)]">
              <AlertTriangle size={11} /> FP risk
            </span>
          )}
        </div>
        <ChevronRight size={16} className="text-[var(--color-text-tertiary)] shrink-0" />
      </div>

      <div className="flex items-center gap-3 mb-2">
        <div className="flex-1">
          <ConfidenceBar value={candidate.final_confidence} />
        </div>
        <span className="font-mono-nums text-[12px] font-semibold text-[var(--color-text-primary)] w-12 text-right">
          {(candidate.final_confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div className="flex items-center gap-1 text-[var(--color-text-tertiary)]">
          <Orbit size={11} />
          <span className="font-mono-nums text-[var(--color-text-secondary)]">{f.period_days.toFixed(3)}d</span>
        </div>
        <div className="text-[var(--color-text-tertiary)]">
          depth <span className="font-mono-nums text-[var(--color-text-secondary)]">{f.depth_ppm.toFixed(0)}ppm</span>
        </div>
        <div className="text-[var(--color-text-tertiary)]">
          SNR <span className="font-mono-nums text-[var(--color-text-secondary)]">{f.snr.toFixed(1)}</span>
        </div>
      </div>
    </button>
  );
}
