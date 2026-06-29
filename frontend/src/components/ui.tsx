import type { ReactNode } from 'react';
import type { SignalClass } from '../types/api';

export const CLASS_LABELS: Record<SignalClass, string> = {
  PLANET_TRANSIT: 'Planet Transit',
  ECLIPSING_BINARY: 'Eclipsing Binary',
  STELLAR_VARIABILITY: 'Stellar Variability',
  STARSPOT_ACTIVITY: 'Starspot Activity',
  INSTRUMENT_NOISE: 'Instrument Noise',
};

export const CLASS_COLORS: Record<SignalClass, string> = {
  PLANET_TRANSIT: 'var(--color-class-planet)',
  ECLIPSING_BINARY: 'var(--color-class-eb)',
  STELLAR_VARIABILITY: 'var(--color-class-variability)',
  STARSPOT_ACTIVITY: 'var(--color-class-starspot)',
  INSTRUMENT_NOISE: 'var(--color-class-noise)',
};

export function ClassBadge({ label }: { label: SignalClass | string }) {
  const color = CLASS_COLORS[label as SignalClass] ?? 'var(--color-text-tertiary)';
  const text = CLASS_LABELS[label as SignalClass] ?? label;
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border"
      style={{ color, borderColor: `color-mix(in srgb, ${color} 35%, transparent)`, background: `color-mix(in srgb, ${color} 12%, transparent)` }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {text}
    </span>
  );
}

export function ConfidenceBar({ value, height = 6 }: { value: number; height?: number }) {
  const color = value >= 0.7 ? 'var(--color-conf-high)' : value >= 0.4 ? 'var(--color-conf-medium)' : 'var(--color-conf-low)';
  return (
    <div className="w-full rounded-full bg-[var(--color-bg-inset)] overflow-hidden border border-[var(--color-border-subtle)]" style={{ height }}>
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(2, value * 100))}%`, background: color }}
      />
    </div>
  );
}

export function Panel({ children, className = '', title, icon, action }: {
  children: ReactNode; className?: string; title?: string; icon?: ReactNode; action?: ReactNode;
}) {
  return (
    <div className={`bg-[var(--color-bg-panel)] border border-[var(--color-border-subtle)] rounded-xl ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border-subtle)]">
          <div className="flex items-center gap-2">
            {icon}
            <h3 className="text-[13px] font-semibold text-[var(--color-text-primary)] font-[var(--font-display)]">{title}</h3>
          </div>
          {action}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

export function StatTile({ label, value, unit, mono = true }: { label: string; value: string | number; unit?: string; mono?: boolean }) {
  return (
    <div className="bg-[var(--color-bg-inset)] rounded-lg border border-[var(--color-border-subtle)] px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] mb-1">{label}</div>
      <div className={`text-[15px] font-semibold text-[var(--color-text-primary)] ${mono ? 'font-mono-nums' : ''}`}>
        {value} {unit && <span className="text-[11px] font-normal text-[var(--color-text-tertiary)]">{unit}</span>}
      </div>
    </div>
  );
}

export function EvidenceTag({ direction, children }: { direction: 'supports' | 'against'; children: ReactNode }) {
  const isFor = direction === 'supports';
  return (
    <div
      className="flex items-start gap-2 px-3 py-2 rounded-md text-[12px] leading-relaxed border"
      style={{
        background: isFor ? 'color-mix(in srgb, var(--color-class-planet) 6%, transparent)' : 'color-mix(in srgb, var(--color-class-eb) 6%, transparent)',
        borderColor: isFor ? 'color-mix(in srgb, var(--color-class-planet) 25%, transparent)' : 'color-mix(in srgb, var(--color-class-eb) 25%, transparent)',
        color: 'var(--color-text-secondary)',
      }}
    >
      <span className="mt-0.5 shrink-0 font-bold" style={{ color: isFor ? 'var(--color-class-planet)' : 'var(--color-class-eb)' }}>
        {isFor ? '✓' : '✗'}
      </span>
      <span>{children}</span>
    </div>
  );
}
