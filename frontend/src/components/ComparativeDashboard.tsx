import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, Cell,
} from 'recharts';
import type { CandidateResult } from '../types/api';
import { CLASS_COLORS } from './ui';

interface Props {
  candidates: CandidateResult[];
  height?: number;
}

/** Scatter plot comparing candidates: period vs depth, sized by SNR, colored by class.
 * Standard astronomy "discovery space" visualization for comparative candidate review. */
export default function ComparativeDashboard({ candidates, height = 320 }: Props) {
  const data = useMemo(
    () =>
      candidates.map((c) => ({
        period: c.features.period_days,
        depth: c.features.depth_ppm,
        snr: c.features.snr,
        label: c.final_label,
        confidence: c.final_confidence,
        rank: c.rank,
      })),
    [candidates]
  );

  return (
    <div className="bg-[var(--color-bg-inset)] rounded-lg border border-[var(--color-border-subtle)] p-3">
      <h4 className="text-[12px] font-semibold text-[var(--color-text-primary)] mb-2">
        Discovery Space — Period vs. Depth
      </h4>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="var(--color-border-subtle)" strokeDasharray="2 4" />
          <XAxis
            dataKey="period"
            type="number"
            scale="log"
            domain={['auto', 'auto']}
            tick={{ fill: 'var(--color-text-tertiary)', fontSize: 10 }}
            stroke="var(--color-border-default)"
            label={{ value: 'Period (days, log scale)', position: 'insideBottom', offset: -4, fill: 'var(--color-text-tertiary)', fontSize: 10 }}
          />
          <YAxis
            dataKey="depth"
            type="number"
            scale="log"
            domain={['auto', 'auto']}
            tick={{ fill: 'var(--color-text-tertiary)', fontSize: 10 }}
            stroke="var(--color-border-default)"
            width={70}
            label={{ value: 'Depth (ppm, log scale)', angle: -90, position: 'insideLeft', fill: 'var(--color-text-tertiary)', fontSize: 10 }}
          />
          <ZAxis dataKey="snr" range={[60, 400]} name="SNR" />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{
              background: 'var(--color-bg-panel-raised)',
              border: '1px solid var(--color-border-default)',
              borderRadius: 6,
              fontSize: 11,
            }}
            formatter={(value: any, name: any) => {
              if (name === 'period') return [`${Number(value).toFixed(3)} d`, 'Period'];
              if (name === 'depth') return [`${Number(value).toFixed(0)} ppm`, 'Depth'];
              if (name === 'snr') return [Number(value).toFixed(1), 'SNR'];
              return [String(value), String(name ?? '')];
            }}
            labelFormatter={() => ''}
          />
          <Scatter data={data} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={CLASS_COLORS[d.label as keyof typeof CLASS_COLORS] ?? '#888'} fillOpacity={0.4 + d.confidence * 0.5} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-[var(--color-text-disabled)] mt-1 text-center">
        Bubble size = detection SNR · Color = classification · Opacity = confidence
      </p>
    </div>
  );
}
