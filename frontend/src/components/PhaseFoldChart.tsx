import { useMemo } from 'react';
import {
  ComposedChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

interface Props {
  values: number[];
  saliency: number[];
  title: string;
  subtitle?: string;
  height?: number;
  phaseRange: [number, number];
}

export default function PhaseFoldChart({ values, saliency, title, subtitle, height = 220, phaseRange }: Props) {
  const data = useMemo(() => {
    const n = values.length;
    return values.map((v, i) => ({
      phase: phaseRange[0] + (i / Math.max(n - 1, 1)) * (phaseRange[1] - phaseRange[0]),
      flux: 1 + v,
      saliency: saliency[i] ?? 0,
    }));
  }, [values, saliency, phaseRange]);

  return (
    <div className="bg-[var(--color-bg-inset)] rounded-lg border border-[var(--color-border-subtle)] p-3">
      <div className="flex items-baseline justify-between mb-1.5">
        <h4 className="text-[12px] font-semibold text-[var(--color-text-primary)]">{title}</h4>
        {subtitle && <span className="text-[10px] text-[var(--color-text-tertiary)]">{subtitle}</span>}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--color-border-subtle)" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="phase"
            type="number"
            domain={phaseRange}
            tick={{ fill: 'var(--color-text-tertiary)', fontSize: 9 }}
            tickFormatter={(v) => v.toFixed(3)}
            stroke="var(--color-border-default)"
          />
          <YAxis
            dataKey="flux"
            domain={['auto', 'auto']}
            tick={{ fill: 'var(--color-text-tertiary)', fontSize: 9 }}
            tickFormatter={(v) => v.toFixed(4)}
            stroke="var(--color-border-default)"
            width={56}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--color-bg-panel-raised)',
              border: '1px solid var(--color-border-default)',
              borderRadius: 6,
              fontSize: 11,
            }}
            formatter={(value: any, name: any) => [Number(value).toFixed(6), name === 'flux' ? 'Flux' : String(name ?? '')]}
            labelFormatter={(label) => `phase = ${Number(label).toFixed(4)}`}
          />
          <Scatter dataKey="flux" isAnimationActive={false}>
            {data.map((d, i) => {
              const t = d.saliency;
              const color = t > 0.5 ? 'var(--color-accent)' : 'var(--color-data-cyan)';
              return <Cell key={i} fill={color} fillOpacity={0.35 + t * 0.6} />;
            })}
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
