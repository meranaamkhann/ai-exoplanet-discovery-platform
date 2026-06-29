import { useMemo, useState, useCallback } from 'react';
import {
  ComposedChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, Line,
} from 'recharts';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import type { TimeSeriesPoint } from '../types/api';

interface TransitMarker { epoch: number; period: number; }

interface Props {
  raw: TimeSeriesPoint[];
  cleaned: TimeSeriesPoint[];
  trend: TimeSeriesPoint[];
  height?: number;
  transitMarkers?: TransitMarker[];
}

const MAX_RENDER_POINTS = 4000;

function downsamplePoints(pts: TimeSeriesPoint[], maxPts: number): TimeSeriesPoint[] {
  if (pts.length <= maxPts) return pts;
  const step = pts.length / maxPts;
  return Array.from({ length: maxPts }, (_, i) => pts[Math.floor(i * step)]);
}

export default function LightCurveViewer({ raw, cleaned, trend, height = 340, transitMarkers }: Props) {
  const [view, setView] = useState<'cleaned' | 'raw'>('cleaned');
  const [domain, setDomain] = useState<[number, number] | null>(null);
  const [selecting, setSelecting] = useState<{ x1: number; x2: number | null }>({ x1: 0, x2: null });
  const [isSelecting, setIsSelecting] = useState(false);

  const allData = view === 'cleaned' ? cleaned : raw;

  const fullDomain = useMemo<[number, number]>(() => {
    if (!allData.length) return [0, 1];
    return [allData[0].t, allData[allData.length - 1].t];
  }, [allData]);

  const activeDomain = domain ?? fullDomain;

  const visibleData = useMemo(() => {
    const filtered = allData.filter(p => p.t >= activeDomain[0] && p.t <= activeDomain[1]);
    return downsamplePoints(filtered, MAX_RENDER_POINTS);
  }, [allData, activeDomain]);

  const trendVisible = useMemo(() => {
    if (view !== 'raw') return [];
    const filtered = trend.filter(p => p.t >= activeDomain[0] && p.t <= activeDomain[1]);
    return downsamplePoints(filtered, 800);
  }, [trend, activeDomain, view]);

  const markerLines = useMemo(() => {
    if (!transitMarkers?.length) return [];
    const lines: number[] = [];
    for (const m of transitMarkers) {
      let t = m.epoch % m.period;
      while (t < activeDomain[0]) t += m.period;
      while (t <= activeDomain[1]) { lines.push(t); t += m.period; }
    }
    return lines;
  }, [transitMarkers, activeDomain]);

  const handleMouseDown = useCallback((e: any) => {
    if (e && e.activeLabel != null) {
      setIsSelecting(true);
      setSelecting({ x1: e.activeLabel, x2: null });
    }
  }, []);

  const handleMouseMove = useCallback((e: any) => {
    if (isSelecting && e && e.activeLabel != null) {
      setSelecting(s => ({ ...s, x2: e.activeLabel }));
    }
  }, [isSelecting]);

  const handleMouseUp = useCallback(() => {
    if (isSelecting && selecting.x2 != null) {
      const range = Math.abs(selecting.x2 - selecting.x1);
      const totalRange = fullDomain[1] - fullDomain[0];
      if (range > totalRange * 0.005) {
        const x1 = Math.min(selecting.x1, selecting.x2!);
        const x2 = Math.max(selecting.x1, selecting.x2!);
        setDomain([x1, x2]);
      }
    }
    setIsSelecting(false);
    setSelecting({ x1: 0, x2: null });
  }, [isSelecting, selecting, fullDomain]);

  const zoomOut = useCallback(() => {
    if (!domain) return;
    const [d0, d1] = domain;
    const center = (d0 + d1) / 2;
    const halfWidth = (d1 - d0) * 1.5;
    const newD0 = Math.max(fullDomain[0], center - halfWidth);
    const newD1 = Math.min(fullDomain[1], center + halfWidth);
    if (Math.abs(newD0 - fullDomain[0]) < 0.01 && Math.abs(newD1 - fullDomain[1]) < 0.01) {
      setDomain(null);
    } else {
      setDomain([newD0, newD1]);
    }
  }, [domain, fullDomain]);

  const zoomPercent = domain
    ? Math.round((1 - (activeDomain[1] - activeDomain[0]) / (fullDomain[1] - fullDomain[0])) * 100)
    : 0;

  return (
    <div className="bg-[var(--color-bg-inset)] rounded-xl border border-[var(--color-border-subtle)] overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--color-border-subtle)]">
        <div className="flex gap-0.5 bg-[var(--color-bg-panel)] rounded-md p-0.5 border border-[var(--color-border-subtle)]">
          {(['cleaned', 'raw'] as const).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                view === v ? 'bg-[var(--color-accent)]/15 text-[var(--color-accent)]' : 'text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]'
              }`}>
              {v === 'cleaned' ? 'Detrended' : 'Raw + Trend'}
            </button>
          ))}
        </div>

        <div className="flex-1 flex items-center gap-1.5">
          {domain && (
            <span className="text-[10px] font-mono-nums text-[var(--color-accent)] bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20 rounded px-1.5 py-0.5">
              {zoomPercent}% zoom
            </span>
          )}
          <span className="text-[10px] text-[var(--color-text-disabled)] font-mono-nums">
            {activeDomain[0].toFixed(1)}–{activeDomain[1].toFixed(1)} d
          </span>
        </div>

        <div className="flex items-center gap-1">
          {domain && (
            <button onClick={zoomOut} title="Zoom out"
              className="p-1.5 rounded hover:bg-[var(--color-bg-panel-raised)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors">
              <ZoomOut size={13} />
            </button>
          )}
          {domain && (
            <button onClick={() => setDomain(null)} title="Reset zoom"
              className="p-1.5 rounded hover:bg-[var(--color-bg-panel-raised)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors">
              <Maximize2 size={13} />
            </button>
          )}
          {!domain && (
            <span className="text-[10px] text-[var(--color-text-disabled)] flex items-center gap-1">
              <ZoomIn size={11}/> Drag to zoom
            </span>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="px-1 py-2">
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart
            data={visibleData}
            margin={{ top: 8, right: 16, bottom: 20, left: 8 }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{ cursor: isSelecting ? 'col-resize' : 'crosshair' }}
          >
            <CartesianGrid stroke="var(--color-border-subtle)" strokeDasharray="2 5" vertical={false} />
            <XAxis
              dataKey="t" type="number" domain={activeDomain}
              tick={{ fill: 'var(--color-text-tertiary)', fontSize: 10 }}
              tickFormatter={(v) => `${v.toFixed(1)}d`}
              stroke="var(--color-border-default)"
              label={{ value: 'Time (days)', position: 'insideBottom', offset: -10, fill: 'var(--color-text-tertiary)', fontSize: 10 }}
            />
            <YAxis
              dataKey="f" domain={['auto', 'auto']}
              tick={{ fill: 'var(--color-text-tertiary)', fontSize: 10 }}
              tickFormatter={(v) => v.toFixed(4)}
              stroke="var(--color-border-default)"
              width={64}
              label={{ value: 'Norm. Flux', angle: -90, position: 'insideLeft', fill: 'var(--color-text-tertiary)', fontSize: 10, dy: 40 }}
            />
            <Tooltip
              contentStyle={{ background: 'var(--color-bg-panel-raised)', border: '1px solid var(--color-border-default)', borderRadius: 8, fontSize: 11, padding: '6px 10px' }}
              labelStyle={{ color: 'var(--color-text-secondary)', marginBottom: 2 }}
              formatter={(value: any) => [Number(value).toFixed(6), 'Flux']}
              labelFormatter={(label) => `t = ${Number(label).toFixed(4)} days`}
              cursor={{ stroke: 'var(--color-accent)', strokeWidth: 1, strokeDasharray: '3 3' }}
            />

            {/* Transit prediction markers */}
            {markerLines.map((t, i) => (
              <ReferenceArea key={i} x1={t - 0.015} x2={t + 0.015}
                fill="var(--color-accent)" fillOpacity={0.18} stroke="none" />
            ))}

            {/* Data points */}
            <Scatter dataKey="f" fill="var(--color-data-cyan)" fillOpacity={0.7}
              r={visibleData.length > 1500 ? 1 : 1.8} isAnimationActive={false} />

            {/* Trend overlay (raw mode) */}
            {view === 'raw' && trendVisible.length > 0 && (
              <Line
                data={trendVisible} dataKey="f"
                stroke="var(--color-accent)" strokeWidth={1.5}
                dot={false} isAnimationActive={false} type="monotone"
              />
            )}

            {/* Zoom selection area */}
            {isSelecting && selecting.x2 != null && (
              <ReferenceArea
                x1={Math.min(selecting.x1, selecting.x2)} x2={Math.max(selecting.x1, selecting.x2)}
                fill="var(--color-data-cyan)" fillOpacity={0.08}
                stroke="var(--color-data-cyan)" strokeOpacity={0.4} strokeDasharray="4 2"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-[var(--color-border-subtle)] flex items-center justify-between">
        <span className="text-[10px] text-[var(--color-text-disabled)]">
          {visibleData.length.toLocaleString()} points displayed
          {visibleData.length < allData.filter(p => p.t >= activeDomain[0] && p.t <= activeDomain[1]).length && ' (downsampled)'}
        </span>
        {transitMarkers && transitMarkers.length > 0 && (
          <span className="text-[10px] text-[var(--color-text-disabled)] flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm bg-[var(--color-accent)] opacity-40 inline-block"/>
            Predicted transit epochs
          </span>
        )}
      </div>
    </div>
  );
}
