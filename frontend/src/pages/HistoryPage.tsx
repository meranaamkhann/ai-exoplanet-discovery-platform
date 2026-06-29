import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { History, AlertTriangle, ChevronRight, Search, BarChart3 } from 'lucide-react';
import api from '../lib/api';
import type { HistoryResponse } from '../types/api';
import { Panel, ClassBadge, ConfidenceBar } from '../components/ui';

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    api.history(200, 0).then(setHistory);
  }, []);

  const items = history?.items.filter((h) =>
    search ? h.dataset_name.toLowerCase().includes(search.toLowerCase()) : true
  ) ?? [];

  // Summary stats
  const nPlanet = items.filter((h) => h.top_label === 'PLANET_TRANSIT' && !h.is_likely_false_positive).length;
  const nFP = items.filter((h) => h.is_likely_false_positive).length;
  const avgConf = items.length ? items.reduce((s, h) => s + h.top_confidence, 0) / items.length : 0;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="h-16 flex items-center px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur sticky top-0 z-10">
        <h1 className="font-[var(--font-display)] text-[18px] font-semibold">Detection History</h1>
      </div>

      <div className="p-6 max-w-5xl space-y-4">
        {/* Summary strip */}
        {history && (
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'Total Analyses', value: history.total },
              { label: 'Planet Candidates', value: nPlanet, color: 'var(--color-class-planet)' },
              { label: 'FP Flagged', value: nFP, color: 'var(--color-class-eb)' },
              { label: 'Avg Confidence', value: `${(avgConf * 100).toFixed(1)}%` },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-[var(--color-bg-panel)] rounded-xl border border-[var(--color-border-subtle)] p-3 text-center">
                <div className="font-mono-nums text-[22px] font-bold" style={{ color: color ?? 'var(--color-text-primary)' }}>{value}</div>
                <div className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5 uppercase tracking-wide">{label}</div>
              </div>
            ))}
          </div>
        )}

        <Panel title={`${items.length} Analyses`} icon={<BarChart3 size={13} className="text-[var(--color-accent)]"/>}
          action={
            <div className="flex items-center gap-1.5 bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] rounded-lg px-2 py-1">
              <Search size={12} className="text-[var(--color-text-tertiary)]" />
              <input value={search} onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter by name..."
                className="bg-transparent text-[12px] text-[var(--color-text-primary)] placeholder-[var(--color-text-disabled)] outline-none w-40" />
            </div>
          }>
          {items.length > 0 ? (
            <div className="divide-y divide-[var(--color-border-subtle)] -mx-4 -mb-4">
              {items.map((h, idx) => (
                <Link key={h.analysis_id} to={`/results/${h.analysis_id}`}
                  className="flex items-center gap-4 px-4 py-3 hover:bg-[var(--color-bg-inset)] transition-colors group">
                  <span className="text-[10px] font-mono-nums text-[var(--color-text-disabled)] w-6 text-right">{idx + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-medium text-[var(--color-text-primary)] truncate group-hover:text-[var(--color-accent)] transition-colors">
                      {h.dataset_name}
                    </div>
                    <div className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5 font-mono-nums">
                      {new Date(h.created_at).toLocaleString()} · {h.n_candidates} candidate{h.n_candidates !== 1 ? 's' : ''}
                    </div>
                  </div>
                  <ClassBadge label={h.top_label} />
                  {h.is_likely_false_positive && (
                    <AlertTriangle size={13} className="text-[var(--color-class-eb)] shrink-0" />
                  )}
                  <div className="flex items-center gap-2 w-32 shrink-0">
                    <div className="flex-1"><ConfidenceBar value={h.top_confidence} /></div>
                    <span className="font-mono-nums text-[11px] text-[var(--color-text-secondary)] w-10 text-right">
                      {(h.top_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <ChevronRight size={14} className="text-[var(--color-text-disabled)] shrink-0" />
                </Link>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-tertiary)] gap-2">
              <History size={28} />
              <p className="text-[13px]">No analyses match your filter.</p>
              {!search && <Link to="/analyze" className="text-[12px] text-[var(--color-accent)]">Run your first analysis →</Link>}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
