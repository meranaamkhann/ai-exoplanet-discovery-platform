import { useEffect, useState } from 'react';
import { ListChecks, Check, X, Eye, Filter } from 'lucide-react';
import api from '../lib/api';
import { Panel, ClassBadge, ConfidenceBar } from '../components/ui';

const STATUS_COLORS: Record<string, string> = {
  pending: 'var(--color-text-tertiary)',
  confirmed: 'var(--color-class-planet)',
  rejected: 'var(--color-class-eb)',
  needs_review: 'var(--color-class-starspot)',
};

export default function CandidatesPage() {
  const [candidates, setCandidates] = useState<any[]>([]);
  const [statusFilter, setStatusFilter] = useState('');
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    api.candidates(statusFilter || undefined).then((c) => { setCandidates(c); setLoading(false); });
  };
  useEffect(() => { load(); }, [statusFilter]);

  const updateStatus = async (id: string, status: string) => {
    await api.updateCandidateStatus(id, status);
    setCandidates((prev) => prev.map((c) => c.candidate_id === id ? { ...c, status } : c));
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="h-16 flex items-center justify-between px-6 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)]/60 backdrop-blur sticky top-0 z-10">
        <h1 className="font-[var(--font-display)] text-[18px] font-semibold">Candidate Management</h1>
        <div className="flex items-center gap-2">
          <Filter size={13} className="text-[var(--color-text-tertiary)]" />
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-[var(--color-bg-inset)] border border-[var(--color-border-default)] rounded-md text-[12px] px-2 py-1.5 text-[var(--color-text-secondary)]">
            <option value="">All statuses</option>
            {['pending','confirmed','rejected','needs_review'].map((s) => <option key={s} value={s}>{s.replace('_',' ')}</option>)}
          </select>
        </div>
      </div>
      <div className="p-6">
        <Panel title={`${candidates.length} Candidates`} icon={<ListChecks size={14} className="text-[var(--color-accent)]" />}>
          {loading ? <p className="text-[13px] text-[var(--color-text-tertiary)] py-4 text-center">Loading...</p>
          : candidates.length === 0 ? <p className="text-[13px] text-[var(--color-text-tertiary)] py-4 text-center">No candidates found.</p>
          : (
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)] border-b border-[var(--color-border-subtle)]">
                  {['Classification','Confidence','Period','Depth','SNR','Status','Actions'].map(h => <th key={h} className="py-2 font-medium">{h}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border-subtle)]">
                {candidates.map((c) => {
                  const f = c.payload.features;
                  return (
                    <tr key={c.candidate_id}>
                      <td className="py-2.5"><ClassBadge label={c.payload.final_label} /></td>
                      <td className="py-2.5 w-32">
                        <div className="flex items-center gap-2">
                          <div className="w-14"><ConfidenceBar value={c.payload.final_confidence} /></div>
                          <span className="font-mono-nums">{(c.payload.final_confidence*100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="py-2.5 font-mono-nums text-[var(--color-text-secondary)]">{f.period_days.toFixed(3)}d</td>
                      <td className="py-2.5 font-mono-nums text-[var(--color-text-secondary)]">{f.depth_ppm.toFixed(0)}ppm</td>
                      <td className="py-2.5 font-mono-nums text-[var(--color-text-secondary)]">{f.snr.toFixed(1)}</td>
                      <td className="py-2.5">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium border"
                          style={{ color: STATUS_COLORS[c.status], borderColor: `color-mix(in srgb,${STATUS_COLORS[c.status]} 35%,transparent)`, background: `color-mix(in srgb,${STATUS_COLORS[c.status]} 10%,transparent)` }}>
                          {c.status.replace('_',' ')}
                        </span>
                      </td>
                      <td className="py-2.5">
                        <div className="flex items-center justify-end gap-1">
                          <button onClick={() => updateStatus(c.candidate_id,'confirmed')} title="Confirm" className="p-1.5 rounded hover:bg-[var(--color-class-planet)]/15 text-[var(--color-text-tertiary)] hover:text-[var(--color-class-planet)] transition-colors"><Check size={13}/></button>
                          <button onClick={() => updateStatus(c.candidate_id,'needs_review')} title="Needs review" className="p-1.5 rounded hover:bg-[var(--color-class-starspot)]/15 text-[var(--color-text-tertiary)] hover:text-[var(--color-class-starspot)] transition-colors"><Eye size={13}/></button>
                          <button onClick={() => updateStatus(c.candidate_id,'rejected')} title="Reject" className="p-1.5 rounded hover:bg-[var(--color-class-eb)]/15 text-[var(--color-text-tertiary)] hover:text-[var(--color-class-eb)] transition-colors"><X size={13}/></button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
    </div>
  );
}
