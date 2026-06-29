import { NavLink, Outlet } from 'react-router-dom';
import { Telescope, LayoutDashboard, Upload, History, ListChecks, Database, Activity } from 'lucide-react';
import { useEffect, useState } from 'react';
import api from '../lib/api';
import type { HealthResponse } from '../types/api';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/analyze', label: 'New Analysis', icon: Upload },
  { to: '/candidates', label: 'Candidates', icon: ListChecks },
  { to: '/history', label: 'Detection History', icon: History },
  { to: '/real-data', label: 'Real NASA Targets', icon: Database },
];

export default function AppLayout() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const poll = () => api.health().then(setHealth).catch(() => setHealth(null));
    poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen flex bg-[var(--color-bg-base)] bg-starfield">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-[var(--color-border-subtle)] bg-[var(--color-bg-panel)] flex flex-col">
        <div className="h-16 flex items-center gap-3 px-5 border-b border-[var(--color-border-subtle)]">
          <div className="w-8 h-8 rounded-md bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/30 flex items-center justify-center">
            <Telescope size={18} className="text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="font-[var(--font-display)] font-semibold text-[15px] tracking-tight text-[var(--color-text-primary)]">ExoNova</div>
            <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-tertiary)]">Discovery Platform</div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] font-medium transition-colors ${
                  isActive
                    ? 'bg-[var(--color-accent)]/10 text-[var(--color-accent)] border border-[var(--color-accent)]/20'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-panel-raised)] border border-transparent'
                }`
              }
            >
              <item.icon size={15} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-[var(--color-border-subtle)]">
          <div className="flex items-center gap-2 text-[11px]">
            <Activity size={12} className={health?.model_loaded ? 'text-[var(--color-class-planet)]' : 'text-[var(--color-text-tertiary)]'} />
            <span className="text-[var(--color-text-tertiary)]">
              {health?.model_loaded ? (
                <span className="text-[var(--color-text-secondary)]">Model online</span>
              ) : (
                'Model offline'
              )}
            </span>
          </div>
          {health?.model_run_id && (
            <div className="mt-1 font-mono-nums text-[10px] text-[var(--color-text-disabled)] truncate">
              {health.model_run_id}
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0 flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}
