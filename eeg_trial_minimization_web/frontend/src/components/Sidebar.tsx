// frontend/src/components/Sidebar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Cpu,
  Database,
  FileSpreadsheet,
  GitCompare,
  HardDrive,
  Layers,
  PlaySquare,
  Settings,
  ShieldAlert,
  Sparkles,
  Sliders
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: BarChart3 },
  { to: '/live', label: 'Live EEG Stream', icon: Activity },
  { to: '/datasets', label: 'Dataset Explorer', icon: Database },
  { to: '/preprocessing', label: 'EEG Preprocessing', icon: Sliders },
  { to: '/models', label: 'Model Management', icon: BrainCircuit },
  { to: '/inference', label: 'Model Inference', icon: PlaySquare },
  { to: '/minimisation', label: 'Trial Minimisation', icon: Layers },
  { to: '/gan', label: 'GAN Augmentation', icon: Sparkles },
  { to: '/comparison', label: 'Model Comparison', icon: GitCompare },
  { to: '/results', label: 'Results & Reports', icon: FileSpreadsheet },
  { to: '/hardware', label: 'Hardware Connections', icon: Cpu },
  { to: '/settings', label: 'Settings', icon: Settings },
  { to: '/diagnostics', label: 'Diagnostics', icon: ShieldAlert },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-slate-900/90 backdrop-blur border-r border-slate-800 flex flex-col h-screen fixed left-0 top-0 z-30">
      {/* Brand Header */}
      <div className="p-4 border-b border-slate-800 flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-600 to-sky-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <BrainCircuit className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-bold text-sm tracking-wide text-white">ERP Minimisation</h1>
          <p className="text-[10px] text-cyan-400 font-medium">Research Platform v1.0</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center px-3 py-2 text-xs font-medium rounded-lg transition-all duration-150 ${
                isActive
                  ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`
            }
          >
            <Icon className="w-4 h-4 mr-3 flex-shrink-0" />
            <span className="truncate">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/40 text-[11px] text-slate-500 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <HardDrive className="w-3.5 h-3.5 text-slate-400" />
          <span>Drive D: (1.81 TB)</span>
        </div>
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
      </div>
    </aside>
  );
};
