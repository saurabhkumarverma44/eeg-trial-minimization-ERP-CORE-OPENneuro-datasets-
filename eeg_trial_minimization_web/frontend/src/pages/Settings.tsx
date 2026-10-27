// frontend/src/pages/Settings.tsx
import React, { useEffect, useState } from 'react';
import {
  Settings as SettingsIcon,
  HardDrive,
  Folder,
  ShieldCheck,
  Server,
  Save,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { fetchSystemInfo, SystemInfo } from '../api/client';

export const Settings: React.FC = () => {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
  const [savedMessage, setSavedMessage] = useState<string>('');

  useEffect(() => {
    fetchSystemInfo()
      .then((data) => setSysInfo(data))
      .catch((err) => console.log('System info error:', err));
  }, []);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <SettingsIcon className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Application Configuration & Environment</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Inspect configured storage paths, database endpoints, security upload thresholds, and host networking parameters.
        </p>
      </div>

      {savedMessage && (
        <div className="p-3 bg-emerald-950/20 border border-emerald-500/30 rounded-lg text-emerald-300 text-xs flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{savedMessage}</span>
        </div>
      )}

      {/* Directory Paths Section */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
          <Folder className="w-4 h-4 text-cyan-400" />
          <span>Configured Storage Directories</span>
        </h3>

        <div className="space-y-3 text-xs">
          <div className="space-y-1">
            <span className="font-semibold text-slate-300">EEG_PROJECT_ROOT (Original Research Assets)</span>
            <input
              type="text"
              readOnly
              value={sysInfo?.paths?.project_root_path || 'D:\\erp_trial_min_backup_20260626\\erp_trial_min'}
              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-300 font-mono text-[11px]"
            />
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-slate-300">EEG_WEB_ROOT (Website Development Root)</span>
            <input
              type="text"
              readOnly
              value={sysInfo?.paths?.web_root_path || 'D:\\erp_trial_min_backup_20260626\\erp_trial_min\\eeg_trial_minimization_web'}
              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-300 font-mono text-[11px]"
            />
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-slate-300">EEG_RESULTS_ROOT (Application-Generated Results)</span>
            <input
              type="text"
              readOnly
              value={sysInfo?.paths?.results_root_path || 'D:\\erp_trial_min_backup_20260626\\erp_trial_min\\eeg_trial_minimization_web\\storage\\results'}
              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-300 font-mono text-[11px]"
            />
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-slate-300">EEG_MODEL_UPLOAD_ROOT (Uploaded Models Directory)</span>
            <input
              type="text"
              readOnly
              value={sysInfo?.paths?.model_upload_root_path || 'D:\\erp_trial_min_backup_20260626\\erp_trial_min\\eeg_trial_minimization_web\\storage\\uploads'}
              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-300 font-mono text-[11px]"
            />
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-slate-300">EEG_DATABASE_PATH (SQLite Persistent DB)</span>
            <input
              type="text"
              readOnly
              value={sysInfo?.paths?.database_path || 'D:\\erp_trial_min_backup_20260626\\erp_trial_min\\eeg_trial_minimization_web\\storage\\database.sqlite'}
              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-300 font-mono text-[11px]"
            />
          </div>
        </div>
      </div>

      {/* Network & Security Settings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4 text-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>Server & Host Network</span>
          </h3>

          <div className="space-y-3">
            <div>
              <span className="text-slate-400 block mb-1">Backend Host</span>
              <input
                type="text"
                readOnly
                value="127.0.0.1 (Localhost)"
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
              />
            </div>

            <div>
              <span className="text-slate-400 block mb-1">Backend Port</span>
              <input
                type="text"
                readOnly
                value="8000"
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
              />
            </div>

            <div>
              <span className="text-slate-400 block mb-1">Frontend Port</span>
              <input
                type="text"
                readOnly
                value="5173"
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
              />
            </div>
          </div>
        </div>

        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4 text-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Security & Integrity Guardrails</span>
          </h3>

          <div className="space-y-2.5">
            <div className="flex items-center justify-between p-2 bg-slate-950 rounded border border-slate-800">
              <span className="text-slate-300">Path Traversal Protection</span>
              <span className="text-emerald-400 font-semibold text-[11px]">Enforced</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-slate-950 rounded border border-slate-800">
              <span className="text-slate-300">Max Upload Limit</span>
              <span className="font-mono text-cyan-400 text-[11px]">250 MB</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-slate-950 rounded border border-slate-800">
              <span className="text-slate-300">Model Weights-Only Loading</span>
              <span className="text-emerald-400 font-semibold text-[11px]">Active</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-slate-950 rounded border border-slate-800">
              <span className="text-slate-300">Research Asset Deletion Protection</span>
              <span className="text-emerald-400 font-semibold text-[11px]">Strict</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
