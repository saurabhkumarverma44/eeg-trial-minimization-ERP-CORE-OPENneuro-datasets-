// frontend/src/components/Navbar.tsx
import React, { useEffect, useState } from 'react';
import { Activity, Brain, CheckCircle2, AlertTriangle, Radio } from 'lucide-react';
import { fetchHealth, fetchStreamStatus, startStream, stopStream, HealthData, StreamStatus } from '../api/client';

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [stream, setStream] = useState<StreamStatus | null>(null);
  const [loadingStream, setLoadingStream] = useState(false);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch (err) {
        setHealth(null);
      }
      try {
        const s = await fetchStreamStatus();
        setStream(s);
      } catch (err) {
        setStream(null);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleStream = async () => {
    setLoadingStream(true);
    try {
      if (stream?.is_streaming) {
        await stopStream();
      } else {
        await startStream();
      }
      const s = await fetchStreamStatus();
      setStream(s);
    } catch (err) {
      console.error('Stream toggle error:', err);
    } finally {
      setLoadingStream(false);
    }
  };

  return (
    <header className="h-14 bg-slate-900/80 backdrop-blur border-b border-slate-800 px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Breadcrumb / Title */}
      <div className="flex items-center space-x-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Project Workspace</span>
        <span className="text-slate-600">/</span>
        <span className="text-xs text-cyan-400 font-mono bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
          DS002680_gonogo_4
        </span>
      </div>

      {/* Status Bar */}
      <div className="flex items-center space-x-4">
        {/* Backend Health Pill */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800/80 border border-slate-700">
          {health?.status === 'healthy' ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-emerald-400">Backend Online</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              <span className="text-amber-400">Connecting...</span>
            </>
          )}
        </div>

        {/* Active Model Pill */}
        <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800/80 border border-slate-700">
          <Brain className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-300">Active Model:</span>
          <span className="text-cyan-300 font-mono text-[11px] max-w-[140px] truncate">
            {health?.active_model || 'EEGNet DS002680'}
          </span>
        </div>

        {/* Stream Toggle Pill */}
        <button
          onClick={handleToggleStream}
          disabled={loadingStream}
          className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold transition-all ${
            stream?.is_streaming
              ? 'bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25'
              : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/25'
          }`}
        >
          <Radio className={`w-3.5 h-3.5 ${stream?.is_streaming ? 'animate-pulse text-red-400' : 'text-emerald-400'}`} />
          <span>{stream?.is_streaming ? 'Stop Stream' : 'Start Stream'}</span>
        </button>
      </div>
    </header>
  );
};
