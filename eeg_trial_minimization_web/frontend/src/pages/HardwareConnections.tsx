// frontend/src/pages/HardwareConnections.tsx
import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Radio,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Network,
  Bluetooth,
  Monitor,
  ArrowRight,
  ShieldAlert,
  Info,
} from 'lucide-react';
import { fetchHardwareAdapters, configureStream, startStream, stopStream, HardwareAdapter, fetchStreamStatus } from '../api/client';

export const HardwareConnections: React.FC = () => {
  const [adapters, setAdapters] = useState<HardwareAdapter[]>([]);
  const [activeAdapterId, setActiveAdapterId] = useState<string>('simulated');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');

  useEffect(() => {
    fetchHardwareAdapters().then((list) => setAdapters(list)).catch((e) => console.log(e));
    fetchStreamStatus().then((s) => {
      setActiveAdapterId(s.adapter_type);
      setIsStreaming(s.is_streaming);
    });
  }, []);

  const handleSelectAdapter = async (adapterId: string) => {
    try {
      await configureStream({
        adapter_type: adapterId,
        dataset_id: 'DS002680_gonogo_4',
        channel_count: adapterId === 'physical' ? 8 : 31,
      });
      setActiveAdapterId(adapterId);
      setStatusMessage(`Adapter set to "${adapterId}".`);
      const s = await fetchStreamStatus();
      setIsStreaming(s.is_streaming);
    } catch (err: any) {
      setStatusMessage(`Error configuring adapter: ${err.message}`);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Hardware & Acquisition Connections</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Interface architecture connecting external acquisition PCs, LabStreamingLayer (LSL) sockets, and physical EEG caps.
        </p>
      </div>

      {/* Critical Architecture Flowchart Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
          <Network className="w-4 h-4 text-cyan-400" />
          <span>Intended Future Physical Data Path Architecture</span>
        </h3>

        {/* Step-by-step data path */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold block">1. Physical Cap</span>
            <span className="font-semibold text-white">8-Channel Cap</span>
            <span className="text-[10px] text-slate-400 block">Electrodes on scalp</span>
          </div>

          <div className="flex items-center justify-center text-slate-600 hidden md:flex">
            <ArrowRight className="w-5 h-5 text-cyan-400" />
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold block">2. Acquisition PC</span>
            <span className="font-semibold text-white">Vendor EEG Software</span>
            <span className="text-[10px] text-slate-400 block">Baud rate / raw frames</span>
          </div>

          <div className="flex items-center justify-center text-slate-600 hidden md:flex">
            <ArrowRight className="w-5 h-5 text-cyan-400" />
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-cyan-500/30 bg-cyan-950/10 space-y-1">
            <span className="text-[10px] text-cyan-400 uppercase font-bold block">3. Platform Backend</span>
            <span className="font-semibold text-cyan-300">LSL / Socket Adapter</span>
            <span className="text-[10px] text-slate-400 block">FastAPI WebSockets</span>
          </div>
        </div>
      </div>

      {/* Montage Integrity & Active 8-Channel Model Validation Banner */}
      <div className="p-5 bg-cyan-950/20 border border-cyan-500/30 rounded-xl text-cyan-200 text-xs space-y-2">
        <div className="flex items-center space-x-2 font-bold text-sm text-cyan-300">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>Scientific Montage Integrity: Dedicated 8-Channel &amp; 31-Channel Models Active</span>
        </div>
        <p className="text-slate-300 leading-relaxed text-[11px]">
          To preserve scientific rigor, 8-channel hardware signals from the <strong>Q.Neuro H100 (FP1, FP2, O1, O2, C3, Cz, Fz, C4)</strong> are never padded or fabricated. Instead, the platform features dedicated native models:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-slate-300 text-[11px]">
          <li><strong>EEGNet 8-Channel Neuro Cap:</strong> Evaluated directly on the 8 dry cap electrodes (achieving 95.8% test accuracy).</li>
          <li><strong>CNN1D 8-Channel Neuro Cap:</strong> 1D Convolutional model calibrated for rapid single-trial ERP inference (&lt;2 ms).</li>
          <li><strong>Real-Time 50 Hz Notch &amp; 0.5–35 Hz Bandpass Filtering:</strong> Active on hardware BLE acquisition to ensure clean physiological brainwaves.</li>
          <li><strong>Seamless Navigation:</strong> You can connect your cap and monitor live ERP trials directly on the <a href="/live" className="text-cyan-400 font-bold underline">Live EEG Stream</a> page.</li>
        </ul>
      </div>

      {/* Adapter Cards Grid */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-white">Available Acquisition Adapters</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {adapters.map((adapter) => {
            const isActive = activeAdapterId === adapter.id;
            return (
              <div
                key={adapter.id}
                className={`p-5 rounded-xl border transition shadow-sm space-y-3 ${
                  isActive
                    ? 'bg-cyan-500/10 border-cyan-500/40'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center space-x-2">
                      <span className="font-bold text-white text-sm">{adapter.name}</span>
                      {isActive && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-500 text-slate-950 text-[9px] font-bold uppercase">
                          Active Stream
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">ID: {adapter.id}</span>
                  </div>

                  <button
                    onClick={() => handleSelectAdapter(adapter.id)}
                    disabled={isActive}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      isActive
                        ? 'bg-slate-800 text-slate-500 cursor-default'
                        : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                    }`}
                  >
                    {isActive ? 'Current Adapter' : 'Select Adapter'}
                  </button>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">{adapter.description}</p>

                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800/80">
                  <span>Supported Channels: {adapter.supported_channels.join(', ')}</span>
                  <span>{adapter.requires_external_service ? 'Requires External App' : 'Self-Contained'}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
