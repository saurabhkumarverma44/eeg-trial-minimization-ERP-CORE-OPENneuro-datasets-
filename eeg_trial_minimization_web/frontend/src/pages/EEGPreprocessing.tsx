// frontend/src/pages/EEGPreprocessing.tsx
import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Activity,
  Layers,
  Info,
} from 'lucide-react';
import { previewPreprocessing, fetchDatasets, DatasetItem } from '../api/client';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export const EEGPreprocessing: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('DS002680_gonogo_4');
  const [channelToPlot, setChannelToPlot] = useState<string>('Fz');

  // Filter params
  const [lFreq, setLFreq] = useState<number>(0.1);
  const [hFreq, setHFreq] = useState<number>(40.0);
  const [notchFreq, setNotchFreq] = useState<number | null>(50.0);
  const [rereference, setRereference] = useState<string>('average');
  const [tmin, setTmin] = useState<number>(-0.2);
  const [tmax, setTmax] = useState<number>(0.8);
  const [baselineStart, setBaselineStart] = useState<number>(-0.2);
  const [baselineEnd, setBaselineEnd] = useState<number>(0.0);
  const [rejectUv, setRejectUv] = useState<number>(100.0);

  // Results
  const [loading, setLoading] = useState<boolean>(false);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [retainedTrials, setRetainedTrials] = useState<number>(1);
  const [rejectedTrials, setRejectedTrials] = useState<number>(0);

  useEffect(() => {
    fetchDatasets()
      .then((data) => setDatasets(data))
      .catch((err) => console.log('Datasets error:', err));
  }, []);

  const handleRunPreview = async () => {
    setLoading(true);
    setStatusMessage('Computing digital bandpass filtering and baseline correction...');
    try {
      const res = await previewPreprocessing({
        dataset_id: selectedDatasetId,
        channel_to_plot: channelToPlot,
        config: {
          l_freq: lFreq,
          h_freq: hFreq,
          notch_freq: notchFreq,
          rereference: rereference,
          tmin: tmin,
          tmax: tmax,
          baseline: [baselineStart, baselineEnd],
          reject_uv: rejectUv,
        },
      });

      // Downsample for smooth charting (e.g., take every 2nd or 3rd sample out of 1001 points)
      const chartPoints = [];
      const step = 2;
      for (let i = 0; i < res.time_points.length; i += step) {
        chartPoints.push({
          timeMs: Math.round(res.time_points[i] * 1000),
          raw: res.raw_signal[i],
          filtered: res.filtered_signal[i],
        });
      }

      setPreviewData(chartPoints);
      setRetainedTrials(res.retained_epochs);
      setRejectedTrials(res.rejected_epochs);
      setStatusMessage(res.message);
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleRunPreview();
  }, [selectedDatasetId, channelToPlot]);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <Sliders className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">EEG Signal Preprocessing Pipeline</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Configure real zero-phase digital filtering, notch filters, re-referencing, epoch baseline subtraction, and artifact rejection thresholds.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Pipeline Parameters
          </h3>

          <div className="space-y-4 text-xs">
            {/* Target Dataset */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Target Dataset</label>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
              >
                {datasets.map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.dataset_id} ({d.file_type})
                  </option>
                ))}
              </select>
            </div>

            {/* Target Channel */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Preview Channel</label>
              <select
                value={channelToPlot}
                onChange={(e) => setChannelToPlot(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono"
              >
                {["Fp1", "Fz", "F3", "F4", "Cz", "C3", "C4", "Pz", "P3", "P4", "O1", "Oz", "O2"].map((ch) => (
                  <option key={ch} value={ch}>
                    Electrode {ch}
                  </option>
                ))}
              </select>
            </div>

            {/* Bandpass range */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Low Cutoff (Hz)</label>
                <input
                  type="number"
                  step="0.1"
                  value={lFreq}
                  onChange={(e) => setLFreq(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                />
              </div>
              <div>
                <label className="text-slate-400 font-semibold block mb-1">High Cutoff (Hz)</label>
                <input
                  type="number"
                  step="1"
                  value={hFreq}
                  onChange={(e) => setHFreq(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                />
              </div>
            </div>

            {/* Notch Filter */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Mains Notch Filter</label>
              <select
                value={notchFreq || 'none'}
                onChange={(e) => setNotchFreq(e.target.value === 'none' ? null : Number(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
              >
                <option value="50">50 Hz (Europe / Asia / Default)</option>
                <option value="60">60 Hz (North America)</option>
                <option value="none">Disabled</option>
              </select>
            </div>

            {/* Re-referencing */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Re-referencing Montage</label>
              <select
                value={rereference}
                onChange={(e) => setRereference(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
              >
                <option value="average">Common Average Reference (CAR)</option>
                <option value="linked_mastoids">Linked Mastoids (TP9/TP10)</option>
                <option value="cz">Cz Reference</option>
              </select>
            </div>

            {/* Baseline window */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Baseline Start (s)</label>
                <input
                  type="number"
                  step="0.05"
                  value={baselineStart}
                  onChange={(e) => setBaselineStart(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                />
              </div>
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Baseline End (s)</label>
                <input
                  type="number"
                  step="0.05"
                  value={baselineEnd}
                  onChange={(e) => setBaselineEnd(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                />
              </div>
            </div>

            {/* Amplitude Rejection */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Peak-to-Peak Rejection (uV)</label>
              <input
                type="number"
                value={rejectUv}
                onChange={(e) => setRejectUv(Number(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">Epochs exceeding this amplitude will be flagged.</span>
            </div>
          </div>

          <button
            onClick={handleRunPreview}
            disabled={loading}
            className="w-full py-2.5 px-4 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 transition shadow-lg shadow-cyan-500/20"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{loading ? 'Processing...' : 'Apply & Preview Filter'}</span>
          </button>
        </div>

        {/* Visualizer Column */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <span>Before vs After Waveform Overlay ({channelToPlot})</span>
                </h3>
                <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                  Filtered: {lFreq} - {hFreq} Hz {notchFreq ? `+ Notch ${notchFreq}Hz` : ''} | Baseline [{baselineStart * 1000}ms, {baselineEnd * 1000}ms]
                </p>
              </div>

              <div className="flex items-center space-x-2 text-xs">
                <span className="px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-[10px] font-semibold">
                  Retained: {retainedTrials}
                </span>
                <span className="px-2 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-500/30 text-[10px] font-semibold">
                  Rejected: {rejectedTrials}
                </span>
              </div>
            </div>

            {/* Recharts Chart */}
            <div className="w-full h-80 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={previewData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="timeMs"
                    stroke="#64748b"
                    fontSize={11}
                    unit="ms"
                    tickFormatter={(val) => `${val}`}
                  />
                  <YAxis stroke="#64748b" fontSize={11} unit="uV" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                    labelFormatter={(val) => `Time: ${val} ms`}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line
                    type="monotone"
                    dataKey="raw"
                    stroke="#94a3b8"
                    strokeWidth={1.2}
                    dot={false}
                    name="Raw Signal (Unfiltered)"
                  />
                  <Line
                    type="monotone"
                    dataKey="filtered"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    dot={false}
                    name="Processed Signal (Filtered + Baseline)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Status note */}
            {statusMessage && (
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-300 flex items-center space-x-2">
                <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                <span>{statusMessage}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
