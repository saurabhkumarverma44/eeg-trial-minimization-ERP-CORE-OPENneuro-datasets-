// frontend/src/pages/ModelComparison.tsx
import React, { useState } from 'react';
import {
  GitCompare,
  TrendingUp,
  Brain,
  Layers,
  Award,
  Info,
  BarChart3,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

// Verified research comparison data from cnn_vs_eegnet_comparison.csv
const VERIFIED_RESEARCH_DATA = [
  { task: 'gonogo_7', cnnAcc: 89.0, eegnetAcc: 93.1, delta: +4.1, better: 'EEGNet', testSamples: 692 },
  { task: 'flanker', cnnAcc: 57.7, eegnetAcc: 60.8, delta: +3.1, better: 'EEGNet', testSamples: 6775 },
  { task: 'gonogo_9', cnnAcc: 67.2, eegnetAcc: 69.5, delta: +2.4, better: 'EEGNet', testSamples: 676 },
  { task: 'gonogo_4', cnnAcc: 94.2, eegnetAcc: 95.9, delta: +1.7, better: 'EEGNet', testSamples: 688 },
  { task: 'gonogo_8', cnnAcc: 65.3, eegnetAcc: 66.2, delta: +0.9, better: 'EEGNet', testSamples: 692 },
  { task: 'visualsearch', cnnAcc: 64.7, eegnetAcc: 65.4, delta: +0.7, better: 'EEGNet', testSamples: 4874 },
  { task: 'gonogo_2', cnnAcc: 72.0, eegnetAcc: 72.0, delta: 0.0, better: 'Tie', testSamples: 764 },
  { task: 'auditoryoddball', cnnAcc: 78.6, eegnetAcc: 78.6, delta: 0.0, better: 'Tie', testSamples: 9474 },
  { task: 'ERP_MMN', cnnAcc: 80.1, eegnetAcc: 80.0, delta: -0.1, better: 'CNN1D', testSamples: 3698 },
  { task: 'gonogo_11', cnnAcc: 69.6, eegnetAcc: 69.3, delta: -0.3, better: 'CNN1D', testSamples: 672 },
  { task: 'visualoddball', cnnAcc: 76.8, eegnetAcc: 76.4, delta: -0.4, better: 'CNN1D', testSamples: 4746 },
  { task: 'gonogo_5', cnnAcc: 67.4, eegnetAcc: 66.8, delta: -0.6, better: 'CNN1D', testSamples: 632 },
  { task: 'ERP_N170', cnnAcc: 70.1, eegnetAcc: 69.5, delta: -0.7, better: 'CNN1D', testSamples: 2201 },
  { task: 'gonogo_10', cnnAcc: 92.7, eegnetAcc: 91.4, delta: -1.3, better: 'CNN1D', testSamples: 770 },
  { task: 'gonogo_13', cnnAcc: 89.8, eegnetAcc: 88.5, delta: -1.3, better: 'CNN1D', testSamples: 762 },
  { task: 'ERP_P3', cnnAcc: 77.1, eegnetAcc: 75.7, delta: -1.4, better: 'CNN1D', testSamples: 708 },
  { task: 'gonogo_1', cnnAcc: 93.8, eegnetAcc: 92.4, delta: -1.4, better: 'CNN1D', testSamples: 552 },
];

export const ModelComparison: React.FC = () => {
  const [filterDataset, setFilterDataset] = useState<string>('all');

  const filteredData = VERIFIED_RESEARCH_DATA.filter((d) => {
    if (filterDataset === 'gonogo') return d.task.startsWith('gonogo');
    if (filterDataset === 'erp_core') return d.task.startsWith('ERP');
    if (filterDataset === 'ds006018') return !d.task.startsWith('gonogo') && !d.task.startsWith('ERP');
    return true;
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <GitCompare className="w-5 h-5 text-cyan-400" />
            <h1 className="text-2xl font-bold text-white">Model Comparison & Benchmark Evaluation</h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Controlled empirical comparison of EEGNet vs. 1D CNN baseline across all 24 verified experimental paradigms.
          </p>
        </div>

        {/* Filter */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-400">Filter Suite:</span>
          <select
            value={filterDataset}
            onChange={(e) => setFilterDataset(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200"
          >
            <option value="all">All Paradigms ({VERIFIED_RESEARCH_DATA.length})</option>
            <option value="gonogo">DS002680 Go/No-Go Suite</option>
            <option value="erp_core">ERP_CORE Suite</option>
            <option value="ds006018">DS006018 Oddball & Search</option>
          </select>
        </div>
      </div>

      {/* Summary Highlight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold uppercase tracking-wider">Top Performer: Go/No-Go #4</span>
            <Award className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-extrabold text-cyan-400">95.93% vs 94.19%</div>
          <p className="text-[11px] text-slate-400">EEGNet outperforms CNN1D by +1.74% on 688 test trials</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold uppercase tracking-wider">Flanker Paradigm Delta</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400">+3.11% Gain</div>
          <p className="text-[11px] text-slate-400">60.81% (EEGNet) vs 57.70% (CNN1D) on 6,775 test trials</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold uppercase tracking-wider">Statistical Rigor Note</span>
            <Info className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-sm font-semibold text-slate-200">Confidence Interval Bound</div>
          <p className="text-[11px] text-slate-400">Matched fold evaluations; overlapping CIs prevent unconditional superiority claims</p>
        </div>
      </div>

      {/* Accuracy Comparison Bar Chart */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-cyan-400" />
              <span>Side-by-Side Test Accuracy Comparison (EEGNet vs 1D CNN)</span>
            </h3>
            <p className="text-[11px] text-slate-400">
              Extracted directly from verified research results table <code className="text-cyan-300 font-mono">results_dl_1/cnn_vs_eegnet_comparison.csv</code>
            </p>
          </div>
        </div>

        <div className="w-full h-80 pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={filteredData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="task" stroke="#64748b" fontSize={10} angle={-30} textAnchor="end" />
              <YAxis domain={[50, 100]} stroke="#64748b" fontSize={11} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val: any) => [`${val}%`]}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              <Bar dataKey="eegnetAcc" name="EEGNet Accuracy" fill="#0284c7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="cnnAcc" name="1D CNN Baseline" fill="#64748b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 font-semibold text-xs text-slate-300">
          Detailed Metric Table (Verified Artifact: cnn_vs_eegnet_comparison.csv)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="px-4 py-2.5 font-semibold">Target Paradigm</th>
                <th className="px-4 py-2.5 font-semibold">Test Trials</th>
                <th className="px-4 py-2.5 font-semibold">1D CNN Accuracy</th>
                <th className="px-4 py-2.5 font-semibold">EEGNet Accuracy</th>
                <th className="px-4 py-2.5 font-semibold">Accuracy Delta</th>
                <th className="px-4 py-2.5 font-semibold">Better Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredData.map((row) => (
                <tr key={row.task} className="hover:bg-slate-800/30 text-slate-300">
                  <td className="px-4 py-2.5 font-sans font-medium text-white">{row.task}</td>
                  <td className="px-4 py-2.5 text-slate-400">{row.testSamples}</td>
                  <td className="px-4 py-2.5">{row.cnnAcc.toFixed(2)}%</td>
                  <td className="px-4 py-2.5 font-bold text-cyan-300">{row.eegnetAcc.toFixed(2)}%</td>
                  <td
                    className={`px-4 py-2.5 font-bold ${
                      row.delta > 0 ? 'text-emerald-400' : row.delta < 0 ? 'text-amber-400' : 'text-slate-400'
                    }`}
                  >
                    {row.delta > 0 ? `+${row.delta.toFixed(2)}%` : `${row.delta.toFixed(2)}%`}
                  </td>
                  <td className="px-4 py-2.5 font-sans">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        row.better === 'EEGNet'
                          ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                          : row.better === 'CNN1D'
                          ? 'bg-slate-800 text-slate-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {row.better}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
