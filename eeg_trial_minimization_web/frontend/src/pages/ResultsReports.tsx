// frontend/src/pages/ResultsReports.tsx
import React, { useEffect, useState } from 'react';
import {
  FileSpreadsheet,
  Download,
  Eye,
  Calendar,
  Layers,
  Brain,
  CheckCircle2,
  AlertCircle,
  FileText,
  Image as ImageIcon,
} from 'lucide-react';
import { fetchExperiments, fetchExperiment, ExperimentSummary } from '../api/client';

export const ResultsReports: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExp, setSelectedExp] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchExperiments()
      .then((data) => {
        setExperiments(data);
        if (data.length > 0) {
          fetchExperiment(data[0].id).then((d) => setSelectedExp(d));
        }
      })
      .catch((err) => console.log('Experiments error:', err))
      .finally(() => setLoading(false));
  }, []);

  const handleSelectExperiment = async (id: string) => {
    try {
      const details = await fetchExperiment(id);
      setSelectedExp(details);
    } catch (err) {
      console.error('Failed to load experiment details:', err);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <FileSpreadsheet className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Results, Experiment Logs & Scientific Reports</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Review historical trial minimisation trials, download publication-ready CSV/JSON tables, and view high-resolution scientific figures.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Experiment History Table */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
          <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 flex justify-between items-center text-xs">
            <span className="font-semibold text-slate-300">Logged Experiments ({experiments.length})</span>
            <span className="text-slate-500 font-mono text-[11px]">Database Records</span>
          </div>

          <div className="divide-y divide-slate-800/60 overflow-y-auto max-h-[550px]">
            {experiments.map((exp) => {
              const isSelected = selectedExp?.experiment?.id === exp.id;
              return (
                <div
                  key={exp.id}
                  onClick={() => handleSelectExperiment(exp.id)}
                  className={`p-4 cursor-pointer transition text-xs space-y-1.5 ${
                    isSelected
                      ? 'bg-cyan-500/10 border-l-2 border-cyan-400'
                      : 'hover:bg-slate-800/40 text-slate-300'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-bold text-white truncate max-w-[180px]">{exp.name}</span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-semibold ${
                        exp.status === 'completed'
                          ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                          : exp.status === 'running'
                          ? 'bg-cyan-500/15 text-cyan-400 animate-pulse'
                          : 'bg-red-500/15 text-red-400'
                      }`}
                    >
                      {exp.status}
                    </span>
                  </div>

                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>{exp.dataset_name}_{exp.task_name}</span>
                    <span className="font-mono text-cyan-300">{exp.model_type}</span>
                  </div>

                  {exp.summary_metrics && (
                    <div className="text-[10px] text-emerald-400 font-mono">
                      Min Stable: {exp.summary_metrics.min_stable_trials} trials ({exp.summary_metrics.trial_reduction_percentage}% red.)
                    </div>
                  )}

                  <div className="text-[10px] text-slate-500 pt-0.5">
                    {exp.created_at?.slice(0, 19).replace('T', ' ')}
                  </div>
                </div>
              );
            })}

            {experiments.length === 0 && !loading && (
              <div className="p-8 text-center text-slate-500 text-xs">
                No experiments logged yet. Run a trial minimisation experiment to generate results.
              </div>
            )}
          </div>
        </div>

        {/* Right: Detailed Result View & Downloads */}
        <div className="lg:col-span-2 space-y-6">
          {selectedExp?.experiment ? (
            <div className="space-y-6">
              {/* Header card */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white">{selectedExp.experiment.name}</h3>
                    <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                      ID: {selectedExp.experiment.id} | Task: {selectedExp.experiment.dataset_name}_{selectedExp.experiment.task_name}
                    </p>
                  </div>

                  <div className="flex items-center space-x-2">
                    <a
                      href={`/api/results/${selectedExp.experiment.id}/export/csv`}
                      className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-sm transition"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download CSV</span>
                    </a>
                  </div>
                </div>

                {/* Summary Metrics Banner */}
                {selectedExp.experiment.summary_metrics && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs">
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400 block uppercase font-semibold">Baseline Accuracy</span>
                      <span className="text-lg font-bold text-white">
                        {(selectedExp.experiment.summary_metrics.baseline_accuracy * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400 block uppercase font-semibold">Min Stable Trials</span>
                      <span className="text-lg font-bold text-emerald-400">
                        {selectedExp.experiment.summary_metrics.min_stable_trials} Trials
                      </span>
                    </div>

                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400 block uppercase font-semibold">Stability Threshold</span>
                      <span className="text-lg font-bold text-amber-400">
                        {(selectedExp.experiment.summary_metrics.threshold_accuracy * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400 block uppercase font-semibold">Trial Reduction</span>
                      <span className="text-lg font-bold text-cyan-400">
                        {selectedExp.experiment.summary_metrics.trial_reduction_percentage}%
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* High-Resolution Scientific Curve Plot */}
              {selectedExp.experiment.figure_url && (
                <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                    <span className="flex items-center space-x-2">
                      <ImageIcon className="w-4 h-4 text-cyan-400" />
                      <span>Publication-Quality Minimisation Figure</span>
                    </span>
                    <a
                      href={selectedExp.experiment.figure_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 text-[11px] underline"
                    >
                      Open Full Res
                    </a>
                  </div>
                  <div className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center p-2">
                    <img
                      src={selectedExp.experiment.figure_url}
                      alt="Trial Minimization Curve"
                      className="max-h-96 w-auto object-contain rounded"
                    />
                  </div>
                </div>
              )}

              {/* Results Table */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
                <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 text-xs font-semibold text-slate-300">
                  Subsampling Performance Breakdown
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 font-mono text-[11px]">
                      <tr>
                        <th className="px-4 py-2 font-semibold">Trials</th>
                        <th className="px-4 py-2 font-semibold">Fraction</th>
                        <th className="px-4 py-2 font-semibold">Test Acc</th>
                        <th className="px-4 py-2 font-semibold">Precision</th>
                        <th className="px-4 py-2 font-semibold">Recall</th>
                        <th className="px-4 py-2 font-semibold">F1-Score</th>
                        <th className="px-4 py-2 font-semibold">AUC</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                      {selectedExp.results?.map((r: any, idx: number) => (
                        <tr key={idx} className="hover:bg-slate-800/30 text-slate-300">
                          <td className="px-4 py-2 font-bold text-white">{r.n_train_trials}</td>
                          <td className="px-4 py-2">{(r.trial_fraction * 100).toFixed(1)}%</td>
                          <td className="px-4 py-2 text-cyan-300 font-bold">{(r.test_acc * 100).toFixed(2)}%</td>
                          <td className="px-4 py-2">{(r.precision * 100).toFixed(1)}%</td>
                          <td className="px-4 py-2">{(r.recall * 100).toFixed(1)}%</td>
                          <td className="px-4 py-2">{(r.f1_score * 100).toFixed(1)}%</td>
                          <td className="px-4 py-2 text-purple-300">{(r.roc_auc * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-12 text-center text-slate-500 text-xs">
              <FileSpreadsheet className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <span>Select an experiment run from the history log to inspect detailed metrics and export reports.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
