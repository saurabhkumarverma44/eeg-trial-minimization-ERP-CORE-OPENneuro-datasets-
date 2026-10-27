// frontend/src/pages/TrialMinimisation.tsx
import React, { useState, useEffect } from 'react';
import {
  Layers,
  Play,
  RotateCcw,
  Sparkles,
  CheckCircle2,
  Clock,
  TrendingDown,
  Download,
  AlertCircle,
  BarChart2,
} from 'lucide-react';
import {
  createExperiment,
  fetchExperimentProgress,
  fetchExperiments,
  fetchDatasets,
  ExperimentSummary,
  DatasetItem,
} from '../api/client';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

export const TrialMinimisation: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('DS002680');
  const [selectedTask, setSelectedTask] = useState<string>('gonogo_4');
  const [modelType, setModelType] = useState<string>('EEGNet');
  const [epochs, setEpochs] = useState<number>(10);
  const [numFolds, setNumFolds] = useState<number>(5);
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [useGan, setUseGan] = useState<boolean>(false);
  const [ganRatio, setGanRatio] = useState<number>(0.5);

  // Active experiment state
  const [activeExpId, setActiveExpId] = useState<string | null>(null);
  const [expProgress, setExpProgress] = useState<any | null>(null);
  const [isLaunching, setIsLaunching] = useState<boolean>(false);

  // Historical experiments list
  const [history, setHistory] = useState<ExperimentSummary[]>([]);

  useEffect(() => {
    fetchDatasets().then((d) => setDatasets(d)).catch((e) => console.log(e));
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const exps = await fetchExperiments();
      setHistory(exps);
      // Auto-load most recent completed or running experiment if idle
      if (!activeExpId && exps.length > 0) {
        setActiveExpId(exps[0].id);
      }
    } catch (err) {
      console.log('Error loading experiments:', err);
    }
  };

  // Polling for active experiment progress
  useEffect(() => {
    if (!activeExpId) return;

    let isMounted = true;
    const poll = async () => {
      try {
        const prog = await fetchExperimentProgress(activeExpId);
        if (isMounted) {
          setExpProgress(prog);
          if (prog.status === 'completed' || prog.status === 'failed') {
            loadHistory();
          }
        }
      } catch (err) {
        console.log('Progress poll error:', err);
      }
    };

    poll();
    const interval = setInterval(poll, 2500);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [activeExpId]);

  const handleLaunchExperiment = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLaunching(true);
    setExpProgress(null);
    try {
      const res = await createExperiment({
        name: `Minimisation_${selectedDataset}_${selectedTask}_${modelType}`,
        dataset_name: selectedDataset,
        task_name: selectedTask,
        model_type: modelType,
        trial_counts: [5, 10, 20, 30, 50, 75, 100, 150, 200, 'full'],
        num_folds: numFolds,
        random_seed: randomSeed,
        epochs: epochs,
        use_gan_augmentation: useGan,
        gan_synthetic_ratio: ganRatio,
      });

      setActiveExpId(res.experiment_id);
    } catch (err: any) {
      alert(`Launch error: ${err.message}`);
    } finally {
      setIsLaunching(false);
    }
  };

  // Format curve points for Recharts
  const curveData = (expProgress?.results || []).map((r: any) => ({
    trials: r.n_train_trials,
    accuracy: Math.round(r.test_acc * 1000) / 10,
    fraction: Math.round(r.trial_fraction * 100),
    synthetic: r.synthetic_trials || 0,
  }));

  const baselineAcc = expProgress?.summary?.baseline_accuracy
    ? Math.round(expProgress.summary.baseline_accuracy * 1000) / 10
    : curveData[curveData.length - 1]?.accuracy || 95.9;

  const thresholdAcc = Math.round(baselineAcc * 0.95 * 10) / 10;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Trial Minimisation Framework</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Empirically identify the minimum sufficient training trial fraction needed for stable ERP classification using stratified subsampling and optional WGAN augmentation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Configuration Form */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Play className="w-3.5 h-3.5 text-cyan-400" />
            <span>Experiment Setup</span>
          </h3>

          <form onSubmit={handleLaunchExperiment} className="space-y-4 text-xs">
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Target Dataset</label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
              >
                <option value="DS002680">DS002680 (Go/No-Go)</option>
                <option value="ERP_CORE">ERP_CORE (P3, N170, MMN, etc.)</option>
                <option value="DS006018">DS006018 (Auditory/Visual Oddball)</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 font-semibold block mb-1">Task Paradigm</label>
              <input
                type="text"
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
                placeholder="e.g. gonogo_4 or P3"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono"
                required
              />
            </div>

            <div>
              <label className="text-slate-400 font-semibold block mb-1">Model Architecture</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
              >
                <option value="EEGNet">EEGNet (BatchNorm + AdaptivePool)</option>
                <option value="CNN1D">1D CNN (ERPConv1DNet Baseline)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Epochs / Subset</label>
                <input
                  type="number"
                  min={5}
                  max={50}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                />
              </div>
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Random Seed</label>
                <input
                  type="number"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
                />
              </div>
            </div>

            {/* Trial count grid preview */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Trial Subsampling Grid</label>
              <div className="flex flex-wrap gap-1 p-2 bg-slate-950 rounded border border-slate-800 text-[10px] font-mono text-cyan-400">
                {["5", "10", "20", "30", "50", "75", "100", "150", "200", "Full"].map((st) => (
                  <span key={st} className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">
                    {st}
                  </span>
                ))}
              </div>
            </div>

            {/* GAN Augmentation Toggle */}
            <div className="pt-2 border-t border-slate-800 space-y-2">
              <label className="flex items-center space-x-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={useGan}
                  onChange={(e) => setUseGan(e.target.checked)}
                  className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span className="font-semibold flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                  <span>Enable WGAN-GP Augmentation</span>
                </span>
              </label>

              {useGan && (
                <div className="space-y-1 pl-5">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Synthetic Ratio:</span>
                    <span className="font-mono text-purple-400">{(ganRatio * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={ganRatio}
                    onChange={(e) => setGanRatio(Number(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-[10px] text-slate-500 block">
                    Synthetic trials will be injected strictly into the training split only.
                  </span>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isLaunching || expProgress?.status === 'running'}
              className="w-full py-2.5 px-4 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 transition shadow-lg shadow-cyan-500/20"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isLaunching ? 'Starting Worker...' : 'Launch Minimisation Scan'}</span>
            </button>
          </form>
        </div>

        {/* Right: Live Progress & Stability Curve */}
        <div className="lg:col-span-2 space-y-6">
          {/* Progress / Status Bar */}
          {expProgress && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <span className="font-semibold text-slate-200">{expProgress.current_step}</span>
                </div>
                <span className="font-mono text-cyan-400 font-bold">{expProgress.progress?.toFixed(0)}%</span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  style={{ width: `${expProgress.progress || 0}%` }}
                  className="bg-gradient-to-r from-cyan-500 to-sky-400 h-full transition-all duration-300"
                />
              </div>

              {/* Key finding metrics if available */}
              {expProgress.summary && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-xs text-center border-t border-slate-800/80">
                  <div className="p-2 bg-slate-950 rounded">
                    <span className="text-[10px] text-slate-400 block">Baseline Accuracy</span>
                    <span className="font-bold text-white">{(expProgress.summary.baseline_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2 bg-slate-950 rounded">
                    <span className="text-[10px] text-slate-400 block">95% Stability Threshold</span>
                    <span className="font-bold text-amber-400">{(expProgress.summary.threshold_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2 bg-slate-950 rounded">
                    <span className="text-[10px] text-slate-400 block">Min Stable Trials</span>
                    <span className="font-bold text-emerald-400">{expProgress.summary.min_stable_trials} Trials</span>
                  </div>
                  <div className="p-2 bg-slate-950 rounded">
                    <span className="text-[10px] text-slate-400 block">Trial Reduction</span>
                    <span className="font-bold text-cyan-400">{expProgress.summary.trial_reduction_percentage}%</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Accuracy vs Trial Count Curve */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" />
                  <span>Performance vs Training Trial Count Curve</span>
                </h3>
                <p className="text-[11px] text-slate-400">
                  Stability boundary defined at 95% of full baseline accuracy.
                </p>
              </div>

              {activeExpId && (
                <a
                  href={`/api/results/${activeExpId}/export/csv`}
                  className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export CSV</span>
                </a>
              )}
            </div>

            <div className="w-full h-72 pt-2">
              {curveData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curveData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="trials"
                      stroke="#64748b"
                      fontSize={11}
                      label={{ value: 'Training Trials', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
                    />
                    <YAxis
                      domain={[50, 100]}
                      stroke="#64748b"
                      fontSize={11}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                      formatter={(val: any) => [`${val}%`, 'Test Accuracy']}
                      labelFormatter={(val) => `Trials: ${val}`}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                    <ReferenceLine y={baselineAcc} stroke="#64748b" strokeDasharray="3 3" label={{ value: `Baseline: ${baselineAcc}%`, fill: '#94a3b8', fontSize: 10 }} />
                    <ReferenceLine y={thresholdAcc} stroke="#ef4444" strokeDasharray="2 2" label={{ value: `95% Threshold: ${thresholdAcc}%`, fill: '#f87171', fontSize: 10 }} />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#0284c7"
                      strokeWidth={2.5}
                      dot={{ r: 4, fill: '#0284c7' }}
                      activeDot={{ r: 6 }}
                      name="Subsampled Accuracy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                  <span>Launch an experiment or select a past run from history to visualize stability curve.</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
