// frontend/src/pages/ModelInference.tsx
import React, { useEffect, useState } from 'react';
import {
  PlaySquare,
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Layers,
  Brain,
  Zap,
  Activity,
  FileCheck,
} from 'lucide-react';
import {
  fetchModels,
  fetchDatasets,
  runSingleInference,
  runBatchInference,
  ModelItem,
  DatasetItem,
  SingleInferenceResult,
  BatchInferenceResult,
} from '../api/client';

export const ModelInference: React.FC = () => {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [selectedDatasetName, setSelectedDatasetName] = useState<string>('DS002680_gonogo_4');

  // Single inference state
  const [epochIndex, setEpochIndex] = useState<number>(0);
  const [singleResult, setSingleResult] = useState<SingleInferenceResult | null>(null);
  const [singleLoading, setSingleLoading] = useState<boolean>(false);
  const [singleError, setSingleError] = useState<string>('');

  // Batch inference state
  const [batchSamples, setBatchSamples] = useState<number>(50);
  const [batchResult, setBatchResult] = useState<BatchInferenceResult | null>(null);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string>('');

  // Channel mismatch tester (to test and verify Section 11 compatibility safety)
  const [simulateMismatch, setSimulateMismatch] = useState<boolean>(false);

  useEffect(() => {
    fetchModels()
      .then((mList) => {
        setModels(mList);
        const active = mList.find((m) => m.is_active) || mList[0];
        if (active) setSelectedModelId(active.id);
      })
      .catch((err) => console.log('Models fetch error:', err));

    fetchDatasets()
      .then((dList) => setDatasets(dList))
      .catch((err) => console.log('Datasets fetch error:', err));
  }, []);

  const handleRunSingle = async () => {
    setSingleLoading(true);
    setSingleError('');
    setSingleResult(null);
    try {
      let customData = undefined;
      if (simulateMismatch) {
        // Intentionally craft an 8-channel input to verify the strict channel validator
        customData = Array.from({ length: 8 }, () => Array(1001).fill(0));
      }

      const res = await runSingleInference({
        model_id: selectedModelId || undefined,
        dataset_name: selectedDatasetName,
        epoch_index: epochIndex,
        epoch_data: customData,
      });
      setSingleResult(res);
    } catch (err: any) {
      setSingleError(err.message);
    } finally {
      setSingleLoading(false);
    }
  };

  const handleRunBatch = async () => {
    setBatchLoading(true);
    setBatchError('');
    setBatchResult(null);
    try {
      const res = await runBatchInference({
        model_id: selectedModelId || undefined,
        dataset_name: selectedDatasetName,
        split: 'test',
        max_samples: batchSamples,
      });
      setBatchResult(res);
    } catch (err: any) {
      setBatchError(err.message);
    } finally {
      setBatchLoading(false);
    }
  };

  const selectedModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <PlaySquare className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Model Inference & Validation Suite</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Execute authentic epoch inference using verified research checkpoints with strict electrode montage validation and latency profiling.
        </p>
      </div>

      {/* Model & Dataset Selectors Strip */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div>
          <label className="text-slate-400 font-semibold block mb-1">Target Checkpoint Model</label>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.num_channels} ch @ {m.sampling_rate}Hz) {m.is_active ? '★ ACTIVE' : ''}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-slate-400 font-semibold block mb-1">Test Dataset</label>
          <select
            value={selectedDatasetName}
            onChange={(e) => setSelectedDatasetName(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
          >
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id} ({d.num_channels} ch)
              </option>
            ))}
          </select>
        </div>

        {/* Channel Mismatch Safety Tester */}
        <div className="flex items-center space-x-3 pt-5">
          <label className="flex items-center space-x-2 cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={simulateMismatch}
              onChange={(e) => setSimulateMismatch(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span>Simulate 8-Channel Input (Test Safety Validator)</span>
          </label>
        </div>
      </div>

      {/* Main Grid: Single Epoch (Left) & Batch Evaluation (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Single Epoch Inference */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                <span>Single-Epoch Real-Time Inference</span>
              </h3>
              <span className="text-[11px] text-slate-500 font-mono">Test Split Epoch</span>
            </div>

            <div className="flex items-center space-x-3 text-xs">
              <label className="text-slate-400 font-semibold">Epoch Index:</label>
              <input
                type="number"
                min={0}
                max={687}
                value={epochIndex}
                onChange={(e) => setEpochIndex(Number(e.target.value))}
                className="w-24 bg-slate-800 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono"
              />
              <button
                onClick={() => setEpochIndex((prev) => Math.floor(Math.random() * 688))}
                className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px]"
              >
                Random Epoch
              </button>
            </div>

            {/* Error Banner */}
            {singleError && (
              <div className="p-4 bg-red-950/30 border border-red-500/40 rounded-xl text-red-200 text-xs flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <div className="font-bold text-red-300">Channel Compatibility Failure</div>
                  <p className="leading-relaxed text-[11px]">{singleError}</p>
                </div>
              </div>
            )}

            {/* Result Display */}
            {singleResult && (
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Predicted Output</span>
                  <span className="text-[10px] text-emerald-400 font-mono flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{singleResult.inference_time_ms} ms</span>
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="text-2xl font-black text-cyan-400">{singleResult.predicted_label}</div>

                  {singleResult.actual_label !== undefined && singleResult.actual_label !== null && (
                    <div className="flex items-center space-x-1.5">
                      {singleResult.predicted_class === singleResult.actual_label ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold text-[11px] flex items-center space-x-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Ground Truth Match</span>
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded bg-red-500/15 text-red-400 border border-red-500/30 font-semibold text-[11px] flex items-center space-x-1">
                          <XCircle className="w-3.5 h-3.5" />
                          <span>Mismatch (Actual: Class {singleResult.actual_label})</span>
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Probability bars */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Go (Class 0): {(singleResult.probabilities[0] * 100).toFixed(1)}%</span>
                    <span>No-Go (Class 1): {(singleResult.probabilities[1] * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden flex">
                    <div style={{ width: `${singleResult.probabilities[0] * 100}%` }} className="bg-cyan-500 h-full" />
                    <div style={{ width: `${singleResult.probabilities[1] * 100}%` }} className="bg-purple-500 h-full" />
                  </div>
                </div>

                <div className="text-[10px] text-slate-500 font-mono pt-1">
                  Source: {singleResult.source_info} | Input Shape: ({singleResult.channel_count}, {singleResult.time_points})
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleRunSingle}
            disabled={singleLoading}
            className="w-full py-2.5 px-4 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 transition shadow-lg shadow-cyan-500/20"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{singleLoading ? 'Running Inference...' : `Execute Epoch #${epochIndex} Inference`}</span>
          </button>
        </div>

        {/* Batch Test Evaluation */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>Batch Test Set Evaluation</span>
              </h3>
              <span className="text-[11px] text-slate-500 font-mono">Statistical Profiling</span>
            </div>

            <div className="flex items-center space-x-3 text-xs">
              <label className="text-slate-400 font-semibold">Test Sample Count:</label>
              <select
                value={batchSamples}
                onChange={(e) => setBatchSamples(Number(e.target.value))}
                className="bg-slate-800 border border-slate-700 rounded px-2.5 py-1 text-slate-200"
              >
                <option value={20}>20 Samples (Fast)</option>
                <option value={50}>50 Samples</option>
                <option value={100}>100 Samples</option>
                <option value={250}>250 Samples</option>
              </select>
            </div>

            {batchError && (
              <div className="p-3 bg-red-950/30 border border-red-500/30 rounded-lg text-red-200 text-xs">
                {batchError}
              </div>
            )}

            {batchResult && (
              <div className="space-y-3">
                {/* Metrics 4-pack */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Accuracy</span>
                    <span className="text-base font-bold text-cyan-400">{(batchResult.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Balanced Acc</span>
                    <span className="text-base font-bold text-white">{(batchResult.balanced_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">F1-Score</span>
                    <span className="text-base font-bold text-emerald-400">{(batchResult.f1_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Avg Latency</span>
                    <span className="text-base font-mono text-purple-400">{batchResult.avg_inference_time_ms} ms</span>
                  </div>
                </div>

                {/* Confusion Matrix */}
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1.5 text-xs">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Confusion Matrix ({batchResult.total_evaluated} Samples)
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-center font-mono text-[11px] pt-1">
                    <div className="p-2 bg-slate-900 rounded border border-slate-800">
                      <span className="text-slate-400 block text-[9px]">True Class 0</span>
                      <span className="font-bold text-cyan-300">{batchResult.confusion_matrix[0]?.[0] || 0}</span>
                    </div>
                    <div className="p-2 bg-slate-900 rounded border border-slate-800">
                      <span className="text-slate-400 block text-[9px]">False Class 1</span>
                      <span className="font-bold text-red-300">{batchResult.confusion_matrix[0]?.[1] || 0}</span>
                    </div>
                    <div className="p-2 bg-slate-900 rounded border border-slate-800">
                      <span className="text-slate-400 block text-[9px]">False Class 0</span>
                      <span className="font-bold text-red-300">{batchResult.confusion_matrix[1]?.[0] || 0}</span>
                    </div>
                    <div className="p-2 bg-slate-900 rounded border border-slate-800">
                      <span className="text-slate-400 block text-[9px]">True Class 1</span>
                      <span className="font-bold text-purple-300">{batchResult.confusion_matrix[1]?.[1] || 0}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleRunBatch}
            disabled={batchLoading}
            className="w-full py-2.5 px-4 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs flex items-center justify-center space-x-2 border border-slate-700 transition"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{batchLoading ? 'Evaluating Batch...' : `Run Batch Evaluation (${batchSamples} Trials)`}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
