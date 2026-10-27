// frontend/src/pages/ModelManagement.tsx
import React, { useEffect, useState } from 'react';
import {
  Brain,
  Upload,
  CheckCircle2,
  AlertTriangle,
  Trash2,
  Play,
  Layers,
  FileCheck,
  ShieldCheck,
  Power,
  Info,
} from 'lucide-react';
import {
  fetchModels,
  uploadModel,
  validateModel,
  activateModel,
  deactivateModel,
  deleteModel,
  ModelItem,
  ModelValidationResult,
} from '../api/client';

export const ModelManagement: React.FC = () => {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<ModelValidationResult | null>(null);
  const [actionMessage, setActionMessage] = useState<string>('');

  // Upload Form State
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState<string>('');
  const [architecture, setArchitecture] = useState<string>('EEGNet');
  const [numChannels, setNumChannels] = useState<number>(31);
  const [samplingRate, setSamplingRate] = useState<number>(1000.0);
  const [classLabels, setClassLabels] = useState<string>('Go (0), No-Go (1)');

  const loadModels = async () => {
    try {
      const data = await fetchModels();
      setModels(data);
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setModelFile(file);
      if (!modelName) {
        setModelName(file.name.replace(/\.[^/.]+$/, ''));
      }
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelFile) {
      alert('Please select a model checkpoint file (.pt, .pth, .onnx).');
      return;
    }

    setUploading(true);
    setActionMessage('Uploading checkpoint to secure storage...');
    try {
      const formData = new FormData();
      formData.append('file', modelFile);
      formData.append('name', modelName || modelFile.name);
      formData.append('architecture', architecture);
      formData.append('num_channels', numChannels.toString());
      formData.append('sampling_rate', samplingRate.toString());

      const labelsArray = classLabels.split(',').map((s) => s.trim()).filter(Boolean);
      formData.append('class_labels_json', JSON.stringify(labelsArray));

      const newModel = await uploadModel(formData);
      setActionMessage(`Uploaded successfully: ${newModel.name}. Running initial validation...`);
      setModelFile(null);
      setModelName('');

      // Automatically trigger validation
      const val = await validateModel(newModel.id);
      setValidationResult(val);
      await loadModels();
    } catch (err: any) {
      setActionMessage(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleValidate = async (modelId: string) => {
    try {
      setActionMessage('Validating model weights, layers, and test inference...');
      const res = await validateModel(modelId);
      setValidationResult(res);
      setActionMessage(res.message);
      await loadModels();
    } catch (err: any) {
      setActionMessage(`Validation error: ${err.message}`);
    }
  };

  const handleActivate = async (modelId: string) => {
    try {
      await activateModel(modelId);
      setActionMessage('Model activated successfully.');
      await loadModels();
    } catch (err: any) {
      setActionMessage(`Activation error: ${err.message}`);
    }
  };

  const handleDeactivate = async (modelId: string) => {
    try {
      await deactivateModel(modelId);
      setActionMessage('Model deactivated.');
      await loadModels();
    } catch (err: any) {
      setActionMessage(`Deactivation error: ${err.message}`);
    }
  };

  const handleDelete = async (modelId: string, modelName: string, isBuiltin: boolean) => {
    if (isBuiltin) {
      alert('Built-in research models are strictly protected and cannot be deleted.');
      return;
    }
    if (!window.confirm(`Are you sure you want to delete model "${modelName}"?`)) {
      return;
    }
    try {
      await deleteModel(modelId);
      setActionMessage(`Model "${modelName}" removed.`);
      await loadModels();
    } catch (err: any) {
      setActionMessage(`Delete error: ${err.message}`);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Model Management & Persistent Registry</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Upload new PyTorch / ONNX models, validate tensor weights against channel dimensions, run test inference, and manage model lifecycle.
        </p>
      </div>

      {actionMessage && (
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs text-cyan-300 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>{actionMessage}</span>
          </div>
          <button onClick={() => setActionMessage('')} className="text-slate-500 hover:text-slate-300">
            &times;
          </button>
        </div>
      )}

      {/* Validation Result Modal / Callout */}
      {validationResult && (
        <div
          className={`p-5 rounded-xl border text-xs space-y-2 ${
            validationResult.is_valid
              ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-200'
              : 'bg-amber-950/20 border-amber-500/40 text-amber-200'
          }`}
        >
          <div className="flex items-center justify-between font-bold text-sm">
            <div className="flex items-center space-x-2">
              {validationResult.is_valid ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              )}
              <span>Validation Report: {validationResult.architecture}</span>
            </div>
            <button onClick={() => setValidationResult(null)} className="text-slate-400 hover:text-white">
              Dismiss
            </button>
          </div>
          <p>{validationResult.message}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 text-[11px] font-mono">
            <div>Expected Channels: {validationResult.num_channels}</div>
            <div>Test Inference: {validationResult.test_inference_ms ? `${validationResult.test_inference_ms} ms` : 'N/A'}</div>
            <div>Missing Keys: {validationResult.missing_keys.length}</div>
            <div>Unexpected Keys: {validationResult.unexpected_keys.length}</div>
          </div>
        </div>
      )}

      {/* Grid: Upload Box (Left) + Model Table (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Form */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Upload className="w-3.5 h-3.5 text-cyan-400" />
            <span>Upload Model Checkpoint</span>
          </h3>

          <form onSubmit={handleUpload} className="space-y-4 text-xs">
            {/* Drag and Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              className={`border-2 border-dashed rounded-xl p-5 text-center transition cursor-pointer ${
                modelFile
                  ? 'border-cyan-500/60 bg-cyan-950/20 text-cyan-300'
                  : 'border-slate-700 bg-slate-950/60 text-slate-400 hover:border-slate-600'
              }`}
            >
              <input
                type="file"
                id="modelFileInput"
                accept=".pt,.pth,.onnx"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setModelFile(e.target.files[0]);
                    if (!modelName) setModelName(e.target.files[0].name.replace(/\.[^/.]+$/, ''));
                  }
                }}
                className="hidden"
              />
              <label htmlFor="modelFileInput" className="cursor-pointer block">
                <Upload className="w-7 h-7 mx-auto mb-2 text-slate-400" />
                {modelFile ? (
                  <span className="font-semibold text-cyan-300 block truncate">{modelFile.name} ({(modelFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
                ) : (
                  <>
                    <span className="font-medium text-slate-300 block">Drag & drop checkpoint or click to browse</span>
                    <span className="text-[10px] text-slate-500 block mt-1">Supports .pt, .pth, .onnx (Max 250 MB)</span>
                  </>
                )}
              </label>
            </div>

            {/* Model Name */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Model Name</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. EEGNet 8-Channel FineTuned"
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
                required
              />
            </div>

            {/* Architecture */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Architecture</label>
              <select
                value={architecture}
                onChange={(e) => setArchitecture(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
              >
                <option value="EEGNet">EEGNet (Lawhern et al., 2018)</option>
                <option value="CNN1D">1D CNN (ERPConv1DNet)</option>
                <option value="Custom">Custom PyTorch Model</option>
              </select>
            </div>

            {/* Expected Channels */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Channels</label>
                <input
                  type="number"
                  value={numChannels}
                  onChange={(e) => setNumChannels(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
                  required
                />
              </div>
              <div>
                <label className="text-slate-400 font-semibold block mb-1">Sampling (Hz)</label>
                <input
                  type="number"
                  value={samplingRate}
                  onChange={(e) => setSamplingRate(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
                  required
                />
              </div>
            </div>

            {/* Class Labels */}
            <div>
              <label className="text-slate-400 font-semibold block mb-1">Class Labels (comma-separated)</label>
              <input
                type="text"
                value={classLabels}
                onChange={(e) => setClassLabels(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
              />
            </div>

            <button
              type="submit"
              disabled={uploading || !modelFile}
              className="w-full py-2.5 px-4 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 transition shadow-lg shadow-cyan-500/20"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{uploading ? 'Uploading & Validating...' : 'Upload & Register Model'}</span>
            </button>
          </form>
        </div>

        {/* Registered Models Table */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
          <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 flex justify-between items-center text-xs">
            <span className="font-semibold text-slate-300">Registered Model Checkpoints ({models.length})</span>
            <span className="text-[11px] text-slate-400">Weights-Only Safe Loading</span>
          </div>

          <div className="overflow-x-auto flex-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-2.5 font-semibold">Model Name</th>
                  <th className="px-4 py-2.5 font-semibold">Architecture</th>
                  <th className="px-4 py-2.5 font-semibold">Montage</th>
                  <th className="px-4 py-2.5 font-semibold">Status</th>
                  <th className="px-4 py-2.5 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {models.map((m) => (
                  <tr key={m.id} className="hover:bg-slate-800/30 text-slate-300 transition-colors">
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-2">
                        {m.is_builtin && (
                          <span className="px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-[9px] font-bold uppercase">
                            Research Checkpoint
                          </span>
                        )}
                        <span className="font-bold text-white">{m.name}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5 truncate max-w-xs">{m.filename}</div>
                    </td>
                    <td className="px-4 py-3 font-semibold text-cyan-400">{m.architecture}</td>
                    <td className="px-4 py-3 font-mono">{m.num_channels} ch @ {m.sampling_rate}Hz</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-2">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            m.validation_status === 'valid'
                              ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                              : m.validation_status === 'warning'
                              ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {m.validation_status}
                        </span>

                        {m.is_active && (
                          <span className="px-2 py-0.5 rounded bg-emerald-500 text-slate-950 font-bold text-[9px] uppercase">
                            Active
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end space-x-2">
                        {/* Validate */}
                        <button
                          onClick={() => handleValidate(m.id)}
                          title="Validate Checkpoint"
                          className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                        </button>

                        {/* Activate / Deactivate */}
                        {m.is_active ? (
                          <button
                            onClick={() => handleDeactivate(m.id)}
                            title="Deactivate Model"
                            className="p-1 rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30"
                          >
                            <Power className="w-3.5 h-3.5" />
                          </button>
                        ) : (
                          <button
                            onClick={() => handleActivate(m.id)}
                            title="Activate Model"
                            className="p-1 rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                          >
                            <Power className="w-3.5 h-3.5" />
                          </button>
                        )}

                        {/* Delete (disabled for builtin) */}
                        {!m.is_builtin ? (
                          <button
                            onClick={() => handleDelete(m.id, m.name, m.is_builtin)}
                            title="Delete Uploaded Model"
                            className="p-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        ) : (
                          <span title="Research model protected" className="text-slate-600 p-1">
                            <ShieldCheck className="w-3.5 h-3.5" />
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
