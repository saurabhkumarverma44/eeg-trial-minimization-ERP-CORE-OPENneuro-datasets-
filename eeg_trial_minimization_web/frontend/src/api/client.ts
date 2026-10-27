// frontend/src/api/client.ts

const API_BASE = '/api';

export interface HealthData {
  status: string;
  database_connected: boolean;
  models_registered: number;
  active_model?: string;
  stream_active: boolean;
  timestamp: string;
}

export interface SystemInfo {
  os: string;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  device_name: string;
  disk_free_gb: number;
  paths: Record<string, any>;
}

export interface InventoryItem {
  name: string;
  category: string;
  path: string;
  size_mb: number;
  file_type: string;
  status: string;
  description: string;
}

export interface InventoryResponse {
  project_root: string;
  items: InventoryItem[];
  total_files: number;
  total_size_mb: number;
}

export interface DatasetItem {
  dataset_id: string;
  name: string;
  path: string;
  file_type: string;
  size_mb: number;
  num_channels: number;
  channel_names: string[];
  sampling_rate: number;
  duration_sec?: number;
  n_epochs?: number;
  num_classes?: number;
  class_distribution?: Record<string, number>;
  task?: string;
  warnings: string[];
}

export interface ModelItem {
  id: string;
  name: string;
  filename: string;
  filepath: string;
  format: string;
  architecture: string;
  num_channels: number;
  sampling_rate: number;
  input_shape?: string;
  class_labels?: string[];
  preprocessing_info?: Record<string, any>;
  is_active: boolean;
  is_builtin: boolean;
  validation_status: string;
  validation_notes?: string;
  created_at: string;
}

export interface ModelValidationResult {
  model_id: string;
  is_valid: boolean;
  architecture: string;
  num_channels: number;
  sampling_rate: number;
  missing_keys: string[];
  unexpected_keys: string[];
  test_inference_success: boolean;
  test_inference_ms?: number;
  message: string;
}

export interface SingleInferenceResult {
  model_id: string;
  model_name: string;
  predicted_class: number;
  predicted_label: string;
  probabilities: number[];
  inference_time_ms: number;
  channel_count: number;
  time_points: number;
  actual_label?: number;
  source_info: string;
}

export interface BatchInferenceResult {
  model_id: string;
  model_name: string;
  total_evaluated: number;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
  avg_inference_time_ms: number;
  predictions: Array<{
    index: number;
    actual: number;
    predicted: number;
    probabilities: number[];
    correct: boolean;
  }>;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  dataset_name: string;
  task_name: string;
  model_type: string;
  model_id?: string;
  status: string;
  progress: number;
  current_step?: string;
  trial_counts: (number | string)[];
  summary_metrics?: {
    baseline_trials: number;
    baseline_accuracy: number;
    threshold_accuracy: number;
    min_stable_trials: number;
    min_stable_fraction: number;
    trial_reduction_percentage: number;
    total_test_samples: number;
  };
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
}

export interface StreamStatus {
  is_streaming: boolean;
  adapter_type: string;
  sampling_rate: number;
  channel_count: number;
  channel_names: string[];
  samples_transmitted: number;
  dropped_chunks: number;
  latency_ms: number;
  source_label: string;
}

export interface HardwareAdapter {
  id: string;
  name: string;
  status: string;
  description: string;
  supported_channels: number[];
  is_available: boolean;
  requires_external_service: boolean;
  config_params: Record<string, any>;
}

// REST Client functions
export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch health');
  return res.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/system/info`);
  if (!res.ok) throw new Error('Failed to fetch system info');
  return res.json();
}

export async function fetchInventory(): Promise<InventoryResponse> {
  const res = await fetch(`${API_BASE}/project/inventory`);
  if (!res.ok) throw new Error('Failed to fetch inventory');
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetItem[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function previewPreprocessing(payload: any): Promise<any> {
  const res = await fetch(`${API_BASE}/preprocessing/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Preprocessing preview failed');
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelItem[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function uploadModel(formData: FormData): Promise<ModelItem> {
  const res = await fetch(`${API_BASE}/models/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function validateModel(modelId: string): Promise<ModelValidationResult> {
  const res = await fetch(`${API_BASE}/models/${modelId}/validate`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Validation request failed');
  }
  return res.json();
}

export async function activateModel(modelId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/models/${modelId}/activate`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Activation failed');
  }
}

export async function deactivateModel(modelId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/models/${modelId}/deactivate`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Deactivation failed');
  }
}

export async function deleteModel(modelId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/models/${modelId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Deletion failed');
  }
}

export async function runSingleInference(payload: any): Promise<SingleInferenceResult> {
  const res = await fetch(`${API_BASE}/inference/single`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Inference failed');
  }
  return res.json();
}

export async function runBatchInference(payload: any): Promise<BatchInferenceResult> {
  const res = await fetch(`${API_BASE}/inference/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Batch inference failed');
  }
  return res.json();
}

export async function fetchInferenceHistory(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/inference/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function createExperiment(payload: any): Promise<{ experiment_id: string }> {
  const res = await fetch(`${API_BASE}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Experiment creation failed');
  }
  return res.json();
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return res.json();
}

export async function fetchExperiment(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/${id}`);
  if (!res.ok) throw new Error('Failed to fetch experiment');
  return res.json();
}

export async function fetchExperimentProgress(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/${id}/progress`);
  if (!res.ok) throw new Error('Failed to fetch progress');
  return res.json();
}

export async function fetchStreamStatus(): Promise<StreamStatus> {
  const res = await fetch(`${API_BASE}/streaming/status`);
  if (!res.ok) throw new Error('Failed to fetch stream status');
  return res.json();
}

export async function startStream(): Promise<void> {
  const res = await fetch(`${API_BASE}/streaming/start`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start stream');
}

export async function stopStream(): Promise<void> {
  const res = await fetch(`${API_BASE}/streaming/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop stream');
}

export async function configureStream(payload: any): Promise<void> {
  const res = await fetch(`${API_BASE}/streaming/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to configure stream');
}

export async function fetchHardwareAdapters(): Promise<HardwareAdapter[]> {
  const res = await fetch(`${API_BASE}/hardware/adapters`);
  if (!res.ok) throw new Error('Failed to fetch hardware adapters');
  return res.json();
}

export async function updateTrialBudget(trial_budget: number): Promise<void> {
  const res = await fetch(`${API_BASE}/streaming/trial_budget`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trial_budget }),
  });
  if (!res.ok) throw new Error('Failed to update trial budget');
}

export async function fetchComPorts(): Promise<{ ports: { device: string; description: string; hwid: string }[]; count: number }> {
  const res = await fetch(`${API_BASE}/hardware/com_ports`);
  if (!res.ok) throw new Error('Failed to fetch COM ports');
  return res.json();
}

export async function connectCapHardware(port: string = "AUTO", baudRate: number = 115200): Promise<{
  success: boolean;
  is_hardware: boolean;
  connected_port?: string;
  message: string;
  available_ports?: { device: string; description: string }[];
  error?: string;
}> {
  const res = await fetch(`${API_BASE}/hardware/connect_cap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ port, baud_rate: baudRate })
  });
  if (!res.ok) throw new Error('Hardware connection request failed');
  return res.json();
}

export interface BleDeviceItem {
  name: string;
  address: string;
  rssi?: number;
  is_h100: boolean;
}

export async function scanBleDevices(): Promise<{ devices: BleDeviceItem[]; count: number; error?: string }> {
  const res = await fetch(`${API_BASE}/hardware/ble/scan`, { method: 'POST' });
  if (!res.ok) throw new Error('BLE scan failed');
  return res.json();
}

export async function connectH100Ble(address: string = "AUTO", scale: number = 1.0, trialBudget: number = 63): Promise<any> {
  const res = await fetch(`${API_BASE}/hardware/ble/connect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ address, scale, trial_budget: trialBudget })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'H100 BLE connection failed');
  }
  return res.json();
}

export async function fetchLslStreams(): Promise<{ streams: any[]; count: number }> {
  const res = await fetch(`${API_BASE}/hardware/lsl/streams`);
  if (!res.ok) throw new Error('Failed to fetch LSL streams');
  return res.json();
}

export async function connectLslStream(streamName: string = "H100_BLE", trialBudget: number = 63): Promise<any> {
  const res = await fetch(`${API_BASE}/hardware/lsl/connect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stream_name: streamName, trial_budget: trialBudget })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to connect to LSL stream');
  }
  return res.json();
}

export async function updateHardwareGain(gain: number): Promise<{ status: string; gain: number }> {
  const res = await fetch(`${API_BASE}/hardware/gain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gain })
  });
  if (!res.ok) throw new Error('Failed to update hardware gain');
  return res.json();
}

export async function updateHardwareFilter(mode: string): Promise<{ status: string; filter_mode: string }> {
  const res = await fetch(`${API_BASE}/hardware/filter`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  });
  if (!res.ok) throw new Error('Failed to update hardware filter');
  return res.json();
}

export async function fetchHardwareFilter(): Promise<{ status: string; filter_mode: string }> {
  const res = await fetch(`${API_BASE}/hardware/filter`);
  if (!res.ok) throw new Error('Failed to fetch hardware filter');
  return res.json();
}

export interface ParticularTrialData {
  trial_index: number;
  total_trials?: number;
  channels: string[];
  samples: number[][];
  times: number[];
  label: number;
  class_label: string;
  split: string;
  sampling_rate: number;
}

export async function fetchParticularTrial(
  datasetId: string = "DS002680_gonogo_4",
  trialIndex: number = 0,
  channelsMode: string = "cap_8ch",
  split: string = "train"
): Promise<ParticularTrialData> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/trial/${trialIndex}?channels_mode=${channelsMode}&split=${split}`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to fetch trial');
  }
  return res.json();
}


