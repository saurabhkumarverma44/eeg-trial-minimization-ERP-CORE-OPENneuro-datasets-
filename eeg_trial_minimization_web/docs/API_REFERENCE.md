# REST & WebSocket API Reference

The backend provides a complete REST and WebSocket interface running by default at `http://127.0.0.1:8000`.
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 1. System & Health Endpoints

### `GET /api/health`
Returns backend operational status, database connectivity, registered model count, active model, and stream state.
**Response (200 OK):**
```json
{
  "status": "healthy",
  "database_connected": true,
  "models_registered": 1,
  "active_model": "EEGNet DS002680 Go/No-Go #4 (Verified Research Model)",
  "stream_active": false,
  "timestamp": "2026-10-18T06:15:00.000000Z"
}
```

### `GET /api/system/info`
Returns operating system details, Python version, PyTorch version, CUDA capability, free disk space, and directory validation states.

### `GET /api/project/inventory`
Scans and catalogs all research assets (checkpoints, prepared datasets, FIF files, and result CSVs).

---

## 2. Dataset Endpoints

### `GET /api/datasets`
Returns metadata list of all available datasets across DS002680, ERP_CORE, and DS006018.

### `GET /api/datasets/{dataset_id}`
Returns detailed metadata for a single dataset (channels, sampling frequency, duration, class counts, and warnings).

### `POST /api/preprocessing/preview`
Applies digital bandpass filtering, notch filtering, and baseline subtraction to a sample epoch and returns before/after waveform series.
**Request Body:**
```json
{
  "dataset_id": "DS002680_gonogo_4",
  "channel_to_plot": "Fz",
  "config": {
    "l_freq": 0.1,
    "h_freq": 40.0,
    "notch_freq": 50.0,
    "rereference": "average",
    "tmin": -0.2,
    "tmax": 0.8,
    "baseline": [-0.2, 0.0],
    "reject_uv": 100.0
  }
}
```

---

## 3. Model Management Endpoints

### `GET /api/models`
Returns list of all models in the SQLite persistent registry.

### `POST /api/models/upload`
Uploads a new model checkpoint (`.pt`, `.pth`, `.onnx`) with metadata.
- Multipart Form: `file`, `name`, `architecture`, `num_channels`, `sampling_rate`, `class_labels_json`.

### `POST /api/models/{model_id}/validate`
Performs safe weight-only validation, layer inspection, and sample forward pass.

### `POST /api/models/{model_id}/activate`
Activates the target model for live inference.

### `DELETE /api/models/{model_id}`
Deletes an uploaded model from storage and registry. Built-in research checkpoints are protected (HTTP 403).

---

## 4. Inference Endpoints

### `POST /api/inference/single`
Runs real-time inference on a single epoch.
**Request Body:**
```json
{
  "dataset_name": "DS002680_gonogo_4",
  "epoch_index": 0
}
```
**Response (200 OK):**
```json
{
  "model_id": "eegnet-ds002680-gonogo-4",
  "model_name": "EEGNet DS002680 Go/No-Go #4",
  "predicted_class": 0,
  "predicted_label": "Go (0)",
  "probabilities": [0.9852, 0.0148],
  "inference_time_ms": 3.42,
  "channel_count": 31,
  "time_points": 1001,
  "actual_label": 0,
  "source_info": "Saved Dataset: DS002680_gonogo_4 (Test split, epoch #0)"
}
```

### `POST /api/inference/batch`
Evaluates a test split over N samples, computing accuracy, balanced accuracy, F1-score, and confusion matrix.

### `GET /api/inference/history`
Returns chronological log of recent model predictions stored in SQLite.

---

## 5. Experiment Endpoints

### `POST /api/experiments`
Launches an asynchronous trial-minimisation subsampling experiment.
**Request Body:**
```json
{
  "name": "Minimisation_DS002680_gonogo_4_EEGNet",
  "dataset_name": "DS002680",
  "task_name": "gonogo_4",
  "model_type": "EEGNet",
  "trial_counts": [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"],
  "num_folds": 5,
  "random_seed": 42,
  "epochs": 15,
  "use_gan_augmentation": false
}
```

### `GET /api/experiments/{id}/progress`
Returns live progress percentage, current training step, and intermediate results.

### `GET /api/results/{id}/export/csv`
Downloads the experiment subsampling results as a CSV file.

### `GET /api/results/{id}/figure`
Serves the publication-quality PNG stability curve figure.

---

## 6. Streaming & WebSockets

### `GET /api/streaming/status`
Returns active stream state, adapter type, sampling rate, and total transmitted sample count.

### `POST /api/streaming/start` & `POST /api/streaming/stop`
Starts and stops the streaming engine.

### `POST /api/streaming/configure`
Configures active stream adapter (`simulated`, `playback`, `lsl`, `tcp`, `physical`).

### `WebSocket /ws/eeg`
Broadcasts real-time JSON packets containing multi-channel chunks and event trigger markers.
**Message Format:**
```json
{
  "samples": [[...], [...]],
  "timestamps": [0.0, 0.001, 0.002],
  "events": [{"sample_offset": 14, "code": 1, "label": "Go Stimulus", "timestamp": 1729232100.5}],
  "is_simulated": true,
  "source_type": "Synthetic Simulated EEG Engine",
  "unit": "uV"
}
```
