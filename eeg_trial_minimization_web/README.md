# EEG Trial-Minimisation & Research Web Platform

**Final-Year Research Project:** *Trial Minimisation Framework for EEG Classification using GAN-Based Augmentation and Machine Learning*

A scientific web application and research platform designed to empirically determine the minimum sufficient training trial fraction needed for stable ERP classification using deep learning (EEGNet, 1D CNN) and 1D Conditional WGAN-GP data augmentation.

---

## 1. Verified Research Benchmarks

The platform integrates and reproduces the experimental findings from the parent research project (`D:\erp_trial_min_backup_20260626\erp_trial_min`):

| Metric | Target Paradigm: `DS002680_gonogo_4` |
| :--- | :--- |
| **Verified Checkpoint** | `results_dl_1/checkpoints_eegnet/eegnet_DS002680_gonogo_4_best.pt` (15.4 KB) |
| **EEGNet Test Accuracy** | **95.93%** (688 test samples) |
| **CNN1D Baseline Accuracy** | **94.19%** (688 test samples) |
| **Baseline Training Trials** | **1,266 trials** (100% full dataset) |
| **Minimum Stable Trials** | **63 trials** (95% accuracy retention boundary) |
| **Trial Reduction** | **95.02% reduction** achieved |
| **Electrode Montage** | 31 channels, 1,000 Hz sampling rate, 1,001 time points per epoch |

---

## 2. Technology Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts, 60 FPS HTML5 Canvas.
- **Backend**: Python 3.12, FastAPI, Uvicorn (ASGI), Pydantic V2, aiosqlite, WebSockets.
- **Scientific Stack**: PyTorch 2.14, MNE-Python 1.13, NumPy, SciPy, scikit-learn, Pandas, Matplotlib.
- **Persistence**: SQLite (`storage/database.sqlite`), CSV exports, PNG scientific plots, and PyTorch `.pt` artifacts.
- **Host Environment**: Laptop CPU/CUDA execution (NVIDIA Jetson AGX Orin compatible).

---

## 3. Quickstart & Launch Instructions

### Option A: Launch Everything with One Command (Windows)
Open PowerShell inside `eeg_trial_minimization_web` and run:
```powershell
.\scripts\run_all.ps1
```
This automatically starts both the FastAPI backend and Vite frontend and provides live access links.

### Option B: Start Services Separately

#### 1. Backend Server
```powershell
# Inside eeg_trial_minimization_web
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health: `http://127.0.0.1:8000/api/health`
- Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`

#### 2. Frontend Server
```powershell
# Inside eeg_trial_minimization_web\frontend
npm run dev
```
- Dashboard URL: `http://localhost:5173`

### Option C: Linux / NVIDIA Jetson AGX Orin
```bash
chmod +x ./scripts/start.sh
./scripts/start.sh
```

---

## 4. Platform Features & Dashboards

1. **Overview Dashboard**: Pre-flight checklist, verified research metrics, storage integrity, and quick launches.
2. **Live EEG Stream**: 60 FPS multi-channel canvas scrolling waveforms, stimulus triggers, and live prediction overlay.
3. **Dataset Explorer**: Deep inspection of `.pt`, `.fif`, and BIDS archives with duration, sampling rate, and class counts.
4. **EEG Preprocessing**: Interactive zero-phase digital bandpass filtering, notch filtering, baseline subtraction, and before/after waveform comparison.
5. **Model Management**: Drag-and-drop checkpoint upload (`.pt`, `.pth`, `.onnx`), safe weights-only validation, forward pass test, activation, and protected deletion.
6. **Model Inference**: Single-epoch and batch inference with high-resolution latency profiling and strict channel validation.
7. **Trial Minimisation**: Subsampling grid (`5, 10, 20, 30, 50, 75, 100, 150, 200, Full`), real-time stability curve generation, and CSV export.
8. **GAN Augmentation**: 1D Conditional WGAN-GP generator/critic, PSD spectral similarity checks, and strict training-only injection.
9. **Model Comparison**: Empirical benchmark comparisons between EEGNet and CNN1D across all 24 tasks.
10. **Results & Reports**: Persistent experiment history, CSV/JSON downloads, and high-res scientific figures.
11. **Hardware Connections**: Acquisition PC setup, LSL discovery, TCP stream receiver, and physical cap documentation.
12. **Settings**: Directory path inspection, upload thresholds, networking, and security parameters.
13. **Diagnostics**: Real-time health probes, database connection checks, disk free space, and system event logs.

---

## 5. Automated Testing

Run the automated test suite with `pytest`:
```powershell
.\.venv\Scripts\pytest.exe backend\tests -v
```
**Test Results (100% Passing):**
- `test_list_datasets`: PASSED
- `test_preprocessing_preview`: PASSED
- `test_create_and_list_experiments`: PASSED
- `test_health_endpoint`: PASSED
- `test_system_info_endpoint`: PASSED
- `test_project_inventory_endpoint`: PASSED
- `test_single_epoch_real_inference`: PASSED
- `test_channel_incompatibility_validation`: PASSED
- `test_models_list_and_builtin`: PASSED
- `test_builtin_model_deletion_protected`: PASSED
- `test_streaming_endpoints`: PASSED
- `test_hardware_adapters_list`: PASSED

---

## 6. Channel Integrity & Scientific Safety (Section 11)

- The pre-trained research model expects **31 channels**.
- The physical cap provides **8 channels**.
- To prevent scientific inaccuracies and fabricated predictions, **the platform strictly forbids direct inference of 8-channel data on the 31-channel model** without retraining.
- Channels are **never padded or duplicated**.
- The system includes a dedicated upload and validation pathway for separately trained 8-channel models.

---

## 7. Directory Structure

```
eeg_trial_minimization_web/
├── backend/
│   ├── api/            # REST and WebSocket endpoints
│   ├── core/           # Config, database, security, and logging
│   ├── models/         # EEGNet, CNN1D, WGAN-GP architectures and schemas
│   ├── services/       # Inference, experiment, and streaming services
│   ├── adapters/       # Simulated, Playback, LSL, and Hardware adapters
│   └── tests/          # Automated test suite
├── frontend/
│   ├── src/
│   │   ├── api/        # Typed API client
│   │   ├── components/ # Sidebar, Navbar, and widgets
│   │   └── pages/      # 13 dedicated dashboards
│   ├── package.json
│   └── vite.config.ts
├── config/             # Config files
├── scripts/            # PowerShell and Bash startup scripts
├── docs/               # Architecture, API reference, and hardware guides
├── storage/            # Database, uploaded models, and experiment figures
├── .env.example
├── .gitignore
└── README.md
```
