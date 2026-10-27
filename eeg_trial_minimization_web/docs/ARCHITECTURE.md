# System Architecture & Technical Specification

## 1. High-Level System Architecture

The **EEG Trial-Minimisation Research Platform** is engineered as a decoupled, multi-tier full-stack application designed specifically for experimental electrophysiology, deep learning inference, and trial-subsampling research.

```
+-------------------------------------------------------------------------+
|                          Vite + React Frontend                          |
|  - 13 Specialized Research Dashboards                                   |
|  - 60 FPS HTML5 Canvas Scrolling Waveform Engine                        |
|  - Recharts Scientific Curve & Benchmark Visualizations                 |
|  - Model Upload, Validation & Activation Interface                      |
+------------------------------------+------------------------------------+
                                     | (HTTP REST / WebSocket at ws://.../ws/eeg)
                                     v
+-------------------------------------------------------------------------+
|                          FastAPI Backend Core                           |
|  - Asynchronous Request Routing & CORS                                  |
|  - Lifespan State Management                                            |
|  - Strict Path Traversal & File Extension Security                      |
|  - Chunked Multi-Channel Ring-Buffer WebSocket Broadcast                |
+------------------+------------------+-------------------+---------------+
                   |                  |                   |
                   v                  v                   v
     +--------------------+  +------------------+  +--------------------+
     |   SQLite Database  |  |  PyTorch Stack   |  | Acquisition Layer  |
     |  - Model Registry  |  |  - EEGNet (31ch) |  |  - Simulated Synth |
     |  - Experiment Logs |  |  - 1D CNN Base   |  |  - Saved Playback  |
     |  - Predictions Log |  |  - 1D WGAN-GP    |  |  - LSL / TCP Sockets|
     +--------------------+  +------------------+  +--------------------+
```

---

## 2. Component Decomposition

### 2.1 Backend (`backend/`)
- **`backend/core/`**:
  - `config.py`: Environment-driven settings using Pydantic V2. Validates `EEG_PROJECT_ROOT`, `EEG_WEB_ROOT`, and disk storage paths.
  - `database.py`: Asynchronous SQLite storage using `aiosqlite` with strict foreign key constraints and auto-seeding of the built-in research checkpoint.
  - `security.py`: Canonical path resolution ensuring all read/write operations stay within authorized project roots, blocking path-traversal attacks.
- **`backend/models/`**:
  - `eegnet.py`: PyTorch implementation of EEGNet with `BatchNorm2d`, `AdaptiveAvgPool2d((1, 1))`, and prefix stripper for `_orig_mod.` compiled checkpoints.
  - `cnn1d.py`: 1D CNN baseline (`ERPConv1DNet`) mirroring the research project's benchmark architecture.
  - `wgan.py`: 1D Conditional Wasserstein GAN with Gradient Penalty (`EEGGenerator`, `EEGCritic`, PSD cosine similarity, and ERP correlation evaluation).
  - `schemas.py`: Strongly-typed Pydantic schemas for requests and responses.
- **`backend/services/`**:
  - `dataset_service.py`: Discovers `.pt`, `.fif`, and BIDS datasets; extracts metadata without full-memory load.
  - `model_service.py`: Handles model upload, safe weight-only validation, activation, and protected deletion.
  - `inference_service.py`: Executes single-epoch and batch inference, profiling latency in milliseconds and enforcing Section 11 channel integrity.
  - `experiment_service.py`: Orchestrates multi-step trial minimisation experiments, computing statistical stability boundaries.
  - `streaming_service.py`: Controls real-time chunked signal broadcasting over WebSockets.
- **`backend/adapters/`**:
  - `simulated_adapter.py`: Generates synthetic multi-rhythm EEG with periodic ERP stimulus markers.
  - `playback_adapter.py`: Replays authentic recorded trials from saved local files.
  - `hardware_adapters.py`: LSL discovery receiver, TCP socket listener, and physical cap bridge interface.

### 2.2 Frontend (`frontend/src/`)
- Built with **React 18**, **TypeScript**, and **Tailwind CSS**.
- High-performance canvas-based multi-channel waveform renderer capable of scrolling 31 channels simultaneously without DOM lag.
- **Recharts** integration for trial-reduction curves, before/after filtering comparisons, and benchmark bar charts.
- 13 distinct functional pages accessible via persistent scientific sidebar navigation.

---

## 3. Strict Safety & Scientific Guardrails

1. **Section 11 Channel Incompatibility Constraint**:
   - The primary research checkpoint expects 31 channels.
   - Any input that does not match the model's required channel count (such as an 8-channel physical cap) is rejected with an explicit HTTP 400 error.
   - The platform never pads, duplicates, or invents missing electrode data.
2. **Train/Test Split Separation**:
   - Synthetic data from the 1D WGAN-GP is injected **strictly into training splits only**.
   - Synthetic samples are never mixed into validation or test sets.
3. **Research Asset Immutability**:
   - Built-in research checkpoints (`is_builtin == True`) cannot be deleted through the API.
   - Original research data outside `eeg_trial_minimization_web` is strictly read-only.
