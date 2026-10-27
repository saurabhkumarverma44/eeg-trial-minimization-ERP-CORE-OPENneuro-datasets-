# Trial Minimisation in EEG: Cross-Paradigm Learning Curves and Neural Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-00a087.svg)](https://mne.tools/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2+-61dafb.svg)](https://react.dev/)
[![Paper Status](https://img.shields.io/badge/Paper-Submitted%20to%20Brain%20Informatics-success.svg)]()

> Official repository for the paper:  
> **"Trial Minimisation in EEG: Subject Independent Learning Curves for Six ERP Components under a Matched Protocol"**  
> *Saurabh Kumar Verma, Aditi Vinodkumar Nair, Chirag Srivastava, Shridevi S (corresponding author), and Harish Gunasekaran.*  
> Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India.  
> *H. Gunasekaran is also with Specs SAS (BCI), Snap Inc., Paris, France.*

---

## 📌 Executive Summary & Research Question

Event-Related Potential (ERP) paradigms in cognitive neuroscience and brain-computer interfaces (BCIs) traditionally aggregate tens to hundreds of trials per subject to overcome low signal-to-noise ratios (SNR). However, lengthy recording sessions induce mental fatigue and limit clinical or real-world applicability.

This project addresses the central question:
> **As a classifier is given progressively fewer EEG trials from other subjects, at what point does it decode cognitive states above chance, and at what point does its learned feature representation stop resembling a validated reference scalp topography?**

We evaluate this systematically using a rigorous **Leave-One-Subject-Out (LOSO)** cross-validation protocol across **3 public datasets**, **22 paradigms**, **181 unique subjects**, and **6 model architectures**, spanning over **1,160 model/step training combinations**.

```
Raw EEG Multi-Channel Recordings (181 Subjects, 22 Paradigms)
   │
   ├── [Phase 2 Preprocessing] 0.1-40 Hz Bandpass, ICA Artifact Rejection, Avg Re-ref, Baseline (-200..0 ms)
   │
   ├── [Phase 3 Feature Extraction] 1,920-dim Vector: Band Power + PLV + wPLI + Spectral Entropy
   │      │
   │      └── Trial-Count Grid: S ∈ {5, 10, 20, 30, 50, 75, 100, 150, 200, Full}
   │
   ├── [Classical ML Track] Random Forest, XGBoost (GPU A100), SVM (RBF + Inner CV)
   │
   ├── [Deep Learning Track] Raw Single-Trial Epochs → CNN-1D, EEGNet (Lawhern et al., 2018)
   │
   ├── [Graph Neural Net Track] Electrode Graphs + Dynamic Attention → GAT (WBS Spec)
   │
   ├── [Spatial Fidelity Audit] Neuro-Fidelity Score (NFS) vs. Canonical Reference Topographies
   │
   └── [Full-Stack Platform] EEG Trial-Minimisation & Hardware BCI Web Suite (FastAPI + React 18)
```

---

## 🔬 Datasets & Montages

| Dataset | Subjects | Paradigms / Tasks | Montage | Role |
|:---|:---:|:---|:---:|:---|
| **ERP CORE** (Kappenman et al., 2021) | 40 | 7 (P3, MMN, N2pc, N400, N170, LRP, ERN) | 30 channels | Primary benchmark with published reference topographies |
| **OpenNeuro ds006018** | 127 | 2 used (auditoryoddball, visualoddball) | 26 channels | Cross-dataset generalisation (severe class imbalance study) |
| **OpenNeuro ds002680** | 14 | 13 Go/No-Go variants (`gonogo_1` to `13`) | 31 channels | Cross-dataset generalisation with derived reference topography |
| **Total Scope** | **181** | **22 Paradigms** | **26–31 ch** | **1,160+ total model-paradigm-step combinations** |

*Note on Data Availability:* Raw EEG data files are hosted openly by their respective repositories (OSF for ERP CORE, OpenNeuro for ds006018 and ds002680). Processed epoch files and feature matrices can be reproduced using the scripts in `code/`.

---

## 🏆 Key Experimental Findings

### 1. Deep Learning Outperforms Hand-Crafted Features (+41% Relative Gain)
When evaluated under identical, deterministic Leave-One-Subject-Out (LOSO) cross-validation at full trial counts:

| Model Architecture | Input Representation | Balanced Accuracy (Full Trials) | Scope |
|:---|:---|:---:|:---:|
| **EEGNet** (Lawhern et al., 2018) | Raw single-trial epochs (Z-scored) | **0.7395** | 22 paradigms |
| **CNN-1D** (GroupNorm + Temporal Pool) | Raw single-trial epochs (Z-scored) | **0.7392** | 22 paradigms |
| **Random Forest** (200 trees, max depth 10) | 1,920-dim feature vector | 0.5294 | 22 paradigms |
| **XGBoost** (GPU A100, lr 0.1, depth 6) | 1,920-dim feature vector | 0.5212 | 22 paradigms |
| **SVM** (RBF kernel, 3×2 inner CV) | 1,920-dim feature vector | 0.5185 | 20 paradigms |
| **GAT** (Graph Attention Network) | 5-dim bandpower + 4-dim PLV/wPLI edges | 0.5162 | 6 core paradigms |

### 2. Best Single-Paradigm Decoding
- **Lateralized Readiness Potential (LRP)**: SVM achieved 0.668, while **CNN-1D reached 0.956** and **EEGNet reached 0.971** balanced accuracy.
- **Error-Related Negativity (ERN)**: Classical ML plateaued at 0.525, while **CNN-1D achieved 0.766** and **EEGNet achieved 0.795**.
- **N400 (Semantic Processing)**: Classical ML reached 0.586, GAT reached 0.589, CNN-1D achieved 0.667, and **EEGNet achieved 0.692**.

### 3. Neuro-Fidelity Score (NFS) & Spatial Fidelity
NFS computes the correlation between model-derived importance maps and canonical scalp topographies:
- **Classical ML NFS**: RF ranges from -0.34 to +0.53; XGBoost ranges from -0.25 to +0.45 (verified across two independent pipelines).
- **GAT NFS**: Ranged from -0.31 to +0.14 across 60 combinations. Formal Wilcoxon signed-rank testing with Bonferroni correction showed that classical ML demonstrated statistically higher spatial fidelity on ERN ($p = 0.010$), with no significant difference on other components.
- **ds002680 Derived Reference Replication**: Classical ML again surpassed GAT on average NFS (RF: +0.106, XGB: +0.091 vs. GAT: -0.018).

### 4. Minimum Trial Convergence
- **Classical ML**: Converged reliably in 9 of 10 applicable cases with as few as **5 to 75 trials** (e.g., XGBoost converged at 5 trials for P3, MMN, and ERN; RF at 10 trials for N400 and 30 for P3).
- **GAT**: Flattened earlier and only converged in 1 of 6 paradigms (N2pc at step 20).

---

## 🛠️ Engineering Integrity: 16 Confirmed Bugs Found & Resolved

A central principle of this project was **verifying fixes against regenerated raw output data**, rather than relying on absence of errors or completion markers.

```
Preprocessing (Bugs 1-5)  ───► Classical ML (Bug 6)  ───► Deep Learning (Bugs 7-10) ───► GAT & Stats (Bugs 11-16)
• Event trigger code mismatch  • NFS reference 33 vs 30 ch • BFloat16/Numpy type cast  • Edge attribute wiring
• Metadata index corruption    • Two-pipeline stale cache  • BatchNorm LOSO collapse   • GlobalMeanPool spatial loss
• Orphaned stale files (1261)  • Inner CV grid explosion   • Raw µV amplitude scaling  • Unseeded RNG restarts
• Token substring collisions   • cuML Python 3.13 incompat • Dataset name prefix split • Pseudo-replication pooling
```

<details>
<summary><b>Click to expand the full 16-Bug Technical Audit Log</b></summary>

1. **Event-Code Identifier Mismatch (Preprocessing):** Trigger parser defaulted to sequential IDs, mislabeling 18/20 sampled trials in one paradigm. Resolved via explicit event dictionaries.
2. **Epoch-Concatenation Metadata Corruption (Preprocessing):** Sample-index metadata corrupted during sub-group merges. Fixed by explicitly re-indexing.
3. **Orphaned Stale Output Files (Preprocessing):** 1,261 un-migrated pre-fix files were silently reloaded. Detected via filesystem timestamp audit and purged.
4. **Keyword-Substring Token Collisions (Preprocessing):** Substring match (`"go"` in `"nogo"`, `"related"` in `"unrelated"`) collapsed N400 to a single class. Fixed via strict whole-token equality (`CONDITION_SPEC`).
5. **Duplicate Epoch Double-Counting (Preprocessing):** General trial files loaded alongside split condition files. Fixed by enforcing isolated condition paths.
6. **NFS 33 vs. 30 Channel-Count Mismatch (Stage 1 ML):** Reference topography contained 3 non-EEG EOG channels (`HEOG_left`, `HEOG_right`, `VEOG_lower`), silently zeroing NFS everywhere. An initial patch failed to regenerate outputs; isolated, re-fixed, and verified across both `results_stage4_classical_ml` and `results_stage4_gpu`.
7. **BFloat16 / NumPy Incompatibility (Deep Learning):** Inference in `torch.autocast(dtype=torch.bfloat16)` caused `TypeError: Got unsupported ScalarType BFloat16` when converting to NumPy arrays. Resolved with explicit float32 casting.
8. **BatchNorm / LOSO Statistics Drift (Deep Learning):** Running batch statistics learned over 9 subjects failed on the 10th unseen subject under LOSO, freezing validation balanced accuracy at 0.5000. Replaced with GroupNorm.
9. **Raw EEG Microvolt Amplitude Scale (Deep Learning - Root Cause):** Unnormalized raw EEG signals ($\sigma \approx 8 \times 10^{-6}$ V) produced vanishing gradients, causing frozen probabilities ($\sigma_{prob} = 0.000$). Resolved by fitting per-channel Z-score normalization strictly on train splits.
10. **Dataset Prefix Splitting Bug (Deep Learning):** `key.split("_", 1)` split `ERP_CORE_N400` into `ERP` and `CORE_N400`, failing all 7 ERP CORE paradigms. Fixed with explicit prefix matching (`ERP_CORE`, `DS006018`, `DS002680`).
11. **GAT Edge-Attribute Disconnection (GAT):** PLV/wPLI edge matrices were computed but omitted in `GATConv` calls. Wired via `edge_dim=4` into multi-head attention.
12. **GlobalMeanPool Spatial Degradation (GAT):** Averaging graph nodes destroyed spatial electrode signatures (isolated diagnostic showed baseline drop from 0.5548 to 0.5050). Documented and preserved in accordance with WBS specification.
13. **Unseeded Kernel Randomness (GAT):** Non-deterministic CUDA kernels yielded variable scores across restarts. Fixed by enforcing internal deterministic seeding inside `run_paradigm_step()`.
14. **Script Variable Scope & Ordering (GAT):** Output directories were referenced before initialization, breaking background detached executions. Resolved by restructuring script preamble.
15. **CUDA Device Enumeration Mismatch (Shared Infrastructure):** Discrepancy between system physical GPUs (8) and PyTorch-visible devices (7) crashed device queries. Mitigated via process-scoped `CUDA_VISIBLE_DEVICES`.
16. **Pseudo-Replication in ds002680 Statistical Testing (Validation):** Pooling subject $\times$ variant pairs treated 14 unique participants as 144 independent observations, artificially inflating significance ($p < 10^{-16}$). Corrected by subject-wise averaging ($N = 12$ independent subjects), confirming genuine fronto-central significance ($p = 0.003 - 0.006$).
</details>

---

## 🌐 Scientific Web Platform (`eeg_trial_minimization_web`)

The project includes a complete, production-ready research and BCI platform built to visualize trial-minimisation findings, run real-time inference, and stream live EEG.

```
eeg_trial_minimization_web/
├── backend/                  FastAPI (Python 3.10-3.12) server
│   ├── api/                  REST endpoints (datasets, experiments, inference, streaming, health)
│   ├── adapters/             Hardware adapters (H100 BLE, NeuroCap 8-ch, Playback, Simulated)
│   ├── models/               PyTorch architectures (EEGNet, CNN-1D, Conditional WGAN-GP)
│   ├── services/             Experiment runners, streaming engines, model registry
│   └── tests/                Comprehensive pytest suite
├── frontend/                 React 18 + Vite + Tailwind CSS + TypeScript
│   ├── src/pages/            Interactive research pages:
│   │   ├── Overview.tsx              Executive summary & verified research benchmarks
│   │   ├── TrialMinimisation.tsx     Interactive trial reduction curves & boundary checks
│   │   ├── ModelComparison.tsx       Multi-paradigm accuracy & confusion matrix explorer
│   │   ├── GANAugmentation.tsx       1D WGAN-GP synthetic trial generator & FID auditor
│   │   ├── LiveEEG.tsx               60 FPS Canvas multi-channel oscilloscope
│   │   ├── HardwareConnections.tsx   BLE & Serial EEG cap manager (H100 BLE integration)
│   │   ├── EEGPreprocessing.tsx      Bandpass, ICA, and artifact visualizer
│   │   └── ResultsReports.tsx        Scientific export engine & PDF generation
│   └── tailwind.config.js    Modern academic dark-mode palette
└── run_platform.bat / .ps1   One-click Windows launcher for backend + frontend
```

### Launching the Web Platform

#### Option A: Quickstart Launcher (Windows)
```powershell
cd D:\erp_trial_min_backup_20260626\erp_trial_min\eeg_trial_minimization_web
.\run_platform.ps1
```

#### Option B: Manual Startup
```powershell
# 1. Backend (Terminal 1)
cd eeg_trial_minimization_web
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Frontend (Terminal 2)
cd eeg_trial_minimization_web\frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser. The backend API documentation is available at **`http://localhost:8000/docs`**.

---

## 📁 Repository Layout

```
.
├── code/                               Core scientific pipeline scripts
│   ├── preprocessing_batch/            Filtering, ICA, channel montage, epoch extraction
│   ├── stage3_feature_extraction/      Band power, PLV, wPLI, entropy, trial subsampling
│   ├── stage4_classical_ml/            Random Forest, XGBoost, SVM drivers and nfs_fixed.py
│   ├── dl_models/                      CNN-1D, EEGNet, GAT, graph caches, LOSO trainers
│   ├── dl_prep/                        Subsampling, scaling, and explicit epoch loaders
│   ├── configs/                        Model & training hyperparameters
│   ├── run_classical_nfs.py            Classical ML execution driver with NFS map generation
│   ├── run_dl_v3.py, run_dl_v4.py      CNN-1D and EEGNet LOSO evaluation runs
│   ├── run_gat_v3.py, run_gat_v4.py    Graph Attention Network runs and attention maps
│   ├── analysis_v3.py                  Bootstrap CIs, slope regressions, paired t/Wilcoxon tests
│   └── cross_paradigm_null.py          Cross-component permutation null generator
├── results/                            Paper-ready CSV summaries and fold accounting
│   ├── learning_curves.csv             Mean balanced accuracy with 95% bootstrap intervals
│   ├── first_N_above_chance.csv        Empirical trial count threshold above chance (0.50)
│   ├── slope_ci.csv                    Learning slopes per ln(N) with cluster bootstrap
│   ├── tests_N200_vs_N20.csv           Paired trial-gain tests (N=20 vs N=200) with Holm p
│   ├── tests_model_pairs.csv           Cross-architecture paired tests at N=200
│   ├── classical_v2.csv, gat_nfs_v2.csv NFS values and statistical cross-component tests
│   ├── per_fold/                       Per-subject, per-step fold accuracy records
│   └── nfs_maps_*/                     Fold-level spatial activation & attention topographies
├── results_stage4_classical_ml/        Stage 4 execution artifacts & diagnostic logs
│   ├── convergence_plots/              66 detailed convergence curve PNGs
│   ├── Stage4_Final_Report.txt         Comprehensive Stage 4 execution audit
│   └── Stage4_GPU_vs_WBS_Verdict.txt   A100 GPU vs. CPU execution validation
├── results_gat/                        Stage 2 GAT artifacts, curves & comparison tables
├── notebooks/                          Iterative exploratory & accuracy improvement notebooks
├── preprocessing/                      Batch preprocessing scan and evoked scripts
├── eeg_trial_minimization_web/         Interactive full-stack web application & BCI platform
├── requirements.txt                    Root environment Python dependencies
└── config.yaml                         Global project paths and configuration
```

---

## 🚀 Reproduction Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/saurabhkumarverma44/eeg-trial-minimization-ERP-CORE-OPENneuro-datasets-.git
cd eeg-trial-minimization-ERP-CORE-OPENneuro-datasets-
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Preprocessing & Feature Extraction
```bash
# Execute batch preprocessing across datasets
python code/preprocessing_batch/run_all_datasets.py

# Extract 1,920-dim feature blocks and trial subsampling grids
python code/stage3_feature_extraction/run_stage3_all.py
```

### 3. Run Classical ML & Deep Learning LOSO Pipelines
```bash
# Classical ML with verified NFS maps
python code/run_classical_nfs.py

# Deep Learning (CNN-1D & EEGNet)
python code/run_dl_v4.py

# Graph Attention Network (GAT)
python code/run_gat_nfs.py
```

### 4. Reproduce Statistical Analysis & Paper Figures
```bash
python code/analysis_v3.py
python code/cross_paradigm_null.py
```

---

## 👥 Authors & Acknowledgments

- **Saurabh Kumar Verma** — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India (`saurabhsarthak5@gmail.com`)
- **Aditi Vinodkumar Nair** — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India
- **Chirag Srivastava** — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India
- **Dr. Shridevi S** *(Corresponding Author)* — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India (`shridevi.s@vit.ac.in`)
- **Harish Gunasekaran** — Specs SAS (BCI), Snap Inc., Paris, France; Centre for Neuroinformatics, VIT, Chennai, India

*Citation:*
```bibtex
@article{verma2026trialminimisation,
  title={Trial Minimisation in EEG: Subject Independent Learning Curves for Six ERP Components under a Matched Protocol},
  author={Verma, Saurabh Kumar and Nair, Aditi Vinodkumar and Srivastava, Chirag and Shridevi, S. and Gunasekaran, Harish},
  journal={Submitted to Brain Informatics},
  year={2026}
}
```
