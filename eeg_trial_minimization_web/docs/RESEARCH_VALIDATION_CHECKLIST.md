# Research Validation Checklist

Pre-flight checklist to verify reproducibility and methodological validity before running or publishing trial-minimisation experiments.

---

## 1. Asset & Model Checklist

- [x] **Primary EEGNet Checkpoint Verified**: Located at `results_dl_1/checkpoints_eegnet/eegnet_DS002680_gonogo_4_best.pt` (15.4 KB).
- [x] **Architecture Match Verified**: BatchNorm, `AdaptiveAvgPool2d((1, 1))`, and linear classifier (16 -> 2) verified matching `dl_models/eegnet_BACKUP_before_groupnorm.py`.
- [x] **Prepared Dataset Verified**: `results_dl_1/datasets/DS002680_gonogo_4_prepared.pt` (97.3 MB) containing `(1266, 31, 1001)` training, `(276, 31, 1001)` validation, and `(688, 31, 1001)` test epochs.
- [x] **Benchmark Accuracy Verified**: 95.93% (EEGNet) and 94.19% (CNN1D) matching `cnn_vs_eegnet_comparison.csv`.
- [x] **Minimum Stable Trials Verified**: 63 trials (95.02% reduction from 1,266 trials) matching `summary_minimization.csv`.

---

## 2. Experimental Rigor Guardrails

- [x] **No Test Leakage**: All subsampling and data scaling operations are fitted strictly on the training partition.
- [x] **Augmentation Integrity**: Synthetic trials from the 1D WGAN-GP are injected exclusively into the training fold and never evaluated in test metrics.
- [x] **No Channel Duplication**: Strict dimension validator prevents passing 8-channel cap data directly into 31-channel checkpoints.
- [x] **Authentic Subsampling**: Real trials are never duplicated to reach requested subsampling numbers.
- [x] **Seed Reproducibility**: Random seed (default: 42) ensures deterministic trial subsampling and stratified class distributions.
- [x] **Local Execution Verified**: Fully operational on laptop CPU/GPU without cloud dependencies or external paid APIs.
