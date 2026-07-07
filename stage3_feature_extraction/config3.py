# stage3_feature_extraction/config3.py
#
# Shared configuration for Phase 3 (Trial Subsampling Grid & Feature Extraction).
# Does NOT modify or import anything from preprocessing_batch/config.py — Phase 2
# is frozen/submitted. This is a standalone config for Stage 3 only.
#
# IMPORTANT: run everything from project root:
#   cd /lp-dev/23BRS1328/erp_trial_min
#   python stage3_feature_extraction/run_stage3_all.py

from pathlib import Path

# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(".").resolve()

# Phase 2 output stays exactly where it is — read-only from here.
RESULTS_ROOT = PROJECT_ROOT / "results_batch"

# Stage 3 output now lives in its OWN top-level folder, separate from
# results_batch/ entirely (per request: a dedicated Stage 3 results
# directory, not nested inside Phase 2's output tree).
STAGE3_OUT_ROOT = PROJECT_ROOT / "stage3_results"

SUBSAMPLE_IDX_DIR   = STAGE3_OUT_ROOT / "subsample_indices"
BOOTSTRAP_DIR       = STAGE3_OUT_ROOT / "bootstrap"
BANDPOWER_DIR       = STAGE3_OUT_ROOT / "features" / "band_power"
PLV_DIR             = STAGE3_OUT_ROOT / "features" / "plv"
WPLI_DIR            = STAGE3_OUT_ROOT / "features" / "wpli"
ENTROPY_DIR         = STAGE3_OUT_ROOT / "features" / "entropy"
FEATURE_MATRIX_DIR  = STAGE3_OUT_ROOT / "feature_matrices"
LOG_DIR             = STAGE3_OUT_ROOT / "logs"
AUDIT_DIR           = STAGE3_OUT_ROOT / "audit"
ML_DIR              = STAGE3_OUT_ROOT / "ml"

ALL_OUT_DIRS = [
    STAGE3_OUT_ROOT, SUBSAMPLE_IDX_DIR, BOOTSTRAP_DIR, BANDPOWER_DIR,
    PLV_DIR, WPLI_DIR, ENTROPY_DIR, FEATURE_MATRIX_DIR, LOG_DIR, AUDIT_DIR,
    ML_DIR,
]

# ==================================================
# Datasets / epoch discovery
# ==================================================

DATASETS = ["ERP_CORE", "DS006018", "DS002680"]

# Epoch files are expected at:
#   results_batch/<DATASET>/<subject>/<task>/epochs-epo.fif
# (per Phase 2 handoff). We glob defensively in case naming varies slightly.
EPOCH_GLOB_PATTERNS = ["*-epo.fif", "epochs-epo.fif"]

# ==================================================
# 3.1 — Trial subsampling grid
# ==================================================

SUBSAMPLE_STEPS = [5, 10, 20, 30, 50, 75, 100, "full"]
RANDOM_SEED = 42

# ==================================================
# 3.2 — Bootstrap wrapper
# ==================================================

N_BOOTSTRAP_ITERS = 200
CI_LEVEL = 0.95  # 95% CI

# ==================================================
# 3.3 — Band power
# ==================================================

FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
TOTAL_BAND_RANGE = (1.0, 40.0)  # denominator for "relative" band power

# ==================================================
# 3.4 / 3.5 — Connectivity (PLV / wPLI)
# ==================================================

# Per the Phase 3 spec: alpha + beta are priority bands for connectivity.
CONNECTIVITY_BANDS = {
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
}

# Number of subjects to run the PLV-vs-wPLI documented comparison on (3.5).
WPLI_COMPARISON_N_SUBJECTS = 3

# ==================================================
# 3.6 — Spectral entropy
# ==================================================

ENTROPY_PSD_FMIN = 1.0
ENTROPY_PSD_FMAX = 40.0
ENTROPY_NPERSEG = 128  # Welch segment length (samples); auto-clamped to epoch length

# ==================================================
# Misc
# ==================================================

# Cap how many epochs are used for PLV/wPLI per file when doing connectivity
# (these are O(n_ch^2) per epoch and can get expensive on large files).
MAX_EPOCHS_FOR_CONNECTIVITY = None  # None = no cap; set an int to cap for a quick test run

