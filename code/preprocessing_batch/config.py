# preprocessing_batch/config.py

from pathlib import Path

# ==================================================
# DATA ROOTS (relative to project root)
# ==================================================

ERP_CORE_ROOT = Path("data/raw/erp_core")
DS006018_ROOT = Path("data/raw/ds006018")
DS002680_ROOT = Path("data/raw/ds002680")

RESULTS_ROOT = Path("results_batch")

# ==================================================
# Preprocessing parameters
# ==================================================

LOW_CUT = 0.1
HIGH_CUT = 40.0

# ICA: keep enough components to explain this proportion of variance
ICA_COMPONENTS = 0.99

# Epoching window (seconds, relative to event)
TMIN = -0.2
TMAX = 0.8

# Baseline correction window
BASELINE = (None, 0.0)

# Simple amplitude-based artifact rejection (EEG channels)
REJECT_UV = 100e-6  # 100 µV peak-to-peak (per Phase 2 spec, section 2.6)

# Time points (in seconds) for topography plots
# P3 / oddball-like components around 0.3–0.5 s
DEFAULT_TOPO_TIMES = [0.3, 0.4, 0.5]

# ==================================================
# Montage (electrode position layout)
# ==================================================
# Applied after EOG/ECG channel typing so MNE has real per-channel
# 3-D positions for EEG channels (fixes "digitization points doesn't
# match number of EEG channels" errors in plot_topomap). on_missing is
# always "ignore" at the call site, so EOG/ECG/non-standard channel
# names are simply skipped rather than raising.
MONTAGE_NAME = "standard_1020"