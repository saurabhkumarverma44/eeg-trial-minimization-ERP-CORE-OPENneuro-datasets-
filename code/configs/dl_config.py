import os
from pathlib import Path

BASE_DIR = Path("/lp-dev/23BRS1328/erp_trial_min")
RESULTS_BATCH_DIR = BASE_DIR / "results_batch"
RESULTS_DL_DIR = BASE_DIR / "results_dl"

DATASET_PATHS = {
    "ERP_CORE": RESULTS_BATCH_DIR / "ERP_CORE",
    "DS006018": RESULTS_BATCH_DIR / "DS006018",
    "DS002680": RESULTS_BATCH_DIR / "DS002680",
}

# Complete task paradigm registry across all 3 datasets (Total: 24 task paradigms)
DATASET_TASKS = {
    "ERP_CORE": ["P3", "N170", "N400", "MMN", "ERN", "LRP", "N2pc"],
    "DS006018": ["flanker", "visualoddball", "auditoryoddball", "visualsearch"],
    "DS002680": [f"gonogo_{i}" for i in range(1, 14)],
}

PREP_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    "num_workers": 4,
    "device": "cuda",
}
