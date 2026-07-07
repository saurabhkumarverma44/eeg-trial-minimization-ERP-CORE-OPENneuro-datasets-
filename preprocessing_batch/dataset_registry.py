# preprocessing_batch/dataset_registry.py

from pathlib import Path
from config import ERP_CORE_ROOT, DS006018_ROOT, DS002680_ROOT

# ==================================================
# ERP CORE
# ==================================================

ERP_CORE_TASKS = [
    "P3",
    "MMN",
    "N170",
    "N2pc",
    "ERN",
    "N400",
    "LRP",
]

# ==================================================
# DS006018
# ==================================================

DS006018_TASKS = [
    "visualoddball",
    "auditoryoddball",
    "flanker",
    "visualsearch",
]

# ==================================================
# ROOTS (exported for convenience)
# ==================================================

DATASET_ROOTS = {
    "ERP_CORE": ERP_CORE_ROOT,
    "DS006018": DS006018_ROOT,
    "DS002680": DS002680_ROOT,
}