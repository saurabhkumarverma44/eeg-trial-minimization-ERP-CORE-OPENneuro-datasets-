# stage3_feature_extraction/stage3_utils.py
#
# Shared helpers used by every Phase 3 script. Pure utility module — no side
# effects on import, only reads from results_batch/ (Phase 2 output) and
# writes under results_batch/stage3/ (Phase 3 output).

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

import config3 as C


# ==================================================
# Logging
# ==================================================

def get_logger(name: str) -> logging.Logger:
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on re-import)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(C.LOG_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_all_output_dirs() -> None:
    for d in C.ALL_OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ==================================================
# Epoch file discovery
# ==================================================

@dataclass(frozen=True)
class EpochFileRef:
    dataset: str
    subject: str
    task: str
    path: Path

    @property
    def paradigm(self) -> str:
        """Paradigm key used throughout Stage 3 = '<dataset>__<task>'."""
        return f"{self.dataset}__{self.task}"

    @property
    def file_id(self) -> str:
        """
        Unique-per-file suffix. Some datasets (e.g. DS002680 gonogo) store
        SEPARATE epoch files per condition under the same subject/task
        folder (e.g. 'correct-epo.fif', 'distractor-epo.fif') rather than
        one combined epochs file. Using only (subject, task) to name output
        files would make those collide and silently overwrite each other.
        file_id is the epoch file's own stem (extension stripped), which is
        always unique within a subject/task folder, so every output file
        downstream is keyed by (dataset, subject, task, file_id) instead.
        """
        return self.path.stem

    @property
    def stem(self) -> str:
        return f"{self.dataset}_{self.subject}_{self.task}_{self.file_id}"

    @property
    def key(self) -> str:
        """Filename-safe unique key for this exact epoch file."""
        return f"{self.subject}_{self.task}_{self.file_id}"


def iter_epoch_files(dataset: str | None = None) -> Iterator[EpochFileRef]:
    """
    Walk results_batch/<DATASET>/<subject>/<task>/*-epo.fif (Phase 2 output)
    and yield EpochFileRef for each one found. Read-only — never touches
    Phase 2 files.
    """
    datasets = [dataset] if dataset else C.DATASETS

    for ds in datasets:
        ds_root = C.RESULTS_ROOT / ds
        if not ds_root.exists():
            continue

        seen: set[Path] = set()
        for pattern in C.EPOCH_GLOB_PATTERNS:
            for fif in sorted(ds_root.rglob(pattern)):
                if fif in seen:
                    continue
                seen.add(fif)

                rel = fif.relative_to(ds_root).parts
                subject = rel[0] if len(rel) >= 1 else "unknown"
                task = rel[1] if len(rel) >= 2 else "unknown"

                yield EpochFileRef(dataset=ds, subject=subject, task=task, path=fif)


def list_paradigms() -> list[str]:
    """All unique '<dataset>__<task>' paradigm keys found in results_batch/."""
    paradigms = sorted({ref.paradigm for ref in iter_epoch_files()})
    return paradigms


# ==================================================
# Safe array IO
# ==================================================

def save_npy_checked(path: Path, arr: np.ndarray, logger: logging.Logger | None = None) -> bool:
    """
    Save arr to .npy after checking for NaN/Inf. Returns True if saved cleanly,
    False if NaN/Inf were found (still saves the file, but logs a warning so
    the issue is visible in the audit trail rather than silently passing QC).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = True
    if arr.size > 0:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            clean = False
            msg = f"[NaN/Inf] {path.name}: nan={n_nan} inf={n_inf} shape={arr.shape}"
            if logger:
                logger.warning(msg)
            else:
                print("WARNING:", msg)

    np.save(path, arr)
    return clean


def zscore(arr: np.ndarray, axis: int, eps: float = 1e-12) -> np.ndarray:
    """Z-score normalise along `axis`, guarding against zero-variance columns."""
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std_safe = np.where(std < eps, 1.0, std)
    return (arr - mean) / std_safe


def safe_relative_path(p: Path) -> str:
    try:
        return str(p.relative_to(C.PROJECT_ROOT))
    except ValueError:
        return str(p)

