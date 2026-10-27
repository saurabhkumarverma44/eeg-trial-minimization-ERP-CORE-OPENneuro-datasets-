# stage3_feature_extraction/bootstrap_grid.py
# Phase 3.2 — Bootstrap Wrapper
#
# Generic, reusable bootstrap resampling over the trial-subsampling grid:
# for each subject x paradigm x step, draw N_BOOTSTRAP_ITERS=1000 bootstrap
# samples (WITH replacement) of `step` trials from that subject's epochs,
# compute a scalar metric per draw, and report mean +/- 95% CI per step.
#
# Memory safety: we never hold all 1000 draws' raw epoch data in memory at
# once. For each iteration we only keep the scalar metric value, so memory
# use is O(n_iterations) floats, not O(n_iterations x n_epochs x n_channels x
# n_times) — this comfortably handles 40 subjects.
#
# Default metric: mean Global Field Power (GFP) of the evoked response built
# from the bootstrap-resampled trials. This is a standard, paradigm-agnostic
# scalar that's meaningful for any ERP task. Swap in a different metric_fn if
# your supervisor wants something else (e.g. peak amplitude at a specific
# channel/time-window) — see `metric_mean_gfp` below for the expected
# signature.
#
# Output: results_batch/stage3/bootstrap/<dataset>_<task>_bootstrap.csv
#   columns: dataset, subject, task, step, n_iterations, metric_mean,
#            ci_low, ci_high, metric_std
#
# SAFE TO RUN: only reads epochs-epo.fif from results_batch/. Writes only
# under results_batch/stage3/bootstrap/.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
)

log = get_logger("phase32_bootstrap")

FIELDNAMES = [
    "dataset", "subject", "task", "step", "n_total_epochs", "n_iterations",
    "metric_mean", "metric_std", "ci_low", "ci_high",
]


# ==================================================
# Metric functions
# ==================================================

def metric_mean_gfp(epochs_data: np.ndarray) -> float:
    """
    epochs_data: ndarray (n_trials, n_channels, n_times) — already EEG-picked.
    Returns mean Global Field Power (std across channels at each time point,
    averaged over time and trials) — a single scalar per bootstrap draw.
    """
    if epochs_data.size == 0:
        return float("nan")
    evoked = epochs_data.mean(axis=0)            # (n_channels, n_times)
    gfp = evoked.std(axis=0)                      # (n_times,)
    return float(gfp.mean())


METRIC_FN: Callable[[np.ndarray], float] = metric_mean_gfp
METRIC_NAME = "mean_gfp"


# ==================================================
# Bootstrap core
# ==================================================

def bootstrap_metric(
    epochs_data: np.ndarray,
    step,
    n_iter: int,
    rng: np.random.Generator,
    metric_fn: Callable[[np.ndarray], float] = METRIC_FN,
) -> np.ndarray:
    """
    GPU-accelerated (A100). Index draws stay on CPU via the same numpy
    Generator for identical sample selection vs the original loop-based
    version -- only the mean/std reduction runs on GPU.
    """
    n_trials = epochs_data.shape[0]
    draw_size = n_trials if step == "full" else int(step)

    if metric_fn is metric_mean_gfp:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_t = torch.as_tensor(epochs_data, dtype=torch.float32, device=dev)

        GPU_BATCH = 1000  # A100 has headroom; lower this if you hit OOM
        values = np.empty(n_iter, dtype=np.float64)
        for start in range(0, n_iter, GPU_BATCH):
            end = min(start + GPU_BATCH, n_iter)
            idx = rng.integers(0, n_trials, size=(end - start, draw_size))
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)

            sampled = data_t[idx_t]
            evoked = sampled.mean(dim=1)
            gfp = evoked.std(dim=1, unbiased=False)  # unbiased=False matches numpy's default ddof=0
            values[start:end] = gfp.mean(dim=1).cpu().numpy().astype(np.float64)
            del sampled, evoked, gfp, idx_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return values

    values = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n_trials, size=draw_size)
        values[i] = metric_fn(epochs_data[idx])
    return values


def _ci(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    lo = float(np.nanpercentile(values, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def process_file(ref: EpochFileRef, rng: np.random.Generator) -> list[dict]:
    epochs = mne.read_epochs(ref.path, preload=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False, exclude="bads")
    data = epochs.get_data(picks=picks)  # (n_trials, n_channels, n_times)
    n_total = data.shape[0]

    rows = []
    for step in C.SUBSAMPLE_STEPS:
        if step != "full" and n_total < 1:
            continue
        try:
            values = bootstrap_metric(data, step, C.N_BOOTSTRAP_ITERS, rng)
            lo, hi = _ci(values, C.CI_LEVEL)
            rows.append({
                "dataset": ref.dataset,
                "subject": ref.subject,
                "task": ref.task,
                "step": step,
                "n_total_epochs": n_total,
                "n_iterations": C.N_BOOTSTRAP_ITERS,
                "metric_mean": float(np.nanmean(values)),
                "metric_std": float(np.nanstd(values)),
                "ci_low": lo,
                "ci_high": hi,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning(f"  [WARN] bootstrap failed {ref.stem} step={step}: {exc}")

    del data, epochs  # explicit release before moving to next file
    return rows


def run_bootstrap_grid(overwrite: bool = False) -> None:
    ensure_all_output_dirs()
    rng = np.random.default_rng(C.RANDOM_SEED)

    # One CSV per paradigm (dataset__task), appended subject-by-subject so
    # memory never holds more than one subject's bootstrap draws at a time.
    # `truncated_this_run` tracks which output CSVs we've already reset
    # (when overwrite=True) so we only wipe each file once per run.
    truncated_this_run: set[Path] = set()

    n_files = 0
    for ref in iter_epoch_files():
        out_csv = C.BOOTSTRAP_DIR / f"{ref.dataset}_{ref.task}_bootstrap.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and out_csv not in truncated_this_run and out_csv.exists():
            out_csv.unlink()
        truncated_this_run.add(out_csv)

        try:
            rows = process_file(ref, rng)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ERROR] {ref.stem}: {exc}")
            continue

        if not rows:
            continue

        need_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
        with out_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if need_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

        n_files += 1
        log.info(f"[OK] {ref.stem}: bootstrap done for {len(rows)} steps")

    log.info("=" * 60)
    log.info(f"Phase 3.2 bootstrap grid complete. files processed={n_files}")
    log.info(f"Output: {C.BOOTSTRAP_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bootstrap_grid(overwrite=False)

