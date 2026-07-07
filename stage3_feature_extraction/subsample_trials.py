# stage3_feature_extraction/subsample_trials.py
# Phase 3.1 — Subsampling Function
#
# For every paradigm (dataset x task) and every subject's epoch file, generate
# a fixed, reproducible set of trial-index subsamples at each step in
# config3.SUBSAMPLE_STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"].
#
# Output (one JSON per subject/task):
#   results_batch/stage3/subsample_indices/<dataset>/<subject>_<task>_indices.json
#
# JSON structure:
#   {
#     "dataset": ..., "subject": ..., "task": ..., "n_total_epochs": N,
#     "steps": {
#        "5":   [idx0, idx1, ...],   # len 5, or fewer if N < 5 (flagged)
#        "10":  [...],
#        ...
#        "full": [0, 1, ..., N-1]
#     },
#     "step_integrity": {"5": true, "10": true, ..., "200": false}  # false if N < step
#   }
#
# SAFE TO RUN: only reads epochs-epo.fif headers (to get epoch counts) from
# results_batch/. Writes only under results_batch/stage3/subsample_indices/.

from __future__ import annotations

import json

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
)

log = get_logger("phase31_subsample")


def _subsample_indices_for_step(n_total: int, step, rng: np.random.Generator) -> list[int]:
    """
    Draw a subsample of trial indices (without replacement) for a given step.
    step == "full" -> all indices.
    step > n_total  -> all indices (can't oversample without replacement here;
                        this is recorded as a step-integrity failure, the
                        bootstrap wrapper in 3.2 is what handles resampling
                        WITH replacement when step <= n_total).
    """
    if step == "full":
        return list(range(n_total))
    step = int(step)
    if step >= n_total:
        return list(range(n_total))
    idx = rng.choice(n_total, size=step, replace=False)
    return sorted(int(i) for i in idx)


def build_indices_for_file(ref: EpochFileRef) -> dict:
    raw_epochs = mne.read_epochs(ref.path, preload=False, verbose="ERROR")
    n_total = len(raw_epochs)

    rng = np.random.default_rng(C.RANDOM_SEED + hash((ref.dataset, ref.subject, ref.task)) % (2**32))

    steps_out: dict[str, list[int]] = {}
    integrity: dict[str, bool] = {}

    for step in C.SUBSAMPLE_STEPS:
        key = str(step)
        idx = _subsample_indices_for_step(n_total, step, rng)
        steps_out[key] = idx
        if step == "full":
            integrity[key] = len(idx) == n_total
        else:
            integrity[key] = n_total >= int(step)

    return {
        "dataset": ref.dataset,
        "subject": ref.subject,
        "task": ref.task,
        "file_id": ref.file_id,
        "n_total_epochs": n_total,
        "steps": steps_out,
        "step_integrity": integrity,
    }


def run_subsampling(overwrite: bool = False) -> None:
    ensure_all_output_dirs()

    n_done = 0
    n_skipped = 0
    n_failed = 0
    integrity_failures = []

    for ref in iter_epoch_files():
        out_dir = C.SUBSAMPLE_IDX_DIR / ref.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ref.key}_indices.json"

        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue

        try:
            record = build_indices_for_file(ref)
            with out_path.open("w") as f:
                json.dump(record, f, indent=2)

            failed_steps = [s for s, ok in record["step_integrity"].items() if not ok]
            if failed_steps:
                integrity_failures.append(
                    f"{ref.dataset}/{ref.subject}/{ref.task}: "
                    f"n_total={record['n_total_epochs']} insufficient for steps {failed_steps}"
                )

            log.info(f"[OK] {ref.dataset} {ref.subject} {ref.task} "
                      f"(n_epochs={record['n_total_epochs']})")
            n_done += 1

        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ERROR] {ref.dataset} {ref.subject} {ref.task} {ref.path}: {exc}")
            n_failed += 1

    log.info("=" * 60)
    log.info(f"Phase 3.1 subsampling complete. done={n_done} skipped={n_skipped} failed={n_failed}")
    if integrity_failures:
        log.warning(f"Step-integrity warnings ({len(integrity_failures)} files had insufficient trials "
                     f"for at least one step):")
        for line in integrity_failures:
            log.warning(f"  - {line}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_subsampling(overwrite=False)

