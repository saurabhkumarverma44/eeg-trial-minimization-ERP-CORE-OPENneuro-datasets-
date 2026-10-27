# stage3_feature_extraction/features_wpli.py
# Phase 3.5 — wPLI Matrices
#
# Same per-epoch, per-band structure as 3.4's PLV, but using weighted Phase
# Lag Index (robust to zero-lag volume-conduction artifacts, unlike PLV).
#
# Output:
#   results_batch/stage3/features/wpli/<dataset>/<subject>_<task>_<band>_wpli.npy
#       shape: (n_epochs, n_ch, n_ch)
#   results_batch/stage3/features/wpli/<dataset>/<subject>_<task>_<band>_wpli_flat.npy
#       shape: (n_epochs, n_ch*(n_ch-1)/2)
#
# Additionally, for the first WPLI_COMPARISON_N_SUBJECTS subjects encountered
# (config3.WPLI_COMPARISON_N_SUBJECTS, default 3), this script also loads the
# matching PLV output from 3.4 (if present) and writes a documented
# correlation/difference report:
#   results_batch/stage3/audit/plv_vs_wpli_comparison.csv
#
# SAFE TO RUN: reads epochs-epo.fif from results_batch/, and (read-only) PLV
# .npy files already produced by 3.4 in results_batch/stage3/features/plv/.
# Writes only under results_batch/stage3/features/wpli/ and
# results_batch/stage3/audit/.

from __future__ import annotations

import csv

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
    save_npy_checked,
)
from connectivity_core import wpli_matrix, upper_triangle_flat

log = get_logger("phase35_wpli")

COMPARISON_FIELDNAMES = [
    "dataset", "subject", "task", "band",
    "pearson_corr_plv_vs_wpli", "mean_abs_diff", "n_epochs_compared",
]


def process_file(ref: EpochFileRef) -> None:
    epochs = mne.read_epochs(ref.path, preload=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False, exclude="bads")
    data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_ch, _ = data.shape

    if n_epochs == 0 or n_ch < 2:
        log.warning(f"  [SKIP] {ref.stem}: n_epochs={n_epochs} n_ch={n_ch}")
        return

    n_max = n_epochs if C.MAX_EPOCHS_FOR_CONNECTIVITY is None else min(n_epochs, C.MAX_EPOCHS_FOR_CONNECTIVITY)

    out_dir = C.WPLI_DIR / ref.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for band, (fmin, fmax) in C.CONNECTIVITY_BANDS.items():
        wpli_stack = np.zeros((n_max, n_ch, n_ch), dtype=np.float64)
        for e in range(n_max):
            wpli_stack[e] = wpli_matrix(data[e], sfreq, fmin, fmax)

        flat = np.stack([upper_triangle_flat(wpli_stack[e]) for e in range(n_max)], axis=0)

        out_full = out_dir / f"{ref.key}_{band}_wpli.npy"
        out_flat = out_dir / f"{ref.key}_{band}_wpli_flat.npy"
        clean_full = save_npy_checked(out_full, wpli_stack, logger=log)
        clean_flat = save_npy_checked(out_flat, flat, logger=log)

        log.info(
            f"[OK] {ref.stem} [{band}]: wPLI shape={wpli_stack.shape} "
            f"flat={flat.shape} clean={clean_full and clean_flat}"
        )

    del data, epochs


def _compare_with_plv(ref: EpochFileRef) -> list[dict]:
    """
    For subjects selected for the documented comparison: load matching PLV
    (3.4) and wPLI (this script's) flattened outputs and report correlation
    + mean absolute difference per band.
    """
    rows = []
    for band in C.CONNECTIVITY_BANDS:
        plv_flat_path = C.PLV_DIR / ref.dataset / f"{ref.key}_{band}_plv_flat.npy"
        wpli_flat_path = C.WPLI_DIR / ref.dataset / f"{ref.key}_{band}_wpli_flat.npy"

        if not (plv_flat_path.exists() and wpli_flat_path.exists()):
            log.warning(
                f"  [SKIP comparison] {ref.stem} [{band}]: missing PLV or wPLI output "
                f"(run features_plv.py first)"
            )
            continue

        plv_flat = np.load(plv_flat_path)
        wpli_flat = np.load(wpli_flat_path)

        n = min(plv_flat.shape[0], wpli_flat.shape[0])
        if n == 0:
            continue

        a = plv_flat[:n].ravel()
        b = wpli_flat[:n].ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        mean_abs_diff = float(np.mean(np.abs(a - b)))

        rows.append({
            "dataset": ref.dataset,
            "subject": ref.subject,
            "task": ref.task,
            "band": band,
            "pearson_corr_plv_vs_wpli": corr,
            "mean_abs_diff": mean_abs_diff,
            "n_epochs_compared": n,
        })
    return rows


def run_wpli(overwrite: bool = False) -> None:
    ensure_all_output_dirs()
    n_done, n_skipped, n_failed = 0, 0, 0

    comparison_rows: list[dict] = []
    compared_subjects = 0

    for ref in iter_epoch_files():
        first_band = next(iter(C.CONNECTIVITY_BANDS))
        marker = C.WPLI_DIR / ref.dataset / f"{ref.key}_{first_band}_wpli.npy"

        if not (marker.exists() and not overwrite):
            try:
                process_file(ref)
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[ERROR] {ref.stem}: {exc}")
                n_failed += 1
                continue
        else:
            n_skipped += 1

        if compared_subjects < C.WPLI_COMPARISON_N_SUBJECTS:
            rows = _compare_with_plv(ref)
            if rows:
                comparison_rows.extend(rows)
                compared_subjects += 1

    if comparison_rows:
        out_csv = C.AUDIT_DIR / "plv_vs_wpli_comparison.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
            writer.writeheader()
            for row in comparison_rows:
                writer.writerow(row)
        log.info(f"PLV-vs-wPLI comparison documented for {compared_subjects} subjects -> {out_csv}")
    else:
        log.warning(
            "No PLV-vs-wPLI comparison rows generated — make sure features_plv.py "
            "(3.4) has been run BEFORE this script for at least one subject."
        )

    log.info("=" * 60)
    log.info(f"Phase 3.5 wPLI complete. done={n_done} skipped={n_skipped} failed={n_failed}")
    log.info(f"Output: {C.WPLI_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_wpli(overwrite=False)

