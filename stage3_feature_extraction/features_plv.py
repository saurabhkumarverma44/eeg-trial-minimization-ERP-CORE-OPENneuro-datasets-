# stage3_feature_extraction/features_plv.py
# Phase 3.4 — PLV Matrices
#
# For every epoch file: compute a Phase-Locking-Value connectivity matrix per
# epoch, per priority frequency band (alpha, beta — config3.CONNECTIVITY_BANDS),
# then also save the flattened upper-triangle version for direct use as ML
# features.
#
# Output (per subject/task/band):
#   results_batch/stage3/features/plv/<dataset>/<subject>_<task>_<band>_plv.npy
#       shape: (n_epochs, n_ch, n_ch)
#   results_batch/stage3/features/plv/<dataset>/<subject>_<task>_<band>_plv_flat.npy
#       shape: (n_epochs, n_ch*(n_ch-1)/2)
#
# SAFE TO RUN: reads epochs-epo.fif from results_batch/. Writes only under
# results_batch/stage3/features/plv/.

from __future__ import annotations

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
    save_npy_checked,
)
from connectivity_core import plv_matrix, upper_triangle_flat

log = get_logger("phase34_plv")


def process_file(ref: EpochFileRef) -> None:
    epochs = mne.read_epochs(ref.path, preload=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False, exclude="bads")
    data = epochs.get_data(picks=picks)  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_ch, _ = data.shape

    if n_epochs == 0 or n_ch < 2:
        log.warning(f"  [SKIP] {ref.stem}: n_epochs={n_epochs} n_ch={n_ch}")
        return

    n_max = n_epochs if C.MAX_EPOCHS_FOR_CONNECTIVITY is None else min(n_epochs, C.MAX_EPOCHS_FOR_CONNECTIVITY)

    out_dir = C.PLV_DIR / ref.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for band, (fmin, fmax) in C.CONNECTIVITY_BANDS.items():
        plv_stack = np.zeros((n_max, n_ch, n_ch), dtype=np.float64)
        for e in range(n_max):
            plv_stack[e] = plv_matrix(data[e], sfreq, fmin, fmax)

        flat = np.stack([upper_triangle_flat(plv_stack[e]) for e in range(n_max)], axis=0)

        out_full = out_dir / f"{ref.key}_{band}_plv.npy"
        out_flat = out_dir / f"{ref.key}_{band}_plv_flat.npy"
        clean_full = save_npy_checked(out_full, plv_stack, logger=log)
        clean_flat = save_npy_checked(out_flat, flat, logger=log)

        log.info(
            f"[OK] {ref.stem} [{band}]: PLV shape={plv_stack.shape} "
            f"flat={flat.shape} clean={clean_full and clean_flat}"
        )

    del data, epochs


def run_plv(overwrite: bool = False) -> None:
    ensure_all_output_dirs()
    n_done, n_skipped, n_failed = 0, 0, 0

    for ref in iter_epoch_files():
        first_band = next(iter(C.CONNECTIVITY_BANDS))
        marker = C.PLV_DIR / ref.dataset / f"{ref.key}_{first_band}_plv.npy"
        if marker.exists() and not overwrite:
            n_skipped += 1
            continue
        try:
            process_file(ref)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ERROR] {ref.stem}: {exc}")
            n_failed += 1

    log.info("=" * 60)
    log.info(f"Phase 3.4 PLV complete. done={n_done} skipped={n_skipped} failed={n_failed}")
    log.info(f"Output: {C.PLV_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_plv(overwrite=False)

