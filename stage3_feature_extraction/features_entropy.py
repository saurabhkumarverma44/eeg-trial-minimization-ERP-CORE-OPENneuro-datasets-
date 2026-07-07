# stage3_feature_extraction/features_entropy.py
# Phase 3.6 — Spectral Entropy
#
# For every subject/task: compute Shannon spectral entropy per channel per
# epoch (normalised PSD treated as a probability distribution over
# frequency bins, entropy computed in nats then normalised by log(n_bins)
# so values lie in [0, 1]).
#
# Output per subject/task: results_batch/stage3/features/entropy/<dataset>/<subject>_<task>_entropy.npy
#   shape: (n_epochs, n_channels)
#
# Aggregated across ALL subjects within a dataset (as named in the spec,
# "entropy_features.npy"):
#   results_batch/stage3/features/entropy/<dataset>/entropy_features.npy
#   shape: (n_subjects, n_epochs_max, n_channels)  -- padded with NaN for
#   subjects/tasks with fewer epochs than the dataset's max, with a sidecar
#   JSON documenting the true n_epochs per subject so padding is never
#   mistaken for real data downstream.
#
# SAFE TO RUN: reads epochs-epo.fif from results_batch/. Writes only under
# results_batch/stage3/features/entropy/.

from __future__ import annotations

import json

import numpy as np
from scipy.signal import welch
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
    save_npy_checked,
)

log = get_logger("phase36_entropy")


def spectral_entropy_per_epoch(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    data: (n_epochs, n_channels, n_times)
    Returns (n_epochs, n_channels) normalised Shannon spectral entropy in [0, 1].
    """
    n_epochs, n_channels, n_times = data.shape
    nperseg = min(C.ENTROPY_NPERSEG, n_times)

    out = np.zeros((n_epochs, n_channels), dtype=np.float64)
    for e in range(n_epochs):
        freqs, psd = welch(data[e], fs=sfreq, nperseg=nperseg, axis=-1)
        mask = (freqs >= C.ENTROPY_PSD_FMIN) & (freqs <= C.ENTROPY_PSD_FMAX)
        psd_band = psd[:, mask]  # (n_channels, n_freqs_in_band)

        psd_sum = psd_band.sum(axis=-1, keepdims=True)
        psd_sum_safe = np.where(psd_sum <= 0, np.nan, psd_sum)
        p = psd_band / psd_sum_safe  # normalised distribution per channel

        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(p * np.log(np.where(p > 0, p, 1.0)), axis=-1)

        norm = np.log(psd_band.shape[-1]) if psd_band.shape[-1] > 1 else 1.0
        out[e] = ent / norm

    return np.nan_to_num(out, nan=0.0)


def process_file(ref: EpochFileRef) -> tuple[np.ndarray, list[str]] | None:
    epochs = mne.read_epochs(ref.path, preload=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False, exclude="bads")
    ch_names = [epochs.ch_names[p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]

    if data.shape[0] == 0:
        log.warning(f"  [SKIP] {ref.stem}: zero epochs")
        return None

    ent = spectral_entropy_per_epoch(data, sfreq)  # (n_epochs, n_channels)

    out_dir = C.ENTROPY_DIR / ref.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"{ref.key}_entropy.npy"
    clean = save_npy_checked(out_npy, ent, logger=log)
    log.info(f"[OK] {ref.stem}: entropy shape={ent.shape} clean={clean}")

    del data, epochs
    return ent, ch_names


def run_entropy(overwrite: bool = False) -> None:
    ensure_all_output_dirs()
    n_done, n_skipped, n_failed = 0, 0, 0

    # Aggregate per dataset, per task, for the dataset-level entropy_features.npy
    per_dataset_task: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}

    for ref in iter_epoch_files():
        out_npy = C.ENTROPY_DIR / ref.dataset / f"{ref.key}_entropy.npy"

        if out_npy.exists() and not overwrite:
            n_skipped += 1
            try:
                ent = np.load(out_npy)
                per_dataset_task.setdefault((ref.dataset, ref.task), []).append((ref.key, ent))
            except Exception:  # noqa: BLE001
                pass
            continue

        try:
            result = process_file(ref)
            if result is not None:
                ent, _ = result
                per_dataset_task.setdefault((ref.dataset, ref.task), []).append((ref.key, ent))
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ERROR] {ref.stem}: {exc}")
            n_failed += 1

    # Build dataset-level aggregate (padded with NaN to a common n_epochs_max)
    for (dataset, task), subj_list in per_dataset_task.items():
        if not subj_list:
            continue
        n_channels = subj_list[0][1].shape[1]
        n_epochs_max = max(arr.shape[0] for _, arr in subj_list)
        n_subjects = len(subj_list)

        agg = np.full((n_subjects, n_epochs_max, n_channels), np.nan, dtype=np.float64)
        true_n_epochs = {}
        for i, (subject, arr) in enumerate(subj_list):
            agg[i, : arr.shape[0], :] = arr
            true_n_epochs[subject] = int(arr.shape[0])

        out_dir = C.ENTROPY_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npy = out_dir / f"entropy_features_{task}.npy"
        np.save(out_npy, agg)

        meta = {
            "dataset": dataset, "task": task,
            "n_subjects": n_subjects, "n_epochs_max": n_epochs_max,
            "n_channels": n_channels,
            "true_n_epochs_per_subject": true_n_epochs,
            "note": "Padded with NaN beyond each subject's true n_epochs — see true_n_epochs_per_subject.",
        }
        with (out_dir / f"entropy_features_{task}.json").open("w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"[AGG] {dataset}/{task}: entropy_features shape={agg.shape} -> {out_npy}")

    log.info("=" * 60)
    log.info(f"Phase 3.6 entropy complete. done={n_done} skipped={n_skipped} failed={n_failed}")
    log.info(f"Output: {C.ENTROPY_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_entropy(overwrite=False)

