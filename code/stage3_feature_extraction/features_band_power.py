# stage3_feature_extraction/features_band_power.py
# Phase 3.3 — Band Power Features
#
# For every epoch file: compute relative band power per EEG electrode for
# delta(1-4) theta(4-8) alpha(8-12) beta(13-30) gamma(30-40) Hz, where
# "relative" = band power / total power in TOTAL_BAND_RANGE (1-40 Hz).
# Z-score normalise across epochs (per channel, per band) before saving.
#
# Output: results_batch/stage3/features/band_power/<dataset>/<subject>_<task>_bandpower.npy
#   shape: (n_epochs, n_channels, n_bands)   bands ordered per config3.FREQ_BANDS
# Plus a small JSON sidecar with channel names / band order / dataset metadata.
#
# SAFE TO RUN: reads epochs-epo.fif from results_batch/. Writes only under
# results_batch/stage3/features/band_power/.

from __future__ import annotations

import json

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, ensure_all_output_dirs, get_logger, iter_epoch_files,
    save_npy_checked, zscore,
)

log = get_logger("phase33_bandpower")

BAND_NAMES = list(C.FREQ_BANDS.keys())


def compute_band_power(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Returns (relative_power, ch_names) where relative_power has shape
    (n_epochs, n_channels, n_bands).
    """
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False, exclude="bads")
    ch_names = [epochs.ch_names[p] for p in picks]

    spectrum = epochs.compute_psd(
        method="welch",
        picks=picks,
        fmin=C.TOTAL_BAND_RANGE[0],
        fmax=C.TOTAL_BAND_RANGE[1],
        verbose="ERROR",
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

    total_mask = (freqs >= C.TOTAL_BAND_RANGE[0]) & (freqs <= C.TOTAL_BAND_RANGE[1])
    total_power = np.trapz(psds[:, :, total_mask], freqs[total_mask], axis=-1)  # (n_epochs, n_channels)
    total_power = np.where(total_power <= 0, np.nan, total_power)

    n_epochs, n_channels = psds.shape[0], psds.shape[1]
    rel_power = np.zeros((n_epochs, n_channels, len(BAND_NAMES)), dtype=np.float64)

    for b_idx, band in enumerate(BAND_NAMES):
        fmin, fmax = C.FREQ_BANDS[band]
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.trapz(psds[:, :, mask], freqs[mask], axis=-1)  # (n_epochs, n_channels)
        rel_power[:, :, b_idx] = band_power / total_power

    return rel_power, ch_names


def process_file(ref: EpochFileRef) -> None:
    out_dir = C.BANDPOWER_DIR / ref.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"{ref.key}_bandpower.npy"
    out_json = out_dir / f"{ref.key}_bandpower.json"

    epochs = mne.read_epochs(ref.path, preload=True, verbose="ERROR")
    if len(epochs) == 0:
        log.warning(f"  [SKIP] {ref.stem}: zero epochs")
        return

    rel_power, ch_names = compute_band_power(epochs)

    # Z-score across epochs (axis=0), independently per channel & band.
    rel_power_z = zscore(rel_power, axis=0)

    clean = save_npy_checked(out_npy, rel_power_z, logger=log)

    meta = {
        "dataset": ref.dataset,
        "subject": ref.subject,
        "task": ref.task,
        "n_epochs": int(rel_power_z.shape[0]),
        "n_channels": int(rel_power_z.shape[1]),
        "channels": ch_names,
        "bands": BAND_NAMES,
        "band_ranges_hz": C.FREQ_BANDS,
        "total_range_hz": list(C.TOTAL_BAND_RANGE),
        "zscored": True,
        "nan_inf_clean": clean,
    }
    with out_json.open("w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"[OK] {ref.stem}: bandpower shape={rel_power_z.shape} clean={clean}")


def run_band_power(overwrite: bool = False) -> None:
    ensure_all_output_dirs()
    n_done, n_skipped, n_failed = 0, 0, 0

    for ref in iter_epoch_files():
        out_npy = C.BANDPOWER_DIR / ref.dataset / f"{ref.key}_bandpower.npy"
        if out_npy.exists() and not overwrite:
            n_skipped += 1
            continue
        try:
            process_file(ref)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ERROR] {ref.stem}: {exc}")
            n_failed += 1

    log.info("=" * 60)
    log.info(f"Phase 3.3 band power complete. done={n_done} skipped={n_skipped} failed={n_failed}")
    log.info(f"Output: {C.BANDPOWER_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_band_power(overwrite=False)

