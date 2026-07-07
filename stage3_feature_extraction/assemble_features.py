# stage3_feature_extraction/assemble_features.py
# Phase 3.7 — Feature Assembly
#
# Combines, for every paradigm (dataset__task) and every subsampling step
# (config3.SUBSAMPLE_STEPS), the outputs of 3.1 (trial indices), 3.3 (band
# power), 3.4 (PLV), 3.5 (wPLI), and 3.6 (entropy) into ONE unified feature
# matrix ready for Stage 1 ML input.
#
# Per-epoch feature vector =
#   [ band_power.flatten()        -> n_ch * n_bands
#     PLV_alpha_flat               -> n_ch*(n_ch-1)/2
#     PLV_beta_flat                -> n_ch*(n_ch-1)/2
#     wPLI_alpha_flat              -> n_ch*(n_ch-1)/2
#     wPLI_beta_flat               -> n_ch*(n_ch-1)/2
#     entropy                      -> n_ch ]
#
# Per paradigm/step, subjects are pooled (rows stacked) into ONE matrix:
#   results_batch/stage3/feature_matrices/<paradigm>/<paradigm>_step-<step>.npy
#       shape: (n_trials_pooled, n_features)
#   results_batch/stage3/feature_matrices/<paradigm>/<paradigm>_step-<step>_meta.csv
#       columns: row_index, subject, epoch_index_in_subject
#   results_batch/stage3/feature_matrices/<paradigm>/<paradigm>_step-<step>_shape.json
#       documents n_trials_pooled, n_features, feature block boundaries
#
# Any subject missing one of the required feature files for a given
# paradigm/step is SKIPPED for that step (logged), rather than aborting the
# whole assembly — this keeps Stage 1 ML input well-formed even if 3.3-3.6
# haven't finished for every subject yet.
#
# SAFE TO RUN: only READS from results_batch/stage3/{subsample_indices,
# features}/. Writes only under results_batch/stage3/feature_matrices/.

from __future__ import annotations

import csv
import json

import numpy as np

import config3 as C
from stage3_utils import ensure_all_output_dirs, get_logger, list_paradigms

log = get_logger("phase37_assemble")

CONN_BANDS = list(C.CONNECTIVITY_BANDS.keys())  # e.g. ["alpha", "beta"]


def _load_subject_feature_blocks(dataset: str, file_key: str):
    """
    Loads all required feature arrays for ONE exact epoch file, identified by
    file_key = '<subject>_<task>_<file_id>' (matches the naming used by
    features_band_power.py / features_plv.py / features_wpli.py /
    features_entropy.py). Returns dict of name -> ndarray, or None if
    anything required is missing.
    """
    bp_path = C.BANDPOWER_DIR / dataset / f"{file_key}_bandpower.npy"
    ent_path = C.ENTROPY_DIR / dataset / f"{file_key}_entropy.npy"

    if not (bp_path.exists() and ent_path.exists()):
        return None

    blocks = {}
    bp = np.load(bp_path)  # (n_epochs, n_ch, n_bands)
    blocks["band_power"] = bp.reshape(bp.shape[0], -1)
    blocks["entropy"] = np.load(ent_path)  # (n_epochs, n_ch)

    n_epochs = bp.shape[0]

    for band in CONN_BANDS:
        plv_path = C.PLV_DIR / dataset / f"{file_key}_{band}_plv_flat.npy"
        wpli_path = C.WPLI_DIR / dataset / f"{file_key}_{band}_wpli_flat.npy"
        if not (plv_path.exists() and wpli_path.exists()):
            return None
        plv = np.load(plv_path)
        wpli = np.load(wpli_path)
        if plv.shape[0] < n_epochs or wpli.shape[0] < n_epochs:
            # connectivity may have been capped by MAX_EPOCHS_FOR_CONNECTIVITY
            n_epochs = min(n_epochs, plv.shape[0], wpli.shape[0])
        blocks[f"plv_{band}"] = plv
        blocks[f"wpli_{band}"] = wpli

    # Trim every block to the common n_epochs (in case connectivity was capped)
    for k in blocks:
        blocks[k] = blocks[k][:n_epochs]

    return blocks, n_epochs


def _feature_vector_for_epoch(blocks: dict, idx: int) -> np.ndarray:
    parts = [blocks["band_power"][idx]]
    for band in CONN_BANDS:
        parts.append(blocks[f"plv_{band}"][idx])
    for band in CONN_BANDS:
        parts.append(blocks[f"wpli_{band}"][idx])
    parts.append(blocks["entropy"][idx])
    return np.concatenate(parts)


def _feature_block_boundaries(blocks: dict) -> dict:
    boundaries = {}
    pos = 0
    order = ["band_power"] + [f"plv_{b}" for b in CONN_BANDS] + [f"wpli_{b}" for b in CONN_BANDS] + ["entropy"]
    for name in order:
        size = blocks[name].shape[1]
        boundaries[name] = [pos, pos + size]
        pos += size
    boundaries["_total_features"] = pos
    return boundaries


def assemble_paradigm_step(dataset: str, task: str, step) -> None:
    paradigm = f"{dataset}__{task}"
    idx_dir = C.SUBSAMPLE_IDX_DIR / dataset

    # Filenames are "<subject>_<task>_<file_id>_indices.json" — match any
    # file_id for this task.
    subject_indices_files = sorted(idx_dir.glob(f"*_{task}_*_indices.json"))
    if not subject_indices_files:
        log.warning(f"  [SKIP] {paradigm} step={step}: no subsample index files found "
                     f"(run subsample_trials.py first)")
        return

    rows_meta = []
    feature_rows = []
    block_boundaries = None

    for idx_path in subject_indices_files:
        with idx_path.open() as f:
            record = json.load(f)
        subject = record["subject"]
        file_id = record.get("file_id", "")
        file_key = f"{subject}_{task}_{file_id}" if file_id else f"{subject}_{task}"

        if record.get("task") != task:
            continue  # defensive: glob can over-match if task names are substrings of each other

        if not record["step_integrity"].get(str(step), False):
            continue  # not enough trials for this step for this subject/file

        epoch_indices = record["steps"][str(step)]
        if not epoch_indices:
            continue

        loaded = _load_subject_feature_blocks(dataset, file_key)
        if loaded is None:
            log.warning(f"  [SKIP file] {paradigm} {file_key} step={step}: "
                         f"missing band_power/entropy/plv/wpli outputs")
            continue
        blocks, n_epochs_available = loaded

        valid_epoch_indices = [i for i in epoch_indices if i < n_epochs_available]
        if not valid_epoch_indices:
            continue

        if block_boundaries is None:
            block_boundaries = _feature_block_boundaries(blocks)

        for i in valid_epoch_indices:
            feature_rows.append(_feature_vector_for_epoch(blocks, i))
            rows_meta.append({
                "row_index": len(feature_rows) - 1,
                "subject": subject,
                "file_id": file_id,
                "epoch_index_in_subject": i,
            })

    if not feature_rows:
        log.warning(f"  [SKIP] {paradigm} step={step}: no usable rows assembled")
        return

    matrix = np.vstack(feature_rows)

    out_dir = C.FEATURE_MATRIX_DIR / paradigm
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{paradigm}_step-{step}"

    out_npy = out_dir / f"{base}.npy"
    n_nan = int(np.isnan(matrix).sum())
    n_inf = int(np.isinf(matrix).sum())
    if n_nan or n_inf:
        log.warning(f"  [NaN/Inf] {base}: nan={n_nan} inf={n_inf}")
    np.save(out_npy, matrix)

    with (out_dir / f"{base}_meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "subject", "file_id", "epoch_index_in_subject"])
        writer.writeheader()
        writer.writerows(rows_meta)

    shape_doc = {
        "paradigm": paradigm,
        "step": step,
        "n_trials_pooled": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_subjects_contributing": len({r["subject"] for r in rows_meta}),
        "feature_block_boundaries": block_boundaries,
        "nan_count": n_nan,
        "inf_count": n_inf,
    }
    with (out_dir / f"{base}_shape.json").open("w") as f:
        json.dump(shape_doc, f, indent=2)

    log.info(f"[OK] {base}: matrix shape={matrix.shape} "
              f"subjects={shape_doc['n_subjects_contributing']} nan={n_nan} inf={n_inf}")


def run_assembly() -> None:
    ensure_all_output_dirs()
    paradigms = list_paradigms()
    if not paradigms:
        log.warning("No paradigms found under results_batch/ — nothing to assemble.")
        return

    for paradigm in paradigms:
        dataset, task = paradigm.split("__", 1)
        for step in C.SUBSAMPLE_STEPS:
            try:
                assemble_paradigm_step(dataset, task, step)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[ERROR] {paradigm} step={step}: {exc}")

    log.info("=" * 60)
    log.info(f"Phase 3.7 feature assembly complete for {len(paradigms)} paradigms.")
    log.info(f"Output: {C.FEATURE_MATRIX_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_assembly()

