# stage3_feature_extraction/extract_features.py
#
# Builds the ML-ready X / y / metadata triple from the FULL (non-subsampled)
# trial set, for every subject and paradigm. This is the "Step 3.1" deliverable
# from the original Phase 3 handoff plan: one X.npy (n_epochs_total, n_features),
# one y.npy (n_epochs_total,) and one metadata.csv row-aligned with both.
#
# This is DIFFERENT from assemble_features.py (3.7): that script builds one
# matrix PER subsampling step for the trial-minimisation grid. This script
# builds ONE matrix using ALL available trials, for the initial ML baseline
# (SVM / LDA / Random Forest) before any trial reduction or GAN augmentation.
#
# PREREQUISITE: run these first (this script only reads their output, it does
# not recompute anything):
#   python stage3_feature_extraction/features_band_power.py
#   python stage3_feature_extraction/features_plv.py
#   python stage3_feature_extraction/features_wpli.py
#   python stage3_feature_extraction/features_entropy.py
#
# Label source: each epoch's label is read directly from the *-epo.fif file's
# own event_id mapping (epochs.event_id, inverted), i.e. whatever condition
# name Phase 2's epoching already assigned (e.g. "target"/"standard",
# "correct"/"error", per event_registry.py). No relabeling or reinterpretation
# is done here — labels are exactly what Phase 2 baked into the epochs.
#
# Output (one triple PER PARADIGM = dataset__task, plus one global combined
# triple across all paradigms):
#   results_batch/ml/<paradigm>/X.npy            (n_epochs, n_features)
#   results_batch/ml/<paradigm>/y.npy             (n_epochs,)            dtype=<U  (string labels)
#   results_batch/ml/<paradigm>/y_codes.npy       (n_epochs,)            dtype=int (label-encoded)
#   results_batch/ml/<paradigm>/label_map.json    {code: label_name}
#   results_batch/ml/<paradigm>/metadata.csv      row_index,dataset,subject,task,epoch_index_in_subject,label
#   results_batch/ml/<paradigm>/feature_block_boundaries.json
#
#   results_batch/ml/ALL_PARADIGMS_COMBINED/{X.npy,y.npy,y_codes.npy,label_map.json,metadata.csv}
#       (only meaningful if you intend to pool across paradigms — usually you
#       will train per-paradigm, since label sets differ by task)
#
# SAFE TO RUN: reads results_batch/<DATASET>/.../*-epo.fif and
# results_batch/stage3/features/{band_power,plv,wpli,entropy}/. Writes only
# under results_batch/ml/.

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import mne

import config3 as C
from stage3_utils import (
    EpochFileRef, get_logger, iter_epoch_files, list_paradigms,
)

log = get_logger("extract_features")

ML_ROOT = C.ML_DIR
CONN_BANDS = list(C.CONNECTIVITY_BANDS.keys())  # e.g. ["alpha", "beta"]


# ==================================================
# Per-subject feature loading (same blocks as 3.7, full trial set)
# ==================================================

def _load_subject_feature_blocks(dataset: str, file_key: str):
    bp_path = C.BANDPOWER_DIR / dataset / f"{file_key}_bandpower.npy"
    ent_path = C.ENTROPY_DIR / dataset / f"{file_key}_entropy.npy"
    if not (bp_path.exists() and ent_path.exists()):
        return None

    bp = np.load(bp_path)            # (n_epochs, n_ch, n_bands)
    blocks = {
        "band_power": bp.reshape(bp.shape[0], -1),
        "entropy": np.load(ent_path),  # (n_epochs, n_ch)
    }
    n_epochs = bp.shape[0]

    for band in CONN_BANDS:
        plv_path = C.PLV_DIR / dataset / f"{file_key}_{band}_plv_flat.npy"
        wpli_path = C.WPLI_DIR / dataset / f"{file_key}_{band}_wpli_flat.npy"
        if not (plv_path.exists() and wpli_path.exists()):
            return None
        plv = np.load(plv_path)
        wpli = np.load(wpli_path)
        n_epochs = min(n_epochs, plv.shape[0], wpli.shape[0])
        blocks[f"plv_{band}"] = plv
        blocks[f"wpli_{band}"] = wpli

    for k in blocks:
        blocks[k] = blocks[k][:n_epochs]

    return blocks, n_epochs


def _feature_block_order() -> list[str]:
    return ["band_power"] + [f"plv_{b}" for b in CONN_BANDS] + [f"wpli_{b}" for b in CONN_BANDS] + ["entropy"]


def _feature_vector_for_epoch(blocks: dict, idx: int) -> np.ndarray:
    return np.concatenate([blocks[name][idx] for name in _feature_block_order()])


def _feature_block_boundaries(blocks: dict) -> dict:
    boundaries, pos = {}, 0
    for name in _feature_block_order():
        size = blocks[name].shape[1]
        boundaries[name] = [pos, pos + size]
        pos += size
    boundaries["_total_features"] = pos
    return boundaries


# ==================================================
# Label extraction directly from the epochs file
# ==================================================

def _labels_for_epochs(ref: EpochFileRef, n_epochs: int) -> list[str]:
    """
    Returns one label string per epoch, taken from the *-epo.fif file's own
    event_id mapping (Phase 2's condition assignment). If the file has no
    meaningful event_id (e.g. {"1": 1}), falls back to the task name as a
    single constant label and logs a warning.
    """
    epochs = mne.read_epochs(ref.path, preload=False, verbose="ERROR")
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    codes = epochs.events[:n_epochs, 2]

    if len(epochs.event_id) <= 1:
        log.warning(f"  [LABEL WARN] {ref.stem}: only one event_id present "
                     f"({epochs.event_id}) — labels will be uninformative for classification.")

    labels = [inv_event_id.get(c, str(c)) for c in codes]
    return labels


# ==================================================
# Per-paradigm assembly
# ==================================================

def build_paradigm(dataset: str, task: str) -> dict | None:
    paradigm = f"{dataset}__{task}"
    refs = [r for r in iter_epoch_files(dataset) if r.task == task]
    if not refs:
        return None

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    meta_rows: list[dict] = []
    block_boundaries = None

    for ref in refs:
        loaded = _load_subject_feature_blocks(dataset, ref.key)
        if loaded is None:
            log.warning(f"  [SKIP file] {paradigm} {ref.key}: "
                         f"missing band_power/entropy/plv/wpli output "
                         f"(run features_band_power.py / features_plv.py / "
                         f"features_wpli.py / features_entropy.py first)")
            continue
        blocks, n_epochs = loaded
        if n_epochs == 0:
            continue

        if block_boundaries is None:
            block_boundaries = _feature_block_boundaries(blocks)

        try:
            subj_labels = _labels_for_epochs(ref, n_epochs)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"  [SKIP file] {paradigm} {ref.key}: label read failed: {exc}")
            continue

        if len(subj_labels) < n_epochs:
            # Defensive guard: should not happen now that features are keyed
            # per exact file, but if a stale/partial cache slips through,
            # skip this file rather than crash the whole run.
            log.warning(f"  [SKIP file] {paradigm} {ref.key}: label count "
                         f"({len(subj_labels)}) < feature epoch count ({n_epochs}) — "
                         f"likely a stale cached feature file; delete it and re-run "
                         f"features_band_power.py / features_plv.py / features_wpli.py / "
                         f"features_entropy.py with overwrite=True for this file.")
            continue

        for i in range(n_epochs):
            feature_rows.append(_feature_vector_for_epoch(blocks, i))
            labels.append(subj_labels[i])
            meta_rows.append({
                "row_index": len(feature_rows) - 1,
                "dataset": dataset,
                "subject": ref.subject,
                "task": task,
                "file_id": ref.file_id,
                "epoch_index_in_subject": i,
                "label": subj_labels[i],
            })

    if not feature_rows:
        log.warning(f"[SKIP] {paradigm}: no usable subjects (run 3.3/3.4/3.5/3.6 first)")
        return None

    X = np.vstack(feature_rows)
    y = np.array(labels)

    unique_labels = sorted(set(labels))
    label_map = {i: lbl for i, lbl in enumerate(unique_labels)}
    label_to_code = {lbl: i for i, lbl in label_map.items()}
    y_codes = np.array([label_to_code[lbl] for lbl in labels], dtype=np.int64)

    n_nan, n_inf = int(np.isnan(X).sum()), int(np.isinf(X).sum())
    if n_nan or n_inf:
        log.warning(f"  [NaN/Inf] {paradigm}: nan={n_nan} inf={n_inf} in X")

    return {
        "paradigm": paradigm, "X": X, "y": y, "y_codes": y_codes,
        "label_map": label_map, "meta_rows": meta_rows,
        "block_boundaries": block_boundaries,
        "n_nan": n_nan, "n_inf": n_inf,
    }


def _write_paradigm_output(result: dict) -> None:
    out_dir = ML_ROOT / result["paradigm"]
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", result["X"])
    np.save(out_dir / "y.npy", result["y"])
    np.save(out_dir / "y_codes.npy", result["y_codes"])

    with (out_dir / "label_map.json").open("w") as f:
        json.dump(result["label_map"], f, indent=2)

    with (out_dir / "feature_block_boundaries.json").open("w") as f:
        json.dump(result["block_boundaries"], f, indent=2)

    with (out_dir / "metadata.csv").open("w", newline="") as f:
        fieldnames = ["row_index", "dataset", "subject", "task", "file_id", "epoch_index_in_subject", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["meta_rows"])

    log.info(
        f"[OK] {result['paradigm']}: X={result['X'].shape} y={result['y'].shape} "
        f"classes={list(result['label_map'].values())} "
        f"n_subjects={len({m['subject'] for m in result['meta_rows']})} "
        f"nan={result['n_nan']} inf={result['n_inf']} -> {out_dir}"
    )


# ==================================================
# Main
# ==================================================

def run_extract_features(combine_all: bool = True) -> None:
    ML_ROOT.mkdir(parents=True, exist_ok=True)
    paradigms = list_paradigms()
    if not paradigms:
        log.warning("No paradigms found under results_batch/ — nothing to extract.")
        return

    all_results = []
    for paradigm in paradigms:
        dataset, task = paradigm.split("__", 1)
        result = build_paradigm(dataset, task)
        if result is None:
            continue
        _write_paradigm_output(result)
        all_results.append(result)

    if combine_all and all_results:
        # Pool across paradigms too, in case a global cross-task baseline is
        # wanted (NOTE: label sets differ per task/dataset — interpret with care).
        X = np.vstack([r["X"] for r in all_results])
        y = np.concatenate([r["y"] for r in all_results])
        unique_labels = sorted(set(y.tolist()))
        label_map = {i: lbl for i, lbl in enumerate(unique_labels)}
        label_to_code = {lbl: i for i, lbl in label_map.items()}
        y_codes = np.array([label_to_code[lbl] for lbl in y], dtype=np.int64)

        meta_rows = []
        offset = 0
        for r in all_results:
            for m in r["meta_rows"]:
                row = dict(m)
                row["row_index"] = offset + m["row_index"]
                meta_rows.append(row)
            offset += r["X"].shape[0]

        out_dir = ML_ROOT / "ALL_PARADIGMS_COMBINED"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        np.save(out_dir / "y_codes.npy", y_codes)
        with (out_dir / "label_map.json").open("w") as f:
            json.dump(label_map, f, indent=2)
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            fieldnames = ["row_index", "dataset", "subject", "task", "file_id", "epoch_index_in_subject", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(meta_rows)
        log.info(f"[OK] ALL_PARADIGMS_COMBINED: X={X.shape} y={y.shape} -> {out_dir}")

    log.info("=" * 60)
    log.info(f"extract_features.py complete. {len(all_results)}/{len(paradigms)} paradigms produced output.")
    log.info(f"Output root: {ML_ROOT}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_extract_features(combine_all=True)

