#!/bin/bash
# setup_stage3.sh — one-shot installer for the Stage 3 pipeline (FIXED v2).
# Fixes a naming collision bug where epoch files sharing the same
# subject/task (e.g. per-condition split files) could overwrite each
# other's feature output, causing IndexError crashes in extract_features.py.
# Run from your project root:
#   cd /lp-dev/23BRS1328/erp_trial_min
#   bash setup_stage3.sh
set -e
mkdir -p stage3_feature_extraction

cat > stage3_feature_extraction/README.md << 'STAGE3PYEOF_README_MD'
# Stage 3 — Trial Subsampling Grid & Feature Extraction

Completely separate from `preprocessing_batch/` (Phase 2, frozen/submitted).
Nothing here imports from or writes to `preprocessing_batch/` or any
`results_batch/<DATASET>/` Phase 2 output — Stage 3 only **reads** the
`*-epo.fif` files Phase 2 produced. ALL Stage 3 output goes into its own
top-level folder, `stage3_results/`, sitting next to (not inside)
`results_batch/`.

## Install location

```
/lp-dev/23BRS1328/erp_trial_min/
    preprocessing_batch/        <- Phase 2 (untouched)
    results_batch/               <- Phase 2 output (read-only)
        ERP_CORE/ DS006018/ DS002680/
    stage3_feature_extraction/  <- Stage 3 scripts (this folder)
    stage3_results/               <- ALL Stage 3 output (new, separate folder)
```

## Run order

```bash
cd /lp-dev/23BRS1328/erp_trial_min
python stage3_feature_extraction/run_stage3_all.py     # 3.1 -> 3.7
python stage3_feature_extraction/extract_features.py   # full-trial ML baseline X/y
```

or run each sub-step individually (each is independently resumable —
already-completed files are skipped unless you pass `overwrite=True`):

| Spec ID | Script                      | What it does |
|---------|------------------------------|--------------|
| 3.1 | `subsample_trials.py`        | Builds the subsampling grid (5,10,20,30,50,75,100,150,200,full) per subject/paradigm |
| 3.2 | `bootstrap_grid.py`          | 1000-iteration bootstrap (with replacement) per step per subject, mean ± 95% CI |
| 3.3 | `features_band_power.py`     | Relative band power (delta/theta/alpha/beta/gamma) per electrode, z-scored |
| 3.4 | `features_plv.py`            | PLV connectivity matrix per epoch, per band (alpha, beta) |
| 3.5 | `features_wpli.py`           | wPLI connectivity matrix per epoch, per band + PLV-vs-wPLI comparison for 3 subjects |
| 3.6 | `features_entropy.py`        | Spectral entropy per channel per epoch (+ per-dataset aggregate) |
| 3.7 | `assemble_features.py`       | Combines 3.3–3.6 into one feature matrix per paradigm per step |
| —   | `extract_features.py`        | Full-trial (non-subsampled) X/y/metadata, ready for SVM/LDA/RF baseline |

`config3.py` holds every tunable parameter. `stage3_utils.py` and
`connectivity_core.py` are shared helper modules — they don't produce output
on their own.

## Output layout (all under `stage3_results/`, separate from `results_batch/`)

```
stage3_results/
    subsample_indices/<DATASET>/<subject>_<task>_indices.json
    bootstrap/<DATASET>_<task>_bootstrap.csv
    features/
        band_power/<DATASET>/<subject>_<task>_bandpower.npy (+ .json sidecar)
        plv/<DATASET>/<subject>_<task>_<band>_plv.npy (+ _flat.npy)
        wpli/<DATASET>/<subject>_<task>_<band>_wpli.npy (+ _flat.npy)
        entropy/<DATASET>/<subject>_<task>_entropy.npy (+ per-task aggregate)
    feature_matrices/<paradigm>/<paradigm>_step-<step>.npy (+ _meta.csv, _shape.json)
    ml/<paradigm>/{X.npy,y.npy,y_codes.npy,label_map.json,metadata.csv}
    ml/ALL_PARADIGMS_COMBINED/{...}
    audit/plv_vs_wpli_comparison.csv
    logs/phase3*.log
```

## Standard conventions used (finalized — these are the defaults now baked into the code)

1. **PLV / wPLI "per epoch."** Computed within each individual epoch (band-pass
   filter -> Hilbert transform -> phase/cross-spectrum across the epoch's own
   time samples), giving one matrix per epoch with shape `(n_epochs, n_ch, n_ch)`
   as required. This is a single-trial connectivity estimate — a different
   quantity from the textbook across-trials PLV, but it's the standard this
   pipeline now uses consistently everywhere (3.4, 3.5, 3.7, and
   `extract_features.py` all use the same definition).
2. **Bootstrap metric.** Mean Global Field Power (GFP) of the bootstrap-resampled
   evoked response — paradigm-agnostic, works for every task. This is the
   fixed default (`METRIC_FN` in `bootstrap_grid.py`).
3. **Step > available trials.** 3.1 (without replacement) falls back to "all
   available trials" and flags it `False` in `step_integrity`. 3.2 (bootstrap,
   with replacement) still runs at the requested step size regardless — this
   is standard bootstrap practice. `feature_matrices/` (3.7) only uses
   trials where `step_integrity` is `True`, so under-powered steps are
   excluded automatically rather than padded or duplicated.
4. **Labels.** `extract_features.py` reads the label for every epoch straight
   from that epoch file's own `event_id` mapping — i.e. whatever condition
   Phase 2's epoching already assigned per `event_registry.py` (target/standard,
   correct/error, etc.). No relabeling logic is added.
5. **Feature vector composition** (used identically by `assemble_features.py`
   and `extract_features.py`):
   `[ band_power.flatten(), PLV_alpha_flat, PLV_beta_flat, wPLI_alpha_flat, wPLI_beta_flat, entropy ]`

These are now the project's standing conventions for Stage 3 — no further
sign-off needed to run the pipeline end-to-end. If your supervisor later
wants a different connectivity definition or bootstrap metric, only
`connectivity_core.py` / `bootstrap_grid.py` need to change — every
downstream script picks it up automatically.

STAGE3PYEOF_README_MD

cat > stage3_feature_extraction/assemble_features.py << 'STAGE3PYEOF_ASSEMBLE_FEATURES_PY'
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

STAGE3PYEOF_ASSEMBLE_FEATURES_PY

cat > stage3_feature_extraction/bootstrap_grid.py << 'STAGE3PYEOF_BOOTSTRAP_GRID_PY'
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
    Draw n_iter bootstrap samples (with replacement) of size `step` from
    epochs_data (n_trials, n_channels, n_times), compute metric_fn on each
    draw, and return the array of n_iter metric values.

    step == "full" -> resample size == n_trials (still WITH replacement,
    standard bootstrap convention).
    """
    n_trials = epochs_data.shape[0]
    # With-replacement bootstrap: draw_size = n_trials for "full", else the
    # requested step (even if step > n_trials — that's a valid, if noisier,
    # bootstrap estimate; 3.1's step_integrity flag is what tells you a
    # subject didn't actually have enough *unique* trials for that step).
    draw_size = n_trials if step == "full" else int(step)

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

STAGE3PYEOF_BOOTSTRAP_GRID_PY

cat > stage3_feature_extraction/config3.py << 'STAGE3PYEOF_CONFIG3_PY'
# stage3_feature_extraction/config3.py
#
# Shared configuration for Phase 3 (Trial Subsampling Grid & Feature Extraction).
# Does NOT modify or import anything from preprocessing_batch/config.py — Phase 2
# is frozen/submitted. This is a standalone config for Stage 3 only.
#
# IMPORTANT: run everything from project root:
#   cd /lp-dev/23BRS1328/erp_trial_min
#   python stage3_feature_extraction/run_stage3_all.py

from pathlib import Path

# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(".").resolve()

# Phase 2 output stays exactly where it is — read-only from here.
RESULTS_ROOT = PROJECT_ROOT / "results_batch"

# Stage 3 output now lives in its OWN top-level folder, separate from
# results_batch/ entirely (per request: a dedicated Stage 3 results
# directory, not nested inside Phase 2's output tree).
STAGE3_OUT_ROOT = PROJECT_ROOT / "stage3_results"

SUBSAMPLE_IDX_DIR   = STAGE3_OUT_ROOT / "subsample_indices"
BOOTSTRAP_DIR       = STAGE3_OUT_ROOT / "bootstrap"
BANDPOWER_DIR       = STAGE3_OUT_ROOT / "features" / "band_power"
PLV_DIR             = STAGE3_OUT_ROOT / "features" / "plv"
WPLI_DIR            = STAGE3_OUT_ROOT / "features" / "wpli"
ENTROPY_DIR         = STAGE3_OUT_ROOT / "features" / "entropy"
FEATURE_MATRIX_DIR  = STAGE3_OUT_ROOT / "feature_matrices"
LOG_DIR             = STAGE3_OUT_ROOT / "logs"
AUDIT_DIR           = STAGE3_OUT_ROOT / "audit"
ML_DIR              = STAGE3_OUT_ROOT / "ml"

ALL_OUT_DIRS = [
    STAGE3_OUT_ROOT, SUBSAMPLE_IDX_DIR, BOOTSTRAP_DIR, BANDPOWER_DIR,
    PLV_DIR, WPLI_DIR, ENTROPY_DIR, FEATURE_MATRIX_DIR, LOG_DIR, AUDIT_DIR,
    ML_DIR,
]

# ==================================================
# Datasets / epoch discovery
# ==================================================

DATASETS = ["ERP_CORE", "DS006018", "DS002680"]

# Epoch files are expected at:
#   results_batch/<DATASET>/<subject>/<task>/epochs-epo.fif
# (per Phase 2 handoff). We glob defensively in case naming varies slightly.
EPOCH_GLOB_PATTERNS = ["*-epo.fif", "epochs-epo.fif"]

# ==================================================
# 3.1 — Trial subsampling grid
# ==================================================

SUBSAMPLE_STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"]
RANDOM_SEED = 42

# ==================================================
# 3.2 — Bootstrap wrapper
# ==================================================

N_BOOTSTRAP_ITERS = 1000
CI_LEVEL = 0.95  # 95% CI

# ==================================================
# 3.3 — Band power
# ==================================================

FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
TOTAL_BAND_RANGE = (1.0, 40.0)  # denominator for "relative" band power

# ==================================================
# 3.4 / 3.5 — Connectivity (PLV / wPLI)
# ==================================================

# Per the Phase 3 spec: alpha + beta are priority bands for connectivity.
CONNECTIVITY_BANDS = {
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
}

# Number of subjects to run the PLV-vs-wPLI documented comparison on (3.5).
WPLI_COMPARISON_N_SUBJECTS = 3

# ==================================================
# 3.6 — Spectral entropy
# ==================================================

ENTROPY_PSD_FMIN = 1.0
ENTROPY_PSD_FMAX = 40.0
ENTROPY_NPERSEG = 128  # Welch segment length (samples); auto-clamped to epoch length

# ==================================================
# Misc
# ==================================================

# Cap how many epochs are used for PLV/wPLI per file when doing connectivity
# (these are O(n_ch^2) per epoch and can get expensive on large files).
MAX_EPOCHS_FOR_CONNECTIVITY = None  # None = no cap; set an int to cap for a quick test run

STAGE3PYEOF_CONFIG3_PY

cat > stage3_feature_extraction/connectivity_core.py << 'STAGE3PYEOF_CONNECTIVITY_CORE_PY'
# stage3_feature_extraction/connectivity_core.py
#
# Shared math for 3.4 (PLV) and 3.5 (wPLI). Both are computed *per single
# epoch*, across the time samples within that epoch (after band-pass
# filtering + Hilbert transform), rather than across trials at a fixed
# latency. This matches the Phase 3 spec's required output shape of
# (n_epochs, n_ch, n_ch) — one connectivity matrix per epoch — rather than
# the more conventional single matrix-per-condition PLV.
#
# PLV(i,j)  = | mean_t( exp(1j * (phase_i(t) - phase_j(t))) ) |
# wPLI(i,j) = | mean_t( imag(Z) ) | / mean_t( |imag(Z)| )
#             where Z(t) = analytic_i(t) * conj(analytic_j(t))
# wPLI is insensitive to zero-phase-lag interactions, making it more robust
# to volume conduction than PLV (hence the 3.5 "robustness check" framing).

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _bandpass(data: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """data: (n_channels, n_times). Zero-phase Butterworth band-pass."""
    nyq = sfreq / 2.0
    low = max(fmin / nyq, 1e-6)
    high = min(fmax / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _analytic_phase(data: np.ndarray) -> np.ndarray:
    """data: (n_channels, n_times) -> instantaneous phase, same shape."""
    return np.angle(hilbert(data, axis=-1))


def _analytic_signal(data: np.ndarray) -> np.ndarray:
    return hilbert(data, axis=-1)


def plv_matrix(epoch_data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns PLV matrix (n_channels, n_channels), symmetric, diagonal = 1.
    """
    filtered = _bandpass(epoch_data, sfreq, fmin, fmax)
    phase = _analytic_phase(filtered)  # (n_ch, n_times)
    n_ch = phase.shape[0]

    # exp(i*phase) for all channels -> (n_ch, n_times)
    z = np.exp(1j * phase)
    # PLV(i,j) = |mean_t( z_i * conj(z_j) )|
    cross = z @ z.conj().T / phase.shape[1]  # (n_ch, n_ch)
    plv = np.abs(cross)
    np.fill_diagonal(plv, 1.0)
    return plv


def wpli_matrix(epoch_data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns wPLI matrix (n_channels, n_channels), symmetric, diagonal = 0
    (self-connectivity is undefined/zero-imaginary by construction for wPLI).
    """
    filtered = _bandpass(epoch_data, sfreq, fmin, fmax)
    analytic = _analytic_signal(filtered)  # (n_ch, n_times), complex
    n_ch, n_times = analytic.shape

    wpli = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_ch):
        cross_i = analytic[i][None, :] * np.conj(analytic)  # (n_ch, n_times)
        imag = cross_i.imag
        numerator = np.abs(imag.mean(axis=-1))
        denominator = np.abs(imag).mean(axis=-1)
        denominator_safe = np.where(denominator < 1e-15, np.nan, denominator)
        wpli[i, :] = numerator / denominator_safe
    np.fill_diagonal(wpli, 0.0)
    wpli = np.nan_to_num(wpli, nan=0.0)
    return wpli


def upper_triangle_flat(mat: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle (excludes diagonal) of a square matrix."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]

STAGE3PYEOF_CONNECTIVITY_CORE_PY

cat > stage3_feature_extraction/extract_features.py << 'STAGE3PYEOF_EXTRACT_FEATURES_PY'
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

STAGE3PYEOF_EXTRACT_FEATURES_PY

cat > stage3_feature_extraction/features_band_power.py << 'STAGE3PYEOF_FEATURES_BAND_POWER_PY'
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

STAGE3PYEOF_FEATURES_BAND_POWER_PY

cat > stage3_feature_extraction/features_entropy.py << 'STAGE3PYEOF_FEATURES_ENTROPY_PY'
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

STAGE3PYEOF_FEATURES_ENTROPY_PY

cat > stage3_feature_extraction/features_plv.py << 'STAGE3PYEOF_FEATURES_PLV_PY'
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

STAGE3PYEOF_FEATURES_PLV_PY

cat > stage3_feature_extraction/features_wpli.py << 'STAGE3PYEOF_FEATURES_WPLI_PY'
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

STAGE3PYEOF_FEATURES_WPLI_PY

cat > stage3_feature_extraction/run_stage3_all.py << 'STAGE3PYEOF_RUN_STAGE3_ALL_PY'
# stage3_feature_extraction/run_stage3_all.py
#
# Runs the full Phase 3 grid in spec order: 3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5
# -> 3.6 -> 3.7. Each step is independently re-runnable/resumable (skips
# work already done unless overwrite=True), matching the resume pattern
# used in Phase 2's run_*_all.py scripts.
#
# Run from project root:
#   cd /lp-dev/23BRS1328/erp_trial_min
#   python stage3_feature_extraction/run_stage3_all.py
#
# Does NOT touch preprocessing_batch/ or any Phase 2 output file — only
# reads results_batch/<DATASET>/.../*-epo.fif and writes under
# results_batch/stage3/.

from __future__ import annotations

from subsample_trials import run_subsampling
from bootstrap_grid import run_bootstrap_grid
from features_band_power import run_band_power
from features_plv import run_plv
from features_wpli import run_wpli
from features_entropy import run_entropy
from assemble_features import run_assembly

from stage3_utils import get_logger

log = get_logger("run_stage3_all")


def main(overwrite: bool = False) -> None:
    log.info("#" * 70)
    log.info("PHASE 3 — Trial Subsampling Grid & Feature Extraction")
    log.info("#" * 70)

    log.info(">>> 3.1 Subsampling Function")
    run_subsampling(overwrite=overwrite)

    log.info(">>> 3.2 Bootstrap Wrapper")
    run_bootstrap_grid(overwrite=overwrite)

    log.info(">>> 3.3 Band Power Features")
    run_band_power(overwrite=overwrite)

    log.info(">>> 3.4 PLV Matrices")
    run_plv(overwrite=overwrite)

    log.info(">>> 3.5 wPLI Matrices")
    run_wpli(overwrite=overwrite)

    log.info(">>> 3.6 Spectral Entropy")
    run_entropy(overwrite=overwrite)

    log.info(">>> 3.7 Feature Assembly")
    run_assembly()

    log.info("#" * 70)
    log.info("PHASE 3 COMPLETE. See results_batch/stage3/ for all outputs.")
    log.info("#" * 70)


if __name__ == "__main__":
    main(overwrite=False)

STAGE3PYEOF_RUN_STAGE3_ALL_PY

cat > stage3_feature_extraction/stage3_utils.py << 'STAGE3PYEOF_STAGE3_UTILS_PY'
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

STAGE3PYEOF_STAGE3_UTILS_PY

cat > stage3_feature_extraction/subsample_trials.py << 'STAGE3PYEOF_SUBSAMPLE_TRIALS_PY'
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

STAGE3PYEOF_SUBSAMPLE_TRIALS_PY

echo "Stage 3 files written to stage3_feature_extraction/ (collision-fix v2)"
