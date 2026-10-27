#!/usr/bin/env python3

"""
Stage 1 Classical ML
====================

Models:
    - Random Forest
    - XGBoost
    - SVM

Evaluation:
    - Leave-One-Subject-Out cross-validation
    - Trial-count subsampling grid
    - Accuracy
    - Balanced accuracy
    - Macro F1
    - AUC
    - NFS where valid

Project root:
    /lp-dev/23BRS1328/erp_trial_min

Example:
    cd /lp-dev/23BRS1328/erp_trial_min

    nohup python3 stage4_classical_ml/run_stage1_classical_ml.py \
        --shard-index 0 \
        --num-shards 4 \
        > logs_shard0.log 2>&1 &

The script is resumable. A completed
(model, paradigm, step) combination receives a marker file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Project root does not exist: {PROJECT_ROOT}")

sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Thread limits
#
# Four shards are intended. Each process gets six CPU threads.
# This avoids four processes fighting over all 24 CPU cores.
# ---------------------------------------------------------------------

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from stage4_classical_ml.dataset import (
    ParadigmData,
    STEPS,
    get_paradigms,
)
from stage4_classical_ml.models2 import (
    RFModel,
    XGBModel,
    SVMModel,
)
from stage4_classical_ml.metrics import compute_metrics
from stage4_classical_ml.nfs import (
    load_reference_topography,
    importance_to_topography,
    bootstrap_nfs,
)

# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------

OUT_ROOT = PROJECT_ROOT / "results_stage4_gpu"

MODEL_DIRS = {
    "random_forest": OUT_ROOT / "random_forest",
    "xgboost": OUT_ROOT / "xgboost",
    "svm": OUT_ROOT / "svm",
}

NFS_DIR = OUT_ROOT / "nfs"
MARKER_DIR = OUT_ROOT / "markers"
ERROR_DIR = OUT_ROOT / "errors"

for directory in [
    *MODEL_DIRS.values(),
    NFS_DIR,
    MARKER_DIR,
    ERROR_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

MODEL_BUILDERS = {
    "random_forest": RFModel,
    "xgboost": XGBModel,
    "svm": SVMModel,
}

# ---------------------------------------------------------------------
# NFS reference mapping
#
# LRP has no reference row in NFS_benchmarks.csv, so it is classified
# normally but NFS is recorded as N/A.
# ---------------------------------------------------------------------

PARADIGM_TO_NFS_NAME = {
    "ERP_CORE__P3": "P3",
    "ERP_CORE__MMN": "MMN",
    "ERP_CORE__N2pc": "N2pc",
    "ERP_CORE__N400": "N400",
    "ERP_CORE__N170": "N170",
    "ERP_CORE__ERN": "ERN",
}

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def safe_step_name(step) -> str:
    """Convert a step to a filename-safe string."""
    return str(step).replace("/", "_").replace(" ", "_")


def marker_path(model_name: str, paradigm: str, step) -> Path:
    step_name = safe_step_name(step)
    return MARKER_DIR / (
        f"{model_name}__{paradigm}__step{step_name}.done"
    )


def error_path(model_name: str, paradigm: str, step) -> Path:
    step_name = safe_step_name(step)
    return ERROR_DIR / (
        f"{model_name}__{paradigm}__step{step_name}.error.txt"
    )


def atomic_touch(path: Path) -> None:
    """
    Create a marker atomically.

    The marker is created only after the current model/paradigm/step
    completed successfully or was intentionally recorded as no-data.
    """
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    temporary.write_text(
        f"completed_by_pid={os.getpid()}\n"
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    os.replace(temporary, path)


def append_row(
    csv_path: Path,
    row: dict,
    key_cols: list[str],
) -> None:
    """
    Upsert one row into a CSV.

    The row identified by key_cols is replaced if it already exists.
    This keeps resumed runs from duplicating summary rows.
    """
    row_df = pd.DataFrame([row])

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(csv_path)

            mask = pd.Series(True, index=existing.index)

            for key in key_cols:
                if key not in existing.columns:
                    mask &= False
                else:
                    mask &= (
                        existing[key].astype(str)
                        == str(row.get(key, ""))
                    )

            existing = existing.loc[~mask]
            combined = pd.concat(
                [existing, row_df],
                ignore_index=True,
                sort=False,
            )
        except Exception:
            backup = csv_path.with_suffix(".corrupt_backup.csv")
            csv_path.replace(backup)
            combined = row_df
    else:
        combined = row_df

    temporary = csv_path.with_suffix(
        csv_path.suffix + f".tmp.{os.getpid()}"
    )

    combined.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)


def write_error(
    model_name: str,
    paradigm: str,
    step,
    exc: Exception,
) -> None:
    path = error_path(model_name, paradigm, step)

    text = (
        f"model={model_name}\n"
        f"paradigm={paradigm}\n"
        f"step={step}\n"
        f"exception={repr(exc)}\n\n"
        f"{traceback.format_exc()}\n"
    )

    path.write_text(text)


def get_nfs_reference_channel_count() -> int | None:
    """
    Read the number of reference channels from NFS_benchmarks.csv.

    Returns None if the file is unavailable or malformed.
    """
    nfs_path = PROJECT_ROOT / "data" / "benchmarks" / "NFS_benchmarks.csv"

    if not nfs_path.exists():
        print(f"[NFS] Reference file not found: {nfs_path}")
        return None

    try:
        df = pd.read_csv(nfs_path)

        if df.shape[1] < 2:
            print("[NFS] Reference file has no channel columns.")
            return None

        return int(df.shape[1] - 1)

    except Exception as exc:
        print(f"[NFS] Could not read reference file: {exc}")
        return None


# ---------------------------------------------------------------------
# NFS handling
# ---------------------------------------------------------------------


def write_nfs_na(
    nfs_csv: Path,
    model_name: str,
    paradigm: str,
    step,
    note: str,
) -> None:
    """Write an explicit N/A NFS row."""
    append_row(
        nfs_csv,
        {
            "dataset_task": paradigm,
            "model": model_name,
            "step": step,
            "nfs": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "note": note,
        },
        key_cols=["dataset_task", "model", "step"],
    )


def calculate_and_save_nfs(
    model_name: str,
    paradigm: str,
    step,
    data: ParadigmData,
    all_importances: list[np.ndarray],
) -> None:
    """
    Compute NFS when possible.

    NFS is skipped safely when:
    - No reference topography exists.
    - Model has no feature importance.
    - Feature/reference channel counts differ.
    - Feature block cannot be interpreted.
    """
    nfs_csv = NFS_DIR / f"{model_name}_nfs.csv"
    reference_name = PARADIGM_TO_NFS_NAME.get(paradigm)

    if reference_name is None:
        write_nfs_na(
            nfs_csv=nfs_csv,
            model_name=model_name,
            paradigm=paradigm,
            step=step,
            note="N/A: no NFS reference topography exists for this paradigm",
        )
        return

    if not all_importances:
        write_nfs_na(
            nfs_csv=nfs_csv,
            model_name=model_name,
            paradigm=paradigm,
            step=step,
            note="N/A: model does not provide usable feature importance",
        )
        return

    reference = load_reference_topography(reference_name)

    if reference is None:
        write_nfs_na(
            nfs_csv=nfs_csv,
            model_name=model_name,
            paradigm=paradigm,
            step=step,
            note="N/A: reference topography could not be loaded",
        )
        return

    try:
        topo_ref, reference_channels = reference

        mean_importance = np.mean(
            np.stack(all_importances, axis=0),
            axis=0,
        )

        # IMPORTANT:
        # The corrected nfs.py returns:
        #     topo_model, n_channels_actual
        #
        # It must not use the NFS reference channel count to infer feature
        # channel count.
        topo_result = importance_to_topography(
            mean_importance,
            data.feature_blocks,
        )

        if isinstance(topo_result, tuple):
            topo_model, n_channels_actual = topo_result
        else:
            # Compatibility with an older nfs.py implementation.
            topo_model = topo_result
            n_channels_actual = len(topo_model)

        n_channels_reference = len(topo_ref)

        if n_channels_actual != n_channels_reference:
            write_nfs_na(
                nfs_csv=nfs_csv,
                model_name=model_name,
                paradigm=paradigm,
                step=step,
                note=(
                    "N/A: feature/reference channel-count mismatch; "
                    f"features={n_channels_actual}, "
                    f"reference={n_channels_reference}"
                ),
            )
            return

        point, ci_low, ci_high = bootstrap_nfs(
            topo_model,
            topo_ref,
            n_iter=1000,
            batch_size=200,
            seed=42,
        )

        append_row(
            nfs_csv,
            {
                "dataset_task": paradigm,
                "model": model_name,
                "step": step,
                "nfs": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "note": "",
            },
            key_cols=["dataset_task", "model", "step"],
        )

    except Exception as exc:
        write_nfs_na(
            nfs_csv=nfs_csv,
            model_name=model_name,
            paradigm=paradigm,
            step=step,
            note=f"N/A: NFS calculation failed safely: {repr(exc)}",
        )

        print(
            f"[NFS WARNING] {model_name} / {paradigm} / step={step}: "
            f"{repr(exc)}"
        )


# ---------------------------------------------------------------------
# One model/paradigm/step
# ---------------------------------------------------------------------


def run_one(
    model_name: str,
    paradigm: str,
    step,
    data: ParadigmData,
) -> dict | None:
    """
    Run one model on one paradigm and one trial-count step.
    """
    marker = marker_path(model_name, paradigm, step)

    if marker.exists():
        return None

    print(
        f"[START] model={model_name} "
        f"paradigm={paradigm} "
        f"step={step}",
        flush=True,
    )

    fold_rows: list[dict] = []
    all_importances: list[np.ndarray] = []

    subjects = data.subjects()

    for fold_number, test_subject in enumerate(subjects, start=1):
        train_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        train_rows: list[int] = []

        for train_subject in train_subjects:
            if str(step) == "full":
                rows = data.all_row_indices_for_subject(
                    train_subject
                )
            else:
                rows = data.row_indices_for_subject_step(
                    train_subject,
                    str(step),
                )

            train_rows.extend(rows.tolist())

        # Test data remains the full held-out subject.
        test_rows = data.all_row_indices_for_subject(test_subject)

        if len(train_rows) == 0 or len(test_rows) == 0:
            print(
                f"[SKIP FOLD] {paradigm} step={step} "
                f"subject={test_subject}: empty train/test",
                flush=True,
            )
            continue

        X_train = data.X[train_rows]
        y_train = data.y[train_rows]
        X_test = data.X[test_rows]
        y_test = data.y[test_rows]

        if len(np.unique(y_train)) < 2:
            print(
                f"[SKIP FOLD] {paradigm} step={step} "
                f"subject={test_subject}: "
                f"training data contains one class",
                flush=True,
            )
            continue

        try:
            model = MODEL_BUILDERS[model_name]()

            if threadpool_limits is not None:
                with threadpool_limits(limits=6):
                    y_pred, y_score = model.train_predict(
                        X_train,
                        y_train,
                        X_test,
                    )
            else:
                y_pred, y_score = model.train_predict(
                    X_train,
                    y_train,
                    X_test,
                )

            metrics = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_score,
            )

            metrics["subject"] = test_subject
            metrics["fold_number"] = fold_number
            metrics["n_train_trials"] = len(train_rows)
            metrics["n_test_trials"] = len(test_rows)

            fold_rows.append(metrics)

            importance = model.feature_importance()

            if importance is not None:
                importance = np.asarray(importance)

                if importance.ndim == 1:
                    all_importances.append(importance)

        except Exception as exc:
            print(
                f"[FOLD ERROR] model={model_name} "
                f"paradigm={paradigm} "
                f"step={step} "
                f"subject={test_subject}: {repr(exc)}",
                flush=True,
            )

            write_error(
                model_name=model_name,
                paradigm=paradigm,
                step=f"{step}__subject_{test_subject}",
                exc=exc,
            )

    if not fold_rows:
        print(
            f"[NO DATA] model={model_name} "
            f"paradigm={paradigm} step={step}",
            flush=True,
        )

        atomic_touch(marker)
        return None

    fold_df = pd.DataFrame(fold_rows)

    out_dir = MODEL_DIRS[model_name] / paradigm
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = out_dir / f"step_{safe_step_name(step)}_folds.csv"
    fold_df.to_csv(fold_csv, index=False)

    if "degenerate_fold" in fold_df.columns:
        non_degenerate = fold_df.loc[
            ~fold_df["degenerate_fold"].astype(bool)
        ]
        n_degenerate = int(
            fold_df["degenerate_fold"].astype(bool).sum()
        )
    else:
        non_degenerate = fold_df
        n_degenerate = 0

    summary = {
        "dataset_task": paradigm,
        "model": model_name,
        "step": step,
        "n_folds": int(len(fold_df)),
        "n_degenerate_folds": n_degenerate,
        "accuracy_mean": float(
            fold_df["accuracy"].mean()
        ),
        "balanced_accuracy_mean": float(
            non_degenerate["balanced_accuracy"].mean()
        )
        if len(non_degenerate)
        else np.nan,
        "macro_f1_mean": float(
            fold_df["macro_f1"].mean()
        ),
        "auc_mean": float(
            non_degenerate["auc"].mean()
        )
        if len(non_degenerate)
        else np.nan,
    }

    summary_csv = out_dir / f"{paradigm}_summary.csv"

    append_row(
        summary_csv,
        summary,
        key_cols=["dataset_task", "model", "step"],
    )

    calculate_and_save_nfs(
        model_name=model_name,
        paradigm=paradigm,
        step=step,
        data=data,
        all_importances=all_importances,
    )

    atomic_touch(marker)

    print(
        f"[DONE] model={model_name} "
        f"paradigm={paradigm} "
        f"step={step} "
        f"accuracy={summary['accuracy_mean']:.4f} "
        f"balanced_accuracy={summary['balanced_accuracy_mean']:.4f} "
        f"degenerate={n_degenerate}/{len(fold_df)}",
        flush=True,
    )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable Stage 1 classical ML."
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index, from 0 to num-shards-1.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards.",
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            "--shard-index must satisfy "
            "0 <= shard-index < num-shards"
        )

    all_paradigms = get_paradigms()

    paradigms = [
        paradigm
        for index, paradigm in enumerate(all_paradigms)
        if index % args.num_shards == args.shard_index
    ]

    print("=" * 80, flush=True)
    print("Stage 1 Classical ML", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Total paradigms: {len(all_paradigms)}", flush=True)
    print(
        f"Shard: {args.shard_index}/{args.num_shards}",
        flush=True,
    )
    print(
        f"Assigned paradigms: {len(paradigms)}",
        flush=True,
    )
    print("=" * 80, flush=True)

    for paradigm in paradigms:
        print(
            f"\n=== {paradigm} ===",
            flush=True,
        )

        try:
            data = ParadigmData(paradigm)

        except Exception as exc:
            print(
                f"[PARADIGM LOAD ERROR] {paradigm}: {repr(exc)}",
                flush=True,
            )

            path = ERROR_DIR / f"load__{paradigm}.error.txt"
            path.write_text(
                f"{repr(exc)}\n\n{traceback.format_exc()}"
            )
            continue

        for model_name in MODEL_BUILDERS:
            for step in STEPS:
                if marker_path(model_name, paradigm, step).exists():
                    print(
                        f"[EXISTS] {model_name} "
                        f"{paradigm} step={step}",
                        flush=True,
                    )
                    continue

                start_time = time.time()

                try:
                    run_one(
                        model_name=model_name,
                        paradigm=paradigm,
                        step=step,
                        data=data,
                    )

                except Exception as exc:
                    print(
                        f"[COMBINATION ERROR] "
                        f"{model_name} / {paradigm} / {step}: "
                        f"{repr(exc)}",
                        flush=True,
                    )

                    write_error(
                        model_name=model_name,
                        paradigm=paradigm,
                        step=step,
                        exc=exc,
                    )

                    # Do not create a .done marker after an unexpected
                    # combination-level failure. It can be retried later.

                elapsed = time.time() - start_time

                print(
                    f"[TIME] {model_name} / {paradigm} "
                    f"step={step}: {elapsed:.1f}s",
                    flush=True,
                )

    print(
        f"\nShard {args.shard_index} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()