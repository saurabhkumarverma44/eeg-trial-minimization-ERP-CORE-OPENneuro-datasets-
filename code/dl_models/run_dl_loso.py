#!/usr/bin/env python3
"""
Stage: CNN-1D / EEGNet, LOSO cross-validation
==============================================

Models:
    - CNN-1D (dl_models/cnn1d.py)
    - EEGNet (dl_models/eegnet.py)

Evaluation:
    - Leave-One-Subject-Out cross-validation (same subject ordering
      convention as classical ML, for fold-matched comparability)
    - Trial-count subsampling grid (same 10 steps as classical ML)
    - Accuracy, balanced accuracy, F1, AUC per fold

Data source: results_batch/ via dl_prep.explicit_epoch_loader.load_explicit()
             (raw, corrected, verified epoch data -- same source used for
             this project's Phase 2 bug-fix verification)

Resumable: a completed (model, paradigm, step) combination receives a
marker file, same convention as run_stage1_classical_ml.py.

Example:
    cd /lp-dev/23BRS1328/erp_trial_min
    nohup python3 dl_models/run_dl_loso.py \
        --shard-index 0 --num-shards 1 \
        > logs_dl_loso.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Project root does not exist: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from dl_prep.explicit_epoch_loader import load_explicit, CONDITION_SPEC
from dl_models.cnn1d import ERPConv1DNet
from dl_models.eegnet import EEGNet
from dl_models.trainer import Trainer, LabelSmoothingCrossEntropy
from dl_models.loso_eval import evaluate_fold

OUT_ROOT = PROJECT_ROOT / "results_dl"
MODEL_DIRS = {"cnn1d": OUT_ROOT / "cnn1d", "eegnet": OUT_ROOT / "eegnet"}
MARKER_DIR = OUT_ROOT / "markers"
ERROR_DIR = OUT_ROOT / "errors"

for d in [OUT_ROOT, MARKER_DIR, ERROR_DIR, *MODEL_DIRS.values()]:
    d.mkdir(parents=True, exist_ok=True)

STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"]

# (dataset, task) pairs -- derived from CONDITION_SPEC keys, same 22
# paradigms as classical ML, in the same "DATASET_task" -> (dataset, task)
# form load_explicit() expects.
def get_paradigms():
    # Match against KNOWN dataset names explicitly -- splitting on the
    # first underscore breaks for "ERP_CORE", which itself contains an
    # underscore (this caused all 7 ERP_CORE paradigms to silently fail
    # with "no subject folders found" once the run reached them).
    KNOWN_DATASETS = ["ERP_CORE", "DS006018", "DS002680"]
    pairs = []
    for key in CONDITION_SPEC:
        matched = False
        for ds in KNOWN_DATASETS:
            if key.startswith(ds + "_"):
                task = key[len(ds) + 1:]
                pairs.append((ds, task, key))
                matched = True
                break
        if not matched:
            raise ValueError(f"Paradigm key '{key}' doesn't match any known dataset prefix")
    return sorted(pairs, key=lambda x: x[2])


def marker_path(model_name, paradigm_key, step):
    return MARKER_DIR / f"{model_name}__{paradigm_key}__step{step}.done"


def build_model(model_name, in_channels, num_classes):
    if model_name == "cnn1d":
        return ERPConv1DNet(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "eegnet":
        return EEGNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(model_name)


def subsample_indices(n_total, step, rng):
    """Same subsampling convention as Phase 3.1: random, without replacement."""
    if step == "full" or step >= n_total:
        return np.arange(n_total)
    return rng.choice(n_total, size=step, replace=False)


def run_paradigm_step(model_name, dataset, task, paradigm_key, step, device):
    marker = marker_path(model_name, paradigm_key, step)
    if marker.exists():
        print(f"[EXISTS] {model_name} {paradigm_key} step={step}", flush=True)
        return

    t0 = time.time()
    try:
        X, y, subject_ids, ch_names, sfreq = load_explicit(
            PROJECT_ROOT / "results_batch", dataset, task
        )
    except Exception as exc:
        print(f"[ERROR] load_explicit failed for {paradigm_key}: {exc}", flush=True)
        (ERROR_DIR / f"{model_name}__{paradigm_key}__step{step}.txt").write_text(str(exc))
        return

    subjects = sorted(set(subject_ids))  # deterministic, matches classical ML convention
    in_channels = X.shape[1]
    num_classes = len(np.unique(y))

    rows = []
    n_degenerate = 0
    rng = np.random.default_rng(42)

    for subject in subjects:
        test_mask = subject_ids == subject
        train_mask = ~test_mask

        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Apply trial-count subsampling to the TRAIN set only (test set
        # always uses all of that subject's real trials, same convention
        # as classical ML's LOSO -- the "step" limits how much training
        # data the model sees, not how much it's tested on).
        idx = subsample_indices(len(X_train_full), step, rng)
        X_train, y_train = X_train_full[idx], y_train_full[idx]

        # Per-channel z-score normalization -- CRITICAL. Raw EEG is
        # ~1e-5 scale (real microvolt amplitudes), which is too small
        # for stable gradient-based learning with standard weight
        # initialization. Fit mean/std on the TRAIN subsample only,
        # apply identically to test -- no leakage from the held-out
        # subject into the normalization statistics.
        ch_mean = X_train.mean(axis=(0, 2), keepdims=True)
        ch_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        X_train = (X_train - ch_mean) / ch_std
        X_test = (X_test - ch_mean) / ch_std

        if len(np.unique(y_train)) < 2:
            n_degenerate += 1
            continue  # cannot train on a single-class fold

        # Simple internal train/val split of the training subjects'
        # trials, purely for early stopping -- NOT a held-out subject,
        # since the true held-out test is the LOSO subject itself.
        n_val = max(1, int(0.15 * len(X_train)))
        perm = rng.permutation(len(X_train))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        if len(np.unique(y_train[tr_idx])) < 2:
            n_degenerate += 1
            continue

        train_ds = TensorDataset(torch.from_numpy(X_train[tr_idx]).float(),
                                   torch.from_numpy(y_train[tr_idx]).long(),
                                   torch.zeros(len(tr_idx)))
        val_ds = TensorDataset(torch.from_numpy(X_train[val_idx]).float(),
                                 torch.from_numpy(y_train[val_idx]).long(),
                                 torch.zeros(len(val_idx)))
        test_ds = TensorDataset(torch.from_numpy(X_test).float(),
                                  torch.from_numpy(y_test).long(),
                                  torch.zeros(len(X_test)))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        model = build_model(model_name, in_channels, num_classes).to(device)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

        ckpt_path = MODEL_DIRS[model_name] / f"{paradigm_key}_step{step}_{subject}.pt"
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion,
                           scheduler=scheduler, device=device, save_path=ckpt_path,
                           patience=8, use_amp=False)
        trainer.fit(train_loader, val_loader, epochs=30)

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        fold_metrics = evaluate_fold(model, test_loader, device, use_amp=False)
        fold_metrics.update({"subject": subject, "n_train_trials": len(tr_idx)})
        rows.append(fold_metrics)

        ckpt_path.unlink(missing_ok=True)  # don't keep 220-paradigm x 40-subject checkpoints
        del model, trainer, optimizer, scheduler
        torch.cuda.empty_cache()

    if not rows:
        print(f"[NO DATA] {model_name} {paradigm_key} step={step}", flush=True)
        marker.touch()
        return

    out_dir = MODEL_DIRS[model_name] / paradigm_key
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    out_csv = out_dir / f"step_{step}_folds.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_bal_acc = np.nanmean([r["balanced_accuracy"] for r in rows])
    mean_acc = np.nanmean([r["accuracy"] for r in rows])
    elapsed = time.time() - t0
    print(f"[DONE] model={model_name} paradigm={paradigm_key} step={step} "
          f"accuracy={mean_acc:.4f} balanced_accuracy={mean_bal_acc:.4f} "
          f"degenerate={n_degenerate}/{len(subjects)}", flush=True)
    print(f"[TIME] {model_name} / {paradigm_key} step={step}: {elapsed:.1f}s", flush=True)

    marker.touch()


def main():
    parser = argparse.ArgumentParser(description="Run resumable CNN-1D/EEGNet LOSO training.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paradigms = get_paradigms()
    shard_paradigms = [p for i, p in enumerate(all_paradigms) if i % args.num_shards == args.shard_index]

    print("=" * 80, flush=True)
    print("CNN-1D / EEGNet LOSO Training", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Total paradigms: {len(all_paradigms)} | Shard: {args.shard_index}/{args.num_shards} | Assigned: {len(shard_paradigms)}", flush=True)
    print("=" * 80, flush=True)

    for dataset, task, paradigm_key in shard_paradigms:
        print(f"\n=== {paradigm_key} ===", flush=True)
        for model_name in ["cnn1d", "eegnet"]:
            for step in STEPS:
                try:
                    run_paradigm_step(model_name, dataset, task, paradigm_key, step, device)
                except Exception as exc:
                    print(f"[ERROR] {model_name} {paradigm_key} step={step}: {exc}", flush=True)
                    (ERROR_DIR / f"{model_name}__{paradigm_key}__step{step}.txt").write_text(str(exc))

    print("\nCNN-1D / EEGNet LOSO training complete.", flush=True)


if __name__ == "__main__":
    main()
