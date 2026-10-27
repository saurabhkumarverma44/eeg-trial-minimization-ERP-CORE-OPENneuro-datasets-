# dl_models/evaluator.py

from typing import Dict, Tuple

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import torch
from torch.utils.data import DataLoader
from torch import nn


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test_loader and save metrics / ROC curve in output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y, _subjects in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Predictions and probabilities
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="binary")
    rec = recall_score(targets, preds, average="binary")
    f1 = f1_score(targets, preds, average="binary")

    # AUC: higher prob of class 1 (assuming labels {0,1})
    if probs.shape[1] == 2:
        auc = roc_auc_score(targets, probs[:, 1])
        fpr, tpr, _ = roc_curve(targets, probs[:, 1])
    else:
        # Fallback: macro-average AUC for multi-class (not expected in this project)
        auc = roc_auc_score(targets, probs, multi_class="ovo", average="macro")
        fpr, tpr, _ = roc_curve(targets, probs[:, 1])  # pick one class arbitrarily

    cm = confusion_matrix(targets, preds)

    metrics = {
        "Test_Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "AUC": float(auc),
        "ConfusionMatrix": cm.tolist(),
        "n_test_samples": int(len(targets)),
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved -> {metrics_path}")

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Eval] ROC curve saved -> {roc_path}")

    return metrics, targets, preds, probs