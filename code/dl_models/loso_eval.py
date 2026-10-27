# dl_models/loso_eval.py

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)


def evaluate_fold(model, test_loader, device, use_amp=False, amp_dtype=torch.bfloat16):
    """
    Lightweight single-fold evaluation -- returns a metrics dict, no disk I/O.
    Used inside a LOSO loop (called once per subject per step per model),
    where writing a file per call would be impractical at scale.
    """
    model.eval()
    model.to(device)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y, _subjects in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                logits = logits.float()  # bfloat16 has no numpy equivalent --
                                          # cast back to float32 before leaving
                                          # the autocast context, not after.
            else:
                logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    # Degenerate-fold guard: a single-class test fold makes several
    # metrics (AUC especially) undefined -- flag rather than crash or
    # silently emit a misleading number.
    unique_targets = np.unique(targets)
    degenerate = len(unique_targets) < 2

    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)) if not degenerate else float("nan"),
        "f1": float(f1_score(targets, preds, average="binary", zero_division=0)),
        "auc": float(roc_auc_score(targets, probs[:, 1])) if (not degenerate and probs.shape[1] == 2) else float("nan"),
        "n_test_trials": int(len(targets)),
        "degenerate_fold": degenerate,
    }
    return metrics
