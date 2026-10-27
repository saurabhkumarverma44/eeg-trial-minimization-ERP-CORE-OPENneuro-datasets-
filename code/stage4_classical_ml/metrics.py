import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes classification metrics for one LOSO fold, with explicit handling
    of degenerate folds (test subject has only one true class after
    subsampling, common at low trial-count steps with imbalanced paradigms).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_true_classes = len(np.unique(y_true))
    degenerate = n_true_classes < 2

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if not degenerate else float("nan"),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0),
        "degenerate_fold": degenerate,
        "n_test_trials": len(y_true),
    }

    if y_proba is not None and not degenerate:
        try:
            out["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")

    return out