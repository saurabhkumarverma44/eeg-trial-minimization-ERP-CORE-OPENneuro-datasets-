import numpy as np
from sklearn.model_selection import GroupShuffleSplit
def subject_wise_split(X, y, subject_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Performs subject-wise (group) train/val/test split to guarantee zero subject-level data leakage.
    """
    unique_subs = np.unique(subject_ids)
    np.random.seed(seed)
    shuffled_subs = np.random.permutation(unique_subs)
    n_subs = len(shuffled_subs)
    n_train = int(n_subs * train_ratio)
    n_val = int(n_subs * val_ratio)
    train_subs = set(shuffled_subs[:n_train])
    val_subs = set(shuffled_subs[n_train:n_train + n_val])
    test_subs = set(shuffled_subs[n_train + n_val:])
    train_mask = np.array([s in train_subs for s in subject_ids])
    val_mask = np.array([s in val_subs for s in subject_ids])
    test_mask = np.array([s in test_subs for s in subject_ids])
    print(f"[Phase 3 Split] Train Subjects ({len(train_subs)}): {train_mask.sum()} trials")
    print(f"[Phase 3 Split] Val Subjects   ({len(val_subs)}): {val_mask.sum()} trials")
    print(f"[Phase 3 Split] Test Subjects  ({len(test_subs)}): {test_mask.sum()} trials")
    return (X[train_mask], y[train_mask], subject_ids[train_mask]), \
           (X[val_mask], y[val_mask], subject_ids[val_mask]), \
           (X[test_mask], y[test_mask], subject_ids[test_mask])
