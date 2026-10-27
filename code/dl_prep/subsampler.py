import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def subsample_training_data(X_train, y_train, sub_train, fraction=1.0, seed=42):
    """
    Subsamples training EEG trials at specified fraction (e.g. 0.50 for 50%)
    using stratified sampling to preserve label ratio.
    """
    if fraction >= 0.99:
        return X_train, y_train, sub_train

    n_samples = len(y_train)
    n_subsample = int(n_samples * fraction)
    n_subsample = max(n_subsample, len(np.unique(y_train)) * 2)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_subsample, random_state=seed)
    train_idx, _ = next(sss.split(X_train, y_train))

    X_sub = X_train[train_idx]
    y_sub = y_train[train_idx]
    sub_sub = [sub_train[i] for i in train_idx]

    return X_sub, y_sub, sub_sub
