import numpy as np
class EEGChannelScaler:
    """
    Normalizes EEG channels using Z-score standard scaling.
    Calculates mean and standard deviation strictly on the training set to eliminate data leakage.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None
    def fit(self, X_train):
        # X shape: (N_trials, N_channels, N_times)
        # Calculate mean & std across trials and time samples for each channel
        self.mean = np.mean(X_train, axis=(0, 2), keepdims=True)
        self.std = np.std(X_train, axis=(0, 2), keepdims=True)
        return self
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted on training data before calling transform.")
        return (X - self.mean) / (self.std + self.eps)
    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
