# dl_models/graph_builder.py

import numpy as np
from scipy.signal import butter, filtfilt

import sys
from pathlib import Path
sys.path.insert(0, str(Path("/lp-dev/23BRS1328/erp_trial_min/stage3_feature_extraction")))
from connectivity_core import plv_matrix, wpli_matrix

FREQ_BANDS = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
}
CONN_BANDS = {"alpha": (8.0, 12.0), "beta": (13.0, 30.0)}


def _bandpower_node_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns (n_channels, 5) relative band power per channel.
    """
    n_ch, n_times = epoch_data.shape
    powers = np.zeros((n_ch, len(FREQ_BANDS)), dtype=np.float64)
    nyq = sfreq / 2.0
    for bi, (fmin, fmax) in enumerate(FREQ_BANDS.values()):
        low, high = max(fmin / nyq, 1e-6), min(fmax / nyq, 0.999)
        b, a = butter(4, [low, high], btype="band")
        filtered = filtfilt(b, a, epoch_data, axis=-1)
        powers[:, bi] = (filtered ** 2).mean(axis=-1)
    total = powers.sum(axis=1, keepdims=True) + 1e-12
    return powers / total  # relative power, per channel


RAW_NODE_FEATURE_LENGTH = 64  # downsampled timepoints per electrode


def _raw_signal_node_features(epoch_data: np.ndarray) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns (n_channels, RAW_NODE_FEATURE_LENGTH) -- each electrode's
    raw signal, downsampled to a fixed length via simple averaging
    over non-overlapping windows. Richer than the 5-value band-power
    summary, giving the model access to temporal shape per electrode,
    not just aggregate spectral content -- closer to what CNN-1D sees,
    while retaining the graph structure across electrodes.

    NOT normalized here -- raw EEG is ~1e-5 scale (confirmed earlier
    this session to break training if fed in unnormalized). Normalize
    downstream, fit on TRAIN data only, same discipline as CNN-1D.
    """
    n_ch, n_times = epoch_data.shape
    window = n_times // RAW_NODE_FEATURE_LENGTH
    if window < 1:
        window = 1
    n_usable = window * RAW_NODE_FEATURE_LENGTH
    trimmed = epoch_data[:, :n_usable]
    reshaped = trimmed.reshape(n_ch, RAW_NODE_FEATURE_LENGTH, window)
    return reshaped.mean(axis=-1)  # (n_ch, RAW_NODE_FEATURE_LENGTH)


def build_graph_for_epoch(epoch_data: np.ndarray, sfreq: float, node_feature_type: str = "raw"):
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    node_feature_type: "bandpower" (5-dim spectral summary, original
        version) or "raw" (RAW_NODE_FEATURE_LENGTH-dim downsampled
        signal, richer temporal information).
    Returns (node_features, edge_index, edge_attr):
      edge_index: (2, n_ch*(n_ch-1)) fully-connected, directed both ways
      edge_attr: (n_ch*(n_ch-1), 4) -- [plv_alpha, plv_beta, wpli_alpha, wpli_beta]
    """
    n_ch = epoch_data.shape[0]
    if node_feature_type == "raw":
        node_features = _raw_signal_node_features(epoch_data)
    else:
        node_features = _bandpower_node_features(epoch_data, sfreq)

    plv_a = plv_matrix(epoch_data, sfreq, *CONN_BANDS["alpha"])
    plv_b = plv_matrix(epoch_data, sfreq, *CONN_BANDS["beta"])
    wpli_a = wpli_matrix(epoch_data, sfreq, *CONN_BANDS["alpha"])
    wpli_b = wpli_matrix(epoch_data, sfreq, *CONN_BANDS["beta"])

    src, dst = np.where(~np.eye(n_ch, dtype=bool))  # all i != j pairs
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = np.stack([plv_a[src, dst], plv_b[src, dst],
                           wpli_a[src, dst], wpli_b[src, dst]], axis=1)

    return node_features, edge_index, edge_attr
