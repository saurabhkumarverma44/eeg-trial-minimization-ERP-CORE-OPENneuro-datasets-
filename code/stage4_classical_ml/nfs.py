from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
NFS_PATH = PROJECT_ROOT / "data" / "benchmarks" / "NFS_benchmarks.csv"

# Confirmed from project handoff spec: band_power feature block is always
# 5 bands x n_channels, regardless of paradigm. Do NOT derive band count
# from the NFS reference channel count -- different paradigms/datasets can
# have a different recorded channel set than the reference topography.
N_BANDS = 5


NON_EEG_COLUMNS = {"HEOG_left", "HEOG_right", "VEOG_lower"}


def load_reference_topography(paradigm_name: str):
    df = pd.read_csv(NFS_PATH)
    row = df[df.iloc[:, 0] == paradigm_name]
    if row.empty:
        return None
    # Exclude non-EEG (EOG/artifact) channels -- fixes a persistent
    # 30-vs-33 channel-count mismatch that prevented NFS from computing.
    channel_cols = [c for c in df.columns[1:] if c not in NON_EEG_COLUMNS]
    values = row.iloc[0][channel_cols].values.astype(float)
    return values, channel_cols


def importance_to_topography(importance, feature_blocks, block="band_power"):
    """
    Reshape the band_power importance block into a per-channel topography by
    averaging across the 5 known frequency bands. Returns (topo, n_channels_actual)
    so the caller can compare against the NFS reference channel count and decide
    whether NFS is computable for this paradigm, rather than assuming a match.
    """
    start, end = feature_blocks[block]
    block_vals = np.asarray(importance[start:end])
    block_size = end - start

    if block_size % N_BANDS != 0:
        raise ValueError(
            f"band_power block size {block_size} not divisible by N_BANDS={N_BANDS}; "
            f"the 5-band assumption itself may be wrong for this paradigm -- "
            f"check assemble_features_binary.py before trusting NFS."
        )

    n_channels_actual = block_size // N_BANDS
    reshaped = block_vals.reshape(N_BANDS, n_channels_actual)
    topo = reshaped.mean(axis=0)
    return topo, n_channels_actual


def bootstrap_nfs(topo_model, topo_ref, n_iter=1000, batch_size=200, seed=42):
    rng = np.random.default_rng(seed)
    n_channels = len(topo_model)

    if np.std(topo_model) == 0 or np.std(topo_ref) == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(np.corrcoef(topo_model, topo_ref)[0, 1])

    corrs = np.empty(n_iter, dtype=np.float64)
    for start in range(0, n_iter, batch_size):
        end = min(start + batch_size, n_iter)
        b = end - start
        idx = rng.integers(0, n_channels, size=(b, n_channels))
        a_batch = topo_model[idx]
        r_batch = topo_ref[idx]
        a_mean = a_batch.mean(axis=1, keepdims=True)
        r_mean = r_batch.mean(axis=1, keepdims=True)
        a_c = a_batch - a_mean
        r_c = r_batch - r_mean
        num = (a_c * r_c).sum(axis=1)
        den = np.sqrt((a_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            batch_corrs = np.where(den > 0, num / den, np.nan)
        corrs[start:end] = batch_corrs

    valid = corrs[~np.isnan(corrs)]
    if len(valid) == 0:
        return point, float("nan"), float("nan")
    ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)