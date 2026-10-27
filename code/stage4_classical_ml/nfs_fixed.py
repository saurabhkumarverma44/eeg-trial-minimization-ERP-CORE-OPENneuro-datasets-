"""Corrected map construction and statistics for the Neuro-Fidelity Score (NFS).

Replaces the map-building and bootstrap parts of nfs.py and attention_topography.py.
Fixes three confirmed defects:
  1. Band-power importances are stored channel-major (index = channel * N_BANDS + band),
     because the features are saved as (n_epochs, n_channels, n_bands) and flattened.
     The old code did reshape(N_BANDS, n_channels), which mixes channels and bands.
  2. The GAT map summed attention over each node's INCOMING edges. Softmax is already
     normalised over exactly those edges, so every electrode got 1/n. The corrected map
     sums the attention each electrode RECEIVES FROM the others (grouped by source node),
     which is not fixed by the softmax.
  3. The old bootstrap resampled electrodes. The corrected one resamples subjects (folds).
"""
import numpy as np

N_BANDS = 5


def importance_to_topography(importance, feature_blocks, n_channels, block="band_power", n_bands=N_BANDS):
    start, end = feature_blocks[block]
    vals = np.asarray(importance[start:end], dtype=float)
    if vals.size != n_channels * n_bands:
        raise ValueError(f"{block} block has {vals.size} values, expected {n_channels} x {n_bands}")
    # channel-major storage: [ch0b0, ch0b1, ..., ch0b4, ch1b0, ...]
    return vals.reshape(n_channels, n_bands).mean(axis=1)


def attention_to_node_topography(edge_index, attention, n_nodes, drop_self_loops=True):
    """edge_index: (2, E) array [source, destination]; attention: (E,) or (E, heads).
    Returns, per electrode, the mean attention it receives from the other electrodes."""
    ei = np.asarray(edge_index)
    a = np.asarray(attention, dtype=float)
    if a.ndim > 1:
        a = a.mean(axis=1)
    src, dst = ei[0], ei[1]
    if drop_self_loops:
        keep = src != dst
        src, a = src[keep], a[keep]
    topo = np.zeros(n_nodes)
    np.add.at(topo, src, a)
    out_deg = np.zeros(n_nodes)
    np.add.at(out_deg, src, 1.0)
    return topo / np.maximum(out_deg, 1.0)


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def subject_bootstrap_nfs(fold_topos, topo_ref, n_iter=2000, seed=0):
    """fold_topos: (n_folds, n_channels), one map per held-out-subject fold.
    Resamples folds with replacement, averages the maps, correlates with the reference.
    Fold maps share most of their training data, so this interval is still optimistic;
    report it as such."""
    rng = np.random.default_rng(seed)
    T = np.asarray(fold_topos, float)
    point = pearson(T.mean(axis=0), topo_ref)
    n = T.shape[0]
    vals = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        vals.append(pearson(T[idx].mean(axis=0), topo_ref))
    vals = np.array([v for v in vals if not np.isnan(v)])
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (float("nan"),) * 2
    return point, float(lo), float(hi)


def permutation_p(topo_model, topo_ref, n_perm=10000, seed=0):
    """Two-sided p-value: how often shuffling the electrodes gives |r| as large."""
    rng = np.random.default_rng(seed)
    obs = pearson(topo_model, topo_ref)
    m = np.asarray(topo_model, float)
    cnt = 0
    for _ in range(n_perm):
        if abs(pearson(rng.permutation(m), topo_ref)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def batch_attention_topography(edge_index, attention, n_ch):
    """Average the per-electrode attention map over EVERY graph in a test batch.
    PyG offsets node ids by n_ch for each graph in the batch, so graph id = src // n_ch.
    Returns (n_ch,) map averaged over graphs."""
    ei = np.asarray(edge_index)
    a = np.asarray(attention, dtype=float)
    if a.ndim > 1:
        a = a.mean(axis=1)
    src, dst = ei[0], ei[1]
    g = src // n_ch
    n_graphs = int(g.max()) + 1
    maps = []
    for k in range(n_graphs):
        m = g == k
        maps.append(attention_to_node_topography(
            np.vstack([src[m] - k * n_ch, dst[m] - k * n_ch]), a[m], n_ch))
    return np.mean(maps, axis=0)


def align_reference(ref_values, ref_channels, feature_channels):
    """Put the reference topography into the same electrode order as the features.
    Raises if any feature channel is missing from the reference, so a silent
    order or name mismatch can no longer pass."""
    lookup = {c.strip().lower(): v for c, v in zip(ref_channels, ref_values)}
    missing = [c for c in feature_channels if c.strip().lower() not in lookup]
    if missing:
        raise ValueError(f"channels missing from the reference topography: {missing}")
    return np.array([lookup[c.strip().lower()] for c in feature_channels], dtype=float)
