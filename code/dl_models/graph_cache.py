# dl_models/graph_cache.py

import numpy as np
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path("/lp-dev/23BRS1328/erp_trial_min")))
from dl_prep.explicit_epoch_loader import load_explicit
from dl_models.graph_builder import build_graph_for_epoch

CACHE_DIR = Path("/lp-dev/23BRS1328/erp_trial_min/results_dl/graph_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_and_cache_paradigm(dataset: str, task: str, paradigm_key: str, results_batch_dir: Path):
    """
    Loads raw epochs, builds a graph for every single epoch, and caches
    the result to disk -- avoids rebuilding graphs on every training run
    of every LOSO fold of every trial-count step for the same paradigm.
    """
    cache_path = CACHE_DIR / f"{paradigm_key}.pkl"
    if cache_path.exists():
        print(f"[CACHED] {paradigm_key} already exists, skipping")
        return cache_path

    X, y, subject_ids, ch_names, sfreq = load_explicit(results_batch_dir, dataset, task)

    node_features_list = []
    edge_index_list = []
    edge_attr_list = []

    for i in range(X.shape[0]):
        nf, ei, ea = build_graph_for_epoch(X[i], sfreq, node_feature_type="bandpower")
        node_features_list.append(nf.astype(np.float32))
        edge_index_list.append(ei.astype(np.int64))
        edge_attr_list.append(ea.astype(np.float32))

    cache_data = {
        "node_features": node_features_list,
        "edge_index": edge_index_list,
        "edge_attr": edge_attr_list,
        "y": y,
        "subject_ids": subject_ids,
        "n_channels": X.shape[1],
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"[BUILT] {paradigm_key}: {X.shape[0]} graphs cached -> {cache_path}")
    return cache_path


def load_cached_paradigm(paradigm_key: str):
    cache_path = CACHE_DIR / f"{paradigm_key}.pkl"
    with open(cache_path, "rb") as f:
        return pickle.load(f)
