#!/usr/bin/env python3
"""
Stage 2: Graph Attention Network, LOSO cross-validation
==========================================================

Data source: pre-cached graphs (dl_models/graph_cache.py), built once
per paradigm from raw epochs via dl_models/graph_builder.py.

Same LOSO protocol, same trial-count grid, same resumable marker
convention as classical ML and CNN-1D/EEGNet, for direct comparability.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
sys.path.insert(0, str(PROJECT_ROOT))

from dl_prep.explicit_epoch_loader import CONDITION_SPEC
from dl_models.graph_cache import build_and_cache_paradigm, load_cached_paradigm
from dl_models.gat_model import EEGGAT
from dl_models.attention_topography import attention_to_node_topography
from stage4_classical_ml.nfs import load_reference_topography, bootstrap_nfs
from dl_prep.explicit_epoch_loader import CONDITION_SPEC as _CS_CHECK  # noqa

PARADIGM_TO_NFS_NAME = {
    "ERP_CORE_P3": "P3", "ERP_CORE_MMN": "MMN", "ERP_CORE_N2pc": "N2pc",
    "ERP_CORE_N400": "N400", "ERP_CORE_N170": "N170", "ERP_CORE_ERN": "ERN",
}
from dl_models.loso_eval import evaluate_fold  # reused, but needs a graph-aware variant -- see below

OUT_ROOT = PROJECT_ROOT / "results_gat"
MARKER_DIR = OUT_ROOT / "markers"
ERROR_DIR = OUT_ROOT / "errors"
MODEL_DIR = OUT_ROOT / "gat"
for d in [OUT_ROOT, MARKER_DIR, ERROR_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NFS_DIR = OUT_ROOT / "nfs"
NFS_DIR.mkdir(parents=True, exist_ok=True)
TOPO_DIR = OUT_ROOT / "attention_weights"
TOPO_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"]


def get_paradigms():
    KNOWN_DATASETS = ["ERP_CORE", "DS006018", "DS002680"]
    pairs = []
    for key in CONDITION_SPEC:
        for ds in KNOWN_DATASETS:
            if key.startswith(ds + "_"):
                pairs.append((ds, key[len(ds) + 1:], key))
                break
    return sorted(pairs, key=lambda x: x[2])


def marker_path(paradigm_key, step):
    return MARKER_DIR / f"gat__{paradigm_key}__step{step}.done"


def subsample_indices(n_total, step, rng):
    if step == "full" or step >= n_total:
        return np.arange(n_total)
    return rng.choice(n_total, size=step, replace=False)


def make_data_list(cache, indices, ch_mean, ch_std):
    graphs = []
    for i in indices:
        nf = (cache["node_features"][i] - ch_mean) / ch_std
        graphs.append(Data(
            x=torch.from_numpy(nf).float(),
            edge_index=torch.from_numpy(cache["edge_index"][i]).long(),
            edge_attr=torch.from_numpy(cache["edge_attr"][i]).float(),
            y=torch.tensor([cache["y"][i]]).long(),
        ))
    return graphs


def run_paradigm_step(paradigm_key, dataset, task, step, device):
    marker = marker_path(paradigm_key, step)
    if marker.exists():
        print(f"[EXISTS] gat {paradigm_key} step={step}", flush=True)
        return

    # Seed here, inside the function itself -- do not rely on the
    # calling notebook/script to have seeded beforehand. A kernel
    # restart without re-seeding was confirmed to produce different
    # (though still real, non-degenerate) results across identical runs.
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    t0 = time.time()
    cache_path = build_and_cache_paradigm(dataset, task, paradigm_key, PROJECT_ROOT / "results_batch")
    cache = load_cached_paradigm(paradigm_key)

    subjects = sorted(set(cache["subject_ids"]))
    rng = np.random.default_rng(42)
    rows = []
    topographies = []  # per-fold attention-derived topographies (WBS 5.4)
    n_degenerate = 0

    for subject in subjects:
        test_mask = cache["subject_ids"] == subject
        train_mask = ~test_mask
        train_idx_full = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        y_train_full = cache["y"][train_idx_full]
        sub_idx = subsample_indices(len(train_idx_full), step, rng)
        train_idx = train_idx_full[sub_idx]
        y_train = cache["y"][train_idx]

        if len(np.unique(y_train)) < 2:
            n_degenerate += 1
            continue

        n_val = max(1, int(0.15 * len(train_idx)))
        perm = rng.permutation(len(train_idx))
        val_sel, tr_sel = perm[:n_val], perm[n_val:]
        tr_idx, val_idx = train_idx[tr_sel], train_idx[val_sel]
        if len(np.unique(cache["y"][tr_idx])) < 2:
            n_degenerate += 1
            continue

        # Per-channel z-score, fit on TRAIN only -- same discipline as
        # CNN-1D/EEGNet, applied here to node features instead of raw signal
        train_node_feats = np.stack([cache["node_features"][i] for i in tr_idx])
        # NOTE: no keepdims here -- ch_mean/ch_std must be (n_channels, n_features),
        # matching a single epoch's node_features shape (31, 5), NOT
        # (1, 31, 5). keepdims=True previously caused numpy to broadcast
        # a single (31,5) array UP to 3D when subtracting, which GATConv
        # correctly rejected ("Static graphs not supported").
        ch_mean = train_node_feats.mean(axis=0)
        ch_std = train_node_feats.std(axis=0) + 1e-8

        train_graphs = make_data_list(cache, tr_idx, ch_mean, ch_std)
        val_graphs = make_data_list(cache, val_idx, ch_mean, ch_std)
        test_graphs = make_data_list(cache, test_idx, ch_mean, ch_std)

        model = EEGGAT(in_channels=5, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(60):  # raised from 30 -- confirmed useful signal
                                  # can still emerge past epoch 30 in diagnostics
            model.train()
            perm_tr = np.random.permutation(len(train_graphs))
            for start in range(0, len(perm_tr), 32):
                batch_idx = perm_tr[start:start + 32]
                batch = Batch.from_data_list([train_graphs[j] for j in batch_idx]).to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_batch = Batch.from_data_list(val_graphs).to(device)
                val_out = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.batch)
                val_preds = torch.argmax(val_out, dim=-1).cpu().numpy()
                val_targets = val_batch.y.cpu().numpy()
                val_acc = (val_preds == val_targets).mean()

            # Track val_acc, not val_loss -- confirmed via diagnostic that
            # the two can diverge (val_loss climbing while val_acc still
            # improves), and val_acc is the metric that actually matters
            # for the paper's reported balanced accuracy.
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 12:  # slightly more patience given
                                             # the noisier val_acc signal
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_batch = Batch.from_data_list(test_graphs).to(device)
            test_out, (attn_edge_index, attn_weights) = model(
                test_batch.x, test_batch.edge_index, test_batch.edge_attr,
                test_batch.batch, return_attention=True
            )
            probs = torch.softmax(test_out, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            targets = test_batch.y.cpu().numpy()

            # Extract a per-electrode topography from this fold's
            # attention weights (WBS 5.4). Since the test batch
            # contains multiple graphs (one per test trial), restrict
            # to edges belonging to the FIRST graph in the batch as a
            # representative single-graph topology (all graphs share
            # identical electrode structure within a paradigm/subject).
            n_ch = cache["n_channels"]
            first_graph_edge_mask = attn_edge_index[0] < n_ch
            fold_topo = attention_to_node_topography(
                attn_edge_index[:, first_graph_edge_mask],
                attn_weights[first_graph_edge_mask],
                n_nodes=n_ch,
            )
            topographies.append(fold_topo)

        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
        unique_t = np.unique(targets)
        degenerate_fold = len(unique_t) < 2
        rows.append({
            "subject": subject,
            "accuracy": accuracy_score(targets, preds),
            "balanced_accuracy": balanced_accuracy_score(targets, preds) if not degenerate_fold else np.nan,
            "auc": roc_auc_score(targets, probs[:, 1]) if not degenerate_fold else np.nan,
            "n_train_trials": len(tr_idx),
            "degenerate_fold": degenerate_fold,
        })

    if not rows:
        print(f"[NO DATA] gat {paradigm_key} step={step}", flush=True)
        marker.touch()
        return

    out_dir = MODEL_DIR / paradigm_key
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with (out_dir / f"step_{step}_folds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_bal_acc = np.nanmean([r["balanced_accuracy"] for r in rows])
    elapsed = time.time() - t0
    print(f"[DONE] model=gat paradigm={paradigm_key} step={step} "
          f"balanced_accuracy={mean_bal_acc:.4f} degenerate={n_degenerate}/{len(subjects)}", flush=True)

    # WBS 5.4: average attention-derived topography across folds, save
    if topographies:
        avg_topo = np.mean(np.stack(topographies), axis=0)
        topo_out_dir = TOPO_DIR / paradigm_key
        topo_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(topo_out_dir / f"step_{step}.npy", avg_topo)

        # WBS 5.5: compute NFS from this topography, if a reference exists
        reference_name = PARADIGM_TO_NFS_NAME.get(paradigm_key)
        nfs_csv = NFS_DIR / "gat_nfs.csv"
        if reference_name is not None:
            ref_result = load_reference_topography(reference_name)
            if ref_result is not None:
                ref_topo, ref_channels = ref_result
                if len(avg_topo) == len(ref_topo):
                    point, ci_low, ci_high = bootstrap_nfs(avg_topo, ref_topo)
                    import csv as _csv
                    need_header = (not nfs_csv.exists()) or nfs_csv.stat().st_size == 0
                    with nfs_csv.open("a", newline="") as f:
                        w = _csv.DictWriter(f, fieldnames=["dataset_task", "model", "step", "nfs", "ci_low", "ci_high", "note"])
                        if need_header:
                            w.writeheader()
                        w.writerow({"dataset_task": paradigm_key, "model": "gat", "step": step,
                                    "nfs": point, "ci_low": ci_low, "ci_high": ci_high, "note": ""})
                    print(f"[NFS] gat {paradigm_key} step={step}: NFS={point:.4f} [{ci_low:.4f}, {ci_high:.4f}]", flush=True)
                else:
                    print(f"[NFS] gat {paradigm_key} step={step}: channel mismatch "
                          f"(topo={len(avg_topo)}, ref={len(ref_topo)}), skipped", flush=True)

    print(f"[TIME] gat / {paradigm_key} step={step}: {elapsed:.1f}s", flush=True)
    marker.touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paradigms = get_paradigms()
    shard = [p for i, p in enumerate(all_paradigms) if i % args.num_shards == args.shard_index]

    print(f"GAT LOSO Training | Device: {device} | Total: {len(all_paradigms)} | Shard: {len(shard)}", flush=True)

    for dataset, task, paradigm_key in shard:
        print(f"\n=== {paradigm_key} ===", flush=True)
        for step in STEPS:
            try:
                run_paradigm_step(paradigm_key, dataset, task, step, device)
            except Exception as exc:
                print(f"[ERROR] gat {paradigm_key} step={step}: {exc}", flush=True)
                (ERROR_DIR / f"gat__{paradigm_key}__step{step}.txt").write_text(str(exc))


if __name__ == "__main__":
    main()