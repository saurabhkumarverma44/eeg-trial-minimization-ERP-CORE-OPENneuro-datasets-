#!/usr/bin/env python3
"""
run_gat_nfs.py  (run from D:\\erp_work)

Corrected GAT LOSO run for the Neuro-Fidelity Score. The training protocol is
copied unchanged from run_gat_loso.py so accuracy stays comparable. What changed:

  * attention map: mean attention each electrode RECEIVES from the others,
    averaged over EVERY test graph of the fold (old code used only the first
    graph and a quantity that is constant by construction)
  * one map is kept per held-out-subject fold and saved
  * the reference topography is matched to the cached electrodes BY NAME
  * the confidence interval resamples folds (subjects), not electrodes
  * an electrode-permutation p-value is added
  * no dependency on the missing explicit_epoch_loader.py: graphs come from
    the verified caches in results_dl_2\\graph_cache and names from
    results_gat\\ch_names (written by verify_cache.py)

Nothing in the backup is modified. Outputs go to --out (default results_gat_v2).

Examples
  python run_gat_nfs.py --paradigms ERP_CORE_P3 --steps 50 --max-folds 3 --out results_gat_timing
  python run_gat_nfs.py --steps 50,200,full
"""
import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_classical_ml.nfs_fixed import (  # noqa: E402
    batch_attention_topography, subject_bootstrap_nfs, permutation_p, align_reference,
)

SRC = Path(r"E:\erp_trial_min_backup_20260626\erp_trial_min")
CACHE_DIR = SRC / "results_dl_2" / "graph_cache"
REF_CSV = Path("data") / "benchmarks" / "NFS_benchmarks.csv"
NAMES_DIR = Path("results_gat") / "ch_names"

PARADIGM_TO_NFS_NAME = {
    "ERP_CORE_P3": "P3", "ERP_CORE_MMN": "MMN", "ERP_CORE_N2pc": "N2pc",
    "ERP_CORE_N400": "N400", "ERP_CORE_N170": "N170", "ERP_CORE_ERN": "ERN",
}
NOTES = {"ERP_CORE_ERN": "epochs are stimulus-locked but the reference is response-locked: not comparable"}
NON_EEG_COLUMNS = {"HEOG_left", "HEOG_right", "VEOG_lower"}
ALL_STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"]
CSV_FIELDS = ["paradigm", "step", "n_folds", "nfs", "ci_low", "ci_high", "p_perm",
              "map_min_std", "mean_bal_acc", "note"]


def load_reference(name):
    import pandas as pd
    df = pd.read_csv(REF_CSV)
    row = df[df.iloc[:, 0] == name]
    if row.empty:
        return None
    cols = [c for c in df.columns[1:] if c not in NON_EEG_COLUMNS]
    return row.iloc[0][cols].values.astype(float), cols


def summarise_and_save(out_dir, paradigm, step, fold_topos, subjects, bal_accs, ch_names):
    """Pure numpy: save fold maps, compute NFS statistics, append one CSV row."""
    fold_topos = np.stack(fold_topos)
    maps_dir = out_dir / "maps" / paradigm
    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / f"step_{step}_foldmaps.npy", fold_topos)
    (maps_dir / f"step_{step}_subjects.json").write_text(json.dumps(
        {"subjects": [str(s) for s in subjects], "ch_names": list(ch_names)}))

    map_min_std = float(fold_topos.std(axis=1).min())
    row = {"paradigm": paradigm, "step": step, "n_folds": len(fold_topos),
           "nfs": "", "ci_low": "", "ci_high": "", "p_perm": "",
           "map_min_std": f"{map_min_std:.3e}",
           "mean_bal_acc": f"{np.nanmean(bal_accs):.4f}" if len(bal_accs) else "",
           "note": NOTES.get(paradigm, "")}
    if map_min_std < 1e-6:
        row["note"] = (row["note"] + " | CONSTANT MAP: nothing to correlate").strip(" |")
    else:
        ref = load_reference(PARADIGM_TO_NFS_NAME[paradigm])
        if ref is None:
            row["note"] = (row["note"] + " | no reference").strip(" |")
        else:
            ref_vals = align_reference(ref[0], ref[1], ch_names)
            point, lo, hi = subject_bootstrap_nfs(fold_topos, ref_vals)
            _, p = permutation_p(fold_topos.mean(axis=0), ref_vals)
            row.update(nfs=f"{point:.4f}", ci_low=f"{lo:.4f}", ci_high=f"{hi:.4f}", p_perm=f"{p:.4f}")
    csv_path = out_dir / "gat_nfs_v2.csv"
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def run_paradigm_step(paradigm, step, device, out_dir, max_folds):
    import torch
    from torch_geometric.data import Data, Batch
    from dl_models.gat_model import EEGGAT
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    marker_dir = out_dir / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"gat__{paradigm}__step{step}.done"
    if marker.exists():
        print(f"[EXISTS] {paradigm} step={step}", flush=True)
        return

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    t0 = time.time()
    with open(CACHE_DIR / f"{paradigm}.pkl", "rb") as f:
        cache = pickle.load(f)
    ch_names = json.loads((NAMES_DIR / f"{paradigm}.json").read_text())
    n_ch = int(cache["n_channels"])
    assert len(ch_names) == n_ch, f"{paradigm}: {len(ch_names)} names vs {n_ch} channels"

    def make_data_list(indices, ch_mean, ch_std):
        out = []
        for i in indices:
            nf = (cache["node_features"][i] - ch_mean) / ch_std
            out.append(Data(x=torch.from_numpy(nf).float(),
                            edge_index=torch.from_numpy(cache["edge_index"][i]).long(),
                            edge_attr=torch.from_numpy(cache["edge_attr"][i]).float(),
                            y=torch.tensor([cache["y"][i]]).long()))
        return out

    subjects = sorted(set(cache["subject_ids"]))
    if max_folds:
        subjects = subjects[:max_folds]
    rng = np.random.default_rng(42)
    fold_topos, fold_subjects, bal_accs = [], [], []

    for subject in subjects:
        tf = time.time()
        test_mask = cache["subject_ids"] == subject
        train_idx_full = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if step == "full" or step >= len(train_idx_full):
            sub_idx = np.arange(len(train_idx_full))
        else:
            sub_idx = rng.choice(len(train_idx_full), size=step, replace=False)
        train_idx = train_idx_full[sub_idx]
        if len(np.unique(cache["y"][train_idx])) < 2:
            continue
        n_val = max(1, int(0.15 * len(train_idx)))
        perm = rng.permutation(len(train_idx))
        val_idx, tr_idx = train_idx[perm[:n_val]], train_idx[perm[n_val:]]
        if len(np.unique(cache["y"][tr_idx])) < 2:
            continue

        feats = np.stack([cache["node_features"][i] for i in tr_idx])
        ch_mean, ch_std = feats.mean(axis=0), feats.std(axis=0) + 1e-8
        train_g = make_data_list(tr_idx, ch_mean, ch_std)
        val_g = make_data_list(val_idx, ch_mean, ch_std)
        test_g = make_data_list(test_idx, ch_mean, ch_std)

        model = EEGGAT(in_channels=5, num_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val, patience, best_state = -1.0, 0, None
        for epoch in range(60):
            model.train()
            order = np.random.permutation(len(train_g))
            for s in range(0, len(order), 32):
                b = Batch.from_data_list([train_g[j] for j in order[s:s + 32]]).to(device)
                opt.zero_grad()
                loss = crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vb = Batch.from_data_list(val_g).to(device)
                vo = model(vb.x, vb.edge_index, vb.edge_attr, vb.batch)
                val_acc = (torch.argmax(vo, -1).cpu().numpy() == vb.y.cpu().numpy()).mean()
            if val_acc > best_val + 1e-4:
                best_val, patience = val_acc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            # attention is computed in chunks of test graphs so memory stays small;
            # each chunk is a whole number of graphs, so graph ids stay valid
            maps, weights, preds_all, targets_all = [], [], [], []
            for s in range(0, len(test_g), 64):
                tb = Batch.from_data_list(test_g[s:s + 64]).to(device)
                out, (ei, aw) = model(tb.x, tb.edge_index, tb.edge_attr, tb.batch, return_attention=True)
                maps.append(batch_attention_topography(ei.cpu().numpy(), aw.cpu().numpy(), n_ch))
                weights.append(tb.num_graphs)
                preds_all.append(torch.argmax(out, -1).cpu().numpy())
                targets_all.append(tb.y.cpu().numpy())
        fold_topos.append(np.average(np.stack(maps), axis=0, weights=weights))
        fold_subjects.append(subject)
        preds, targets = np.concatenate(preds_all), np.concatenate(targets_all)
        bal_accs.append(balanced_accuracy_score(targets, preds) if len(np.unique(targets)) > 1 else np.nan)
        print(f"  fold {subject}: map std={fold_topos[-1].std():.3e} "
              f"bal_acc={bal_accs[-1]:.3f} trials={len(tr_idx)} time={time.time() - tf:.1f}s", flush=True)

    if not fold_topos:
        print(f"[NO DATA] {paradigm} step={step}", flush=True)
        marker.touch()
        return
    row = summarise_and_save(out_dir, paradigm, step, fold_topos, fold_subjects, bal_accs, ch_names)
    print(f"[DONE] {paradigm} step={step} NFS={row['nfs']} [{row['ci_low']}, {row['ci_high']}] "
          f"p={row['p_perm']} map_min_std={row['map_min_std']} time={time.time() - t0:.0f}s", flush=True)
    if not max_folds:  # timing runs never leave a marker behind
        marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigms", default=",".join(PARADIGM_TO_NFS_NAME))
    ap.add_argument("--steps", default="50,200,full")
    ap.add_argument("--max-folds", type=int, default=0, help="use only the first N subjects (timing test)")
    ap.add_argument("--out", default="results_gat_v2")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [s if s == "full" else int(s) for s in args.steps.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} threads={torch.get_num_threads()} out={out_dir}", flush=True)
    for paradigm in args.paradigms.split(","):
        print(f"\n=== {paradigm} ===", flush=True)
        for step in steps:
            try:
                run_paradigm_step(paradigm, step, device, out_dir, args.max_folds)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                (out_dir / f"ERROR_{paradigm}_step{step}.txt").write_text(repr(exc))


if __name__ == "__main__":
    main()
