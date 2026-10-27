#!/usr/bin/env python3
"""
run_gat_v4.py  (v3 plus --class-weight and --readout concat)

run_gat_v3.py  (run from D:\\erp_work)

GAT LOSO under the revised protocol shared with the classical models:
  * N = number of POOLED training trials, drawn stratified by class from the training subjects
  * early stopping uses a SUBJECT-WISE validation set: ~15% of the training subjects (at least 2)
    are held out entirely, and validation trials are drawn only from them (up to 200, stratified).
    Validation trials do not count towards N.
  * early-stopping metric: balanced accuracy (not plain accuracy), so imbalanced paradigms are not rewarded
    for predicting the majority class
  * node-feature z-score fit on the training trials only (unchanged)
  * model, optimiser and epoch limits unchanged from run_gat_loso.py
  * corrected attention map, subject-level bootstrap, permutation p, name-aligned reference (from run_gat_nfs.py)
  * per-fold accuracy written to fold_accuracy_gat_v3.csv so undefined folds can be counted

Usage
  python run_gat_v3.py --paradigms ERP_CORE_P3 --steps 50 --max-folds 3 --out results_gat_v3_timing
  python run_gat_v3.py --steps 5,10,20,30,50,75,100,150,200 --out results_gat_v3
"""
import argparse, csv, json, os, pickle, random, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gat_nfs as G  # noqa: E402
from run_classical_nfs import sample_pooled  # noqa: E402

VAL_MAX = 200
ALL_STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def make_concat_gat(n_ch):
    """Same two attention layers as EEGGAT, but the readout concatenates the 30 node embeddings (30 x 32) instead of
    global mean pooling, so per-electrode information reaches the classifier."""
    import torch.nn as nn
    import torch.nn.functional as F
    from dl_models.gat_model import EEGGAT

    class EEGGATConcat(EEGGAT):
        def __init__(self):
            super().__init__(in_channels=5, num_classes=2)
            self.classifier = nn.Linear(n_ch * 32, 2)

        def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
            x, (ei1, a1) = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
            x = self.dropout(F.elu(x))
            x, (ei2, a2) = self.gat2(x, edge_index, edge_attr, return_attention_weights=True)
            x = F.elu(x)
            pooled = x.reshape(-1, n_ch * 32)   # graphs have exactly n_ch nodes, stored consecutively
            out = self.classifier(self.dropout(pooled))
            return (out, (ei2, a2)) if return_attention else out
    return EEGGATConcat()


def split_subjects(train_subjects, rng):
    subs = np.array(sorted(train_subjects))
    n_val = max(2, int(round(0.15 * len(subs))))
    val = set(rng.choice(subs, size=n_val, replace=False).tolist())
    return val


def run_paradigm_step(paradigm, step, device, out_dir, max_folds, class_weight=False, readout='mean'):
    import torch
    from torch_geometric.data import Data, Batch
    from dl_models.gat_model import EEGGAT
    from sklearn.metrics import balanced_accuracy_score

    marker_dir = out_dir / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"gat3__{paradigm}__step{step}.done"
    if marker.exists():
        print(f"[EXISTS] {paradigm} step={step}", flush=True)
        return

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    t0 = time.time()
    with open(G.CACHE_DIR / f"{paradigm}.pkl", "rb") as f:
        cache = pickle.load(f)
    ch_names = json.loads((G.NAMES_DIR / f"{paradigm}.json").read_text())
    n_ch = int(cache["n_channels"])
    assert len(ch_names) == n_ch
    y_all = np.asarray(cache["y"])
    subj_all = np.asarray([str(s) for s in cache["subject_ids"]])

    def make_data_list(indices, ch_mean, ch_std):
        out = []
        for i in indices:
            nf = (cache["node_features"][i] - ch_mean) / ch_std
            out.append(Data(x=torch.from_numpy(nf).float(),
                            edge_index=torch.from_numpy(cache["edge_index"][i]).long(),
                            edge_attr=torch.from_numpy(cache["edge_attr"][i]).float(),
                            y=torch.tensor([y_all[i]]).long()))
        return out

    subjects = sorted(set(subj_all))
    if max_folds:
        subjects = subjects[:max_folds]
    rng = np.random.default_rng(42)
    fold_topos, fold_subjects, bal_accs = [], [], []
    acc_csv = out_dir / "fold_accuracy_gat_v3.csv"

    for subject in subjects:
        tf = time.time()
        test_idx = np.where(subj_all == subject)[0]
        other = np.where(subj_all != subject)[0]
        val_subs = split_subjects(set(subj_all[other]), rng)
        is_val = np.isin(subj_all[other], list(val_subs))
        pool_tr, pool_val = other[~is_val], other[is_val]
        sel = pool_tr[sample_pooled(y_all[pool_tr], None if step == "full" else int(step), rng)]
        val_idx = pool_val[sample_pooled(y_all[pool_val], VAL_MAX, rng)]
        if len(np.unique(y_all[sel])) < 2 or len(np.unique(y_all[val_idx])) < 2:
            print(f"  fold {subject}: skipped (one class in train or validation)", flush=True)
            continue

        feats = np.stack([cache["node_features"][i] for i in sel])
        ch_mean, ch_std = feats.mean(axis=0), feats.std(axis=0) + 1e-8
        train_g = make_data_list(sel, ch_mean, ch_std)
        val_g = make_data_list(val_idx, ch_mean, ch_std)
        test_g = make_data_list(test_idx, ch_mean, ch_std)

        model = (make_concat_gat(n_ch) if readout == 'concat' else EEGGAT(in_channels=5, num_classes=2)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
        if class_weight:
            cnt = np.maximum(np.bincount(y_all[sel], minlength=2).astype(float), 1.0)
            crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=device),
                                             label_smoothing=0.1)
        else:
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
                val_ba = balanced_accuracy_score(vb.y.cpu().numpy(), torch.argmax(vo, -1).cpu().numpy())
            if val_ba > best_val + 1e-4:
                best_val, patience = val_ba, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            maps, weights, preds_all, targets_all = [], [], [], []
            for s in range(0, len(test_g), 64):
                tb = Batch.from_data_list(test_g[s:s + 64]).to(device)
                out, (ei, aw) = model(tb.x, tb.edge_index, tb.edge_attr, tb.batch, return_attention=True)
                maps.append(G.batch_attention_topography(ei.cpu().numpy(), aw.cpu().numpy(), n_ch))
                weights.append(tb.num_graphs)
                preds_all.append(torch.argmax(out, -1).cpu().numpy())
                targets_all.append(tb.y.cpu().numpy())
        fold_topos.append(np.average(np.stack(maps), axis=0, weights=weights))
        fold_subjects.append(subject)
        preds, targets = np.concatenate(preds_all), np.concatenate(targets_all)
        ba = balanced_accuracy_score(targets, preds) if len(np.unique(targets)) > 1 else np.nan
        bal_accs.append(ba)
        new = not acc_csv.exists() or acc_csv.stat().st_size == 0
        with acc_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["paradigm", "step", "subject", "bal_acc", "val_bal_acc", "n_train", "n_val", "val_subjects"])
            w.writerow([paradigm, step, subject, "" if np.isnan(ba) else f"{ba:.4f}", f"{best_val:.4f}",
                        len(sel), len(val_idx), len(val_subs)])
        print(f"  fold {subject}: map std={fold_topos[-1].std():.3e} bal_acc={ba:.3f} "
              f"n_train={len(sel)} n_val={len(val_idx)} time={time.time() - tf:.1f}s", flush=True)

    if not fold_topos:
        print(f"[NO DATA] {paradigm} step={step}", flush=True)
        marker.touch()
        return
    row = G.summarise_and_save(out_dir, paradigm, step, fold_topos, fold_subjects, bal_accs, ch_names)
    print(f"[DONE] {paradigm} step={step} BA={row['mean_bal_acc']} NFS={row['nfs']} [{row['ci_low']}, {row['ci_high']}] "
          f"p={row['p_perm']} folds={row['n_folds']} time={time.time() - t0:.0f}s", flush=True)
    if not max_folds:
        marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigms", default=",".join(G.PARADIGM_TO_NFS_NAME))
    ap.add_argument("--steps", default=",".join(map(str, ALL_STEPS)))
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--out", default="results_gat_v4")
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--readout", choices=["mean", "concat"], default="mean")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [s if s == "full" else int(s) for s in a.steps.split(",")]
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} threads={torch.get_num_threads()} out={out_dir}", flush=True)
    for p in a.paradigms.split(","):
        print(f"\n=== {p} ===", flush=True)
        for st in steps:
            try:
                run_paradigm_step(p, st, device, out_dir, a.max_folds, a.class_weight, a.readout)
            except Exception as exc:
                import traceback; traceback.print_exc()
                (out_dir / f"ERROR_{p}_step{st}.txt").write_text(repr(exc))


if __name__ == "__main__":
    main()
