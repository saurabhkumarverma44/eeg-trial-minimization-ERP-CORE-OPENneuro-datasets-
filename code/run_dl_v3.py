#!/usr/bin/env python3
"""
run_dl_v3.py  (run from D:\\erp_work)

CNN-1D and EEGNet LOSO under the revised protocol shared with RF/XGBoost/GAT:
  * N = number of POOLED training trials, drawn stratified by class from the training subjects
  * early stopping uses a SUBJECT-WISE validation set: ~15% of the training subjects (at least 2) are held
    out entirely; validation trials come only from them (up to 200, stratified); they do not count towards N
  * per-channel z-score fit on the N training trials only (as before)
  * model, optimiser, scheduler, epochs (30) and patience (8) unchanged from run_dl_loso.py
  * data come from the verified prepared datasets in results_dl_1\\datasets (the epoch files for ERN, N400,
    N2pc, MMN and LRP are missing from the backup). The prepared data differ from raw epochs by one constant
    per-channel scaling, which the per-channel z-score removes exactly.
  * every fold is written to fold_accuracy_dl_v3.csv, including undefined folds, so folds can be counted

Usage
  python run_dl_v3.py --paradigms ERP_CORE_P3 --steps 50 --models cnn1d --max-folds 2 --out results_dl_v3_timing
  python run_dl_v3.py --steps 5,20,50,100,200 --out results_dl_v3
  (two windows at once: add --threads 3 and give each window a different --models)
"""
import argparse, contextlib, csv, gc, io, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classical_nfs import sample_pooled  # noqa: E402

SRC = Path(r"E:\erp_trial_min_backup_20260626\erp_trial_min")
DATASET_DIR = SRC / "results_dl_1" / "datasets"
PARADIGMS = ["ERP_CORE_P3", "ERP_CORE_N170", "ERP_CORE_N400", "ERP_CORE_ERN", "ERP_CORE_N2pc", "ERP_CORE_MMN"]
STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200]
VAL_MAX = 200
FIELDS = ["paradigm", "model", "step", "n_folds", "n_valid", "n_undefined", "mean_acc", "mean_bal_acc", "mean_auc"]


CACHE = Path("dl_cache")


def load_all(key):
    """Return (X, y, subjects, n_classes) with X as a read-only memory map on disk.
    The first call per paradigm writes dl_cache\\<key>_X.npy sequentially (low RAM); later calls just map it,
    so RAM use stays small even for MMN (3 GB) and two runs can share it."""
    import torch
    CACHE.mkdir(exist_ok=True)
    fx, fm = CACHE / f"{key}_X.npy", CACHE / f"{key}_meta.npz"
    if not (fx.exists() and fm.exists()):
        p = torch.load(DATASET_DIR / f"{key}_prepared.pt", mmap=True, weights_only=False)
        n = sum(len(p[sp][1]) for sp in ("train", "val", "test"))
        X = np.lib.format.open_memmap(str(fx) + ".tmp.npy", mode="w+", dtype=np.float32,
                                      shape=(n,) + tuple(p["train"][0].shape[1:]))
        y, s, pos = [], [], 0
        for sp in ("train", "val", "test"):
            Xs, ys, ss = p[sp]
            for a in range(0, len(ys), 500):
                b = min(a + 500, len(ys))
                X[pos + a:pos + b] = Xs[a:b].numpy()
            y.append(np.asarray(ys)); s += [str(v) for v in ss]
            pos += len(ys)
        X.flush(); del X
        np.savez(fm, y=np.concatenate(y).astype(np.int64), s=np.array(s), nc=int(p["num_classes"]))
        os.replace(str(fx) + ".tmp.npy", fx)
        del p
    m = np.load(fm)
    return np.load(fx, mmap_mode="r"), m["y"], m["s"], int(m["nc"])


def build_model(name, in_channels, num_classes):
    if name == "cnn1d":
        from dl_models.cnn1d import ERPConv1DNet
        return ERPConv1DNet(in_channels=in_channels, num_classes=num_classes)
    from dl_models.eegnet import EEGNet
    return EEGNet(in_channels=in_channels, num_classes=num_classes)


def append_csv(path, fields, row):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(key, step, model_name, data, out_dir, device, max_folds):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from dl_models.trainer import Trainer, LabelSmoothingCrossEntropy
    from dl_models.loso_eval import evaluate_fold
    marker = out_dir / "markers" / f"{model_name}__{key}__step{step}.done"
    if marker.exists():
        print(f"[EXISTS] {model_name} {key} step={step}", flush=True)
        return
    (out_dir / "markers").mkdir(parents=True, exist_ok=True)
    X, y, subj, n_cls = data
    subjects = sorted(set(subj))
    if max_folds:
        subjects = subjects[:max_folds]
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    rows = []
    fold_csv = out_dir / "fold_accuracy_dl_v3.csv"
    ff = ["paradigm", "model", "step", "subject", "accuracy", "balanced_accuracy", "f1", "auc",
          "n_train", "n_val", "note"]
    for s in subjects:
        te = np.where(subj == s)[0]
        other = np.where(subj != s)[0]
        subs_other = np.array(sorted(set(subj[other])))
        val_subs = set(rng.choice(subs_other, size=max(2, int(round(0.15 * len(subs_other)))), replace=False).tolist())
        is_val = np.isin(subj[other], list(val_subs))
        pool_tr, pool_val = other[~is_val], other[is_val]
        sel = pool_tr[sample_pooled(y[pool_tr], None if step == "full" else int(step), rng)]
        vsel = pool_val[sample_pooled(y[pool_val], VAL_MAX, rng)]
        if len(np.unique(y[sel])) < 2 or len(np.unique(y[vsel])) < 2:
            append_csv(fold_csv, ff, dict(paradigm=key, model=model_name, step=step, subject=s,
                                          note="skipped_one_class_train_or_val"))
            continue
        Xtr, Xva, Xte = X[sel], X[vsel], X[te]
        m = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
        Xtr, Xva, Xte = (Xtr - m) / sd, (Xva - m) / sd, (Xte - m) / sd
        mk = lambda a, b, shuf: DataLoader(TensorDataset(torch.from_numpy(a).float(), torch.from_numpy(b).long(),
                                                         torch.zeros(len(b))), batch_size=32 if shuf else 64, shuffle=shuf)
        tl, vl, el = mk(Xtr, y[sel], True), mk(Xva, y[vsel], False), mk(Xte, y[te], False)
        model = build_model(model_name, X.shape[1], n_cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
        ck = out_dir / f"_tmp_{model_name}_{os.getpid()}.pt"
        tr = Trainer(model=model, optimizer=opt, criterion=LabelSmoothingCrossEntropy(0.1), scheduler=sch,
                     device=device, save_path=ck, patience=8, use_amp=False)
        with contextlib.redirect_stdout(io.StringIO()):
            tr.fit(tl, vl, epochs=30)
        model.load_state_dict(torch.load(ck, weights_only=True))
        met = evaluate_fold(model, el, device, use_amp=False)
        ck.unlink(missing_ok=True)
        rows.append(met)
        append_csv(fold_csv, ff, dict(paradigm=key, model=model_name, step=step, subject=s,
                                      accuracy=f"{met['accuracy']:.4f}",
                                      balanced_accuracy="" if np.isnan(met["balanced_accuracy"]) else f"{met['balanced_accuracy']:.4f}",
                                      f1=f"{met['f1']:.4f}", auc="" if np.isnan(met["auc"]) else f"{met['auc']:.4f}",
                                      n_train=len(sel), n_val=len(vsel),
                                      note="test_one_class" if met["degenerate_fold"] else ""))
    ba = np.array([r["balanced_accuracy"] for r in rows], float)
    auc = np.array([r["auc"] for r in rows], float)
    row = dict(paradigm=key, model=model_name, step=step, n_folds=len(subjects), n_valid=int(np.isfinite(ba).sum()),
               n_undefined=len(subjects) - int(np.isfinite(ba).sum()),
               mean_acc=f"{np.mean([r['accuracy'] for r in rows]):.4f}" if rows else "",
               mean_bal_acc=f"{np.nanmean(ba):.4f}" if rows else "", mean_auc=f"{np.nanmean(auc):.4f}" if rows else "")
    append_csv(out_dir / "dl_v3.csv", FIELDS, row)
    print(f"[DONE] {key:14s} {model_name:6s} N={str(step):4s} BA={row['mean_bal_acc']} AUC={row['mean_auc']} "
          f"valid={row['n_valid']}/{row['n_folds']} ({time.time() - t0:.0f}s)", flush=True)
    if not max_folds:
        marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigms", default=",".join(PARADIGMS))
    ap.add_argument("--steps", default=",".join(map(str, STEPS)))
    ap.add_argument("--models", default="cnn1d,eegnet")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results_dl_v3")
    ap.add_argument("--prepare-only", action="store_true", help="only build the dl_cache files, then exit")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.prepare_only:
        for key in a.paradigms.split(","):
            load_all(key)
            print(f"[CACHE READY] {key}", flush=True)
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} threads={torch.get_num_threads()} out={out}", flush=True)
    steps = [s if s == "full" else int(s) for s in a.steps.split(",")]
    for key in a.paradigms.split(","):
        print(f"\n=== {key} ===", flush=True)
        data = None
        gc.collect()  # release the previous paradigm before loading the next
        data = load_all(key)
        for st in steps:
            for mn in a.models.split(","):
                try:
                    run_one(key, st, mn, data, out, device, a.max_folds)
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    (out / f"ERROR_{mn}_{key}_{st}.txt").write_text(repr(exc))


if __name__ == "__main__":
    main()
