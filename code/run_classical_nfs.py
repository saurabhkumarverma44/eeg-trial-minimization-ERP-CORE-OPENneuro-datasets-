#!/usr/bin/env python3
"""
run_classical_nfs.py  (run from D:\\erp_work)

Corrected classical LOSO (Random Forest, XGBoost) on the features built by build_features.py.

Protocol (same for every model in the revised paper):
  * N is the number of POOLED training trials, drawn stratified from all training subjects
    (the old classical code used N per epoch file per subject, which is not comparable to the deep nets)
  * scaler fit on the training trials only
  * class imbalance handled by class weights (RF class_weight=balanced, XGB sample weights)
  * metric: balanced accuracy on ALL trials of the held-out subject; folds whose test subject has one
    class are recorded as undefined and counted
  * NFS: feature importances of the band-power block -> channel map (channel-major reshape),
    one map per fold, subject-level bootstrap CI, electrode-permutation p, reference aligned by name

Usage
  python run_classical_nfs.py --paradigms ERP_CORE_P3 --steps 50 --max-folds 3 --out results_cls_timing
  python run_classical_nfs.py --steps 5,10,20,30,50,75,100,150,200 --n-jobs 6
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_classical_ml.nfs_fixed import (importance_to_topography, subject_bootstrap_nfs,  # noqa: E402
                                           permutation_p, align_reference)
import run_gat_nfs as G  # noqa: E402  (reference loading, name map, notes)

ALL_STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def sample_pooled(y, n, rng):
    """Stratified draw of n trials from y (proportional to class frequency, >=1 per class)."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    for c, q in zip(classes, quota):
        pool = np.where(y == c)[0]
        idx.append(rng.choice(pool, size=min(q, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def make_model(name, seed):
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                      n_jobs=1, random_state=seed)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8,
                         n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0)


def one_fold(X, y, subjects, test_subject, step, model_name, blocks, n_ch, seed):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    te = subjects == test_subject
    tr_idx = np.where(~te)[0]
    rng = np.random.default_rng(seed)
    sel = tr_idx[sample_pooled(y[tr_idx], None if step == "full" else int(step), rng)]
    ytr = y[sel]
    if len(np.unique(ytr)) < 2:
        return test_subject, np.nan, None, "train_one_class"
    sc = StandardScaler().fit(X[sel])
    Xtr, Xte = sc.transform(X[sel]), sc.transform(X[te])
    m = make_model(model_name, seed)
    if model_name == "XGB":
        cw = len(ytr) / (2.0 * np.bincount(ytr, minlength=2).clip(1))
        m.fit(Xtr, ytr, sample_weight=cw[ytr])
    else:
        m.fit(Xtr, ytr)
    yte = y[te]
    if len(np.unique(yte)) < 2:
        ba, note = np.nan, "test_one_class"
    else:
        ba, note = float(balanced_accuracy_score(yte, m.predict(Xte))), ""
    topo = importance_to_topography(np.asarray(m.feature_importances_, float), blocks, n_ch)
    return test_subject, ba, topo, note


def append_row(path, fields, row):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def run(paradigm, step, model_name, feat_dir, out_dir, n_jobs, max_folds):
    from joblib import Parallel, delayed
    d = feat_dir / paradigm
    X = np.load(d / "X.npy"); y = np.load(d / "y.npy"); subjects = np.load(d / "subjects.npy")
    meta = json.loads((d / "meta.json").read_text())
    ch, blocks = meta["ch_names"], meta["feature_blocks"]
    subs = sorted(set(subjects))
    if max_folds:
        subs = subs[:max_folds]
    seed0 = 1000 + (0 if step == "full" else int(step))
    res = Parallel(n_jobs=n_jobs)(delayed(one_fold)(X, y, subjects, s, step, model_name, blocks, len(ch), seed0 + i)
                                  for i, s in enumerate(subs))
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, ba, _, note in res:
        append_row(out_dir / "fold_accuracy_v2.csv", ["paradigm", "model", "step", "subject", "bal_acc", "note"],
                   dict(paradigm=paradigm, model=model_name, step=step, subject=s,
                        bal_acc="" if np.isnan(ba) else f"{ba:.4f}", note=note))
    bas = np.array([r[1] for r in res], float)
    topos = [r[2] for r in res if r[2] is not None]
    row = dict(paradigm=paradigm, model=model_name, step=step, n_folds=len(res),
               n_valid=int(np.isfinite(bas).sum()), n_undefined=int(np.isnan(bas).sum()),
               mean_bal_acc=f"{np.nanmean(bas):.4f}", nfs="", ci_low="", ci_high="", p_perm="",
               map_min_std="", note=G.NOTES.get(paradigm, ""))
    if topos:
        T = np.stack(topos)
        mp = out_dir / "maps" / paradigm
        mp.mkdir(parents=True, exist_ok=True)
        np.save(mp / f"{model_name}_step_{step}_foldmaps.npy", T)
        row["map_min_std"] = f"{T.std(axis=1).min():.3e}"
        ref = G.load_reference(G.PARADIGM_TO_NFS_NAME[paradigm]) if paradigm in G.PARADIGM_TO_NFS_NAME else None
        if ref is not None and T.std(axis=1).min() > 1e-12:
            rv = align_reference(ref[0], ref[1], ch)
            p, lo, hi = subject_bootstrap_nfs(T, rv)
            _, pp = permutation_p(T.mean(axis=0), rv)
            row.update(nfs=f"{p:.4f}", ci_low=f"{lo:.4f}", ci_high=f"{hi:.4f}", p_perm=f"{pp:.4f}")
    fields = ["paradigm", "model", "step", "n_folds", "n_valid", "n_undefined", "mean_bal_acc",
              "nfs", "ci_low", "ci_high", "p_perm", "map_min_std", "note"]
    append_row(out_dir / "classical_v2.csv", fields, row)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigms", default=",".join(G.PARADIGM_TO_NFS_NAME))
    ap.add_argument("--steps", default=",".join(map(str, ALL_STEPS)))
    ap.add_argument("--models", default="RF,XGB")
    ap.add_argument("--features", default="features_v2")
    ap.add_argument("--out", default="results_cls_v2")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--max-folds", type=int, default=0)
    a = ap.parse_args()
    for p in a.paradigms.split(","):
        for st in a.steps.split(","):
            st = st if st == "full" else int(st)
            for mn in a.models.split(","):
                t = time.time()
                r = run(p, st, mn, Path(a.features), Path(a.out), a.n_jobs, a.max_folds)
                print(f"{p:14s} {mn:3s} N={str(st):4s} BA={r['mean_bal_acc']} valid={r['n_valid']}/{r['n_folds']} "
                      f"NFS={r['nfs'] or '-'} [{r['ci_low']},{r['ci_high']}] p={r['p_perm'] or '-'} ({time.time()-t:.0f}s)", flush=True)
