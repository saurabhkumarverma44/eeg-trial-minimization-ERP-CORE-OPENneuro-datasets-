#!/usr/bin/env python3
"""
build_features.py  (run from D:\\erp_work)

Rebuilds the Stage 3 feature matrix (1,920 features per trial) for each paradigm directly from
the verified prepared datasets in results_dl_1\\datasets, because the backup does not contain the
corrected epoch files or the Stage 3 results.

The feature definitions are the ones in stage3_feature_extraction (copied constants, same maths):
  band_power  30 channels x 5 bands (relative power, Welch 1-40 Hz)     150
  plv_alpha / plv_beta   upper triangle of the 30x30 PLV matrix        435 each
  wpli_alpha / wpli_beta upper triangle of the 30x30 wPLI matrix       435 each
  entropy     normalised spectral entropy per channel                   30
Block order: band_power, plv_alpha, plv_beta, wpli_alpha, wpli_beta, entropy.

Differences from the old Stage 3, both deliberate:
  * NO z-scoring per file here. Normalisation is done later, using training data only.
  * Input comes from the prepared .pt files, whose only change from the raw epochs is one
    per-channel affine scaling. Every feature above is invariant to that scaling (ratios,
    phases, normalised entropy); --check-equivalence proves this on ERP_CORE_P3.

Usage
  python build_features.py --check-equivalence
  python build_features.py --paradigms ERP_CORE_P3,ERP_CORE_N170 --n-jobs 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "stage3_feature_extraction"))
from connectivity_core import plv_matrix, wpli_matrix, upper_triangle_flat  # noqa: E402

SRC = Path(r"E:\erp_trial_min_backup_20260626\erp_trial_min")
DATASET_DIR = SRC / "results_dl_1" / "datasets"

# constants copied from config3.py
FREQ_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}
TOTAL_BAND_RANGE = (1.0, 40.0)
CONNECTIVITY_BANDS = {"alpha": (8.0, 12.0), "beta": (13.0, 30.0)}
ENTROPY_PSD_FMIN, ENTROPY_PSD_FMAX, ENTROPY_NPERSEG = 1.0, 40.0, 128
BLOCK_ORDER = ["band_power", "plv_alpha", "plv_beta", "wpli_alpha", "wpli_beta", "entropy"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def band_power_block(data, sfreq, ch_names):
    """data (n, ch, T) -> (n, ch*5), channel-major (index = channel*5 + band)."""
    import mne
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(np.asarray(data, dtype=np.float64), info, verbose="ERROR")
    spectrum = epochs.compute_psd(method="welch", fmin=TOTAL_BAND_RANGE[0], fmax=TOTAL_BAND_RANGE[1], verbose="ERROR")
    psds, freqs = spectrum.get_data(return_freqs=True)
    tot_mask = (freqs >= TOTAL_BAND_RANGE[0]) & (freqs <= TOTAL_BAND_RANGE[1])
    total = _trapz(psds[:, :, tot_mask], freqs[tot_mask], axis=-1)
    total = np.where(total <= 0, np.nan, total)
    rel = np.zeros(psds.shape[:2] + (len(FREQ_BANDS),))
    for b, (lo, hi) in enumerate(FREQ_BANDS.values()):
        m = (freqs >= lo) & (freqs <= hi)
        rel[:, :, b] = _trapz(psds[:, :, m], freqs[m], axis=-1) / total
    return rel.reshape(rel.shape[0], -1)


def entropy_block(data, sfreq):
    from scipy.signal import welch
    n, c, t = data.shape
    nperseg = min(ENTROPY_NPERSEG, t)
    out = np.zeros((n, c))
    for e in range(n):
        freqs, psd = welch(data[e], fs=sfreq, nperseg=nperseg, axis=-1)
        m = (freqs >= ENTROPY_PSD_FMIN) & (freqs <= ENTROPY_PSD_FMAX)
        pb = psd[:, m]
        s = pb.sum(axis=-1, keepdims=True)
        p = pb / np.where(s <= 0, np.nan, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(p * np.log(np.where(p > 0, p, 1.0)), axis=-1)
        out[e] = ent / (np.log(pb.shape[-1]) if pb.shape[-1] > 1 else 1.0)
    return np.nan_to_num(out, nan=0.0)


def features_for_chunk(data, sfreq, ch_names):
    """data (n, ch, T) float -> (n, 1920) float32 in BLOCK_ORDER."""
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    blocks = {"band_power": band_power_block(data, sfreq, ch_names)}
    for kind, fn in (("plv", plv_matrix), ("wpli", wpli_matrix)):
        for band, (lo, hi) in CONNECTIVITY_BANDS.items():
            blocks[f"{kind}_{band}"] = np.stack(
                [upper_triangle_flat(fn(data[i], sfreq, lo, hi)) for i in range(n)])
    blocks["entropy"] = entropy_block(data, sfreq)
    out = np.concatenate([blocks[k] for k in BLOCK_ORDER], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def block_boundaries(n_ch):
    n_pairs = n_ch * (n_ch - 1) // 2
    sizes = {"band_power": n_ch * len(FREQ_BANDS), "plv_alpha": n_pairs, "plv_beta": n_pairs,
             "wpli_alpha": n_pairs, "wpli_beta": n_pairs, "entropy": n_ch}
    pos, out = 0, {}
    for k in BLOCK_ORDER:
        out[k] = [pos, pos + sizes[k]]
        pos += sizes[k]
    out["_total_features"] = pos
    return out


def load_payload(key):
    import torch
    return torch.load(DATASET_DIR / f"{key}_prepared.pt", mmap=True, weights_only=False)


def subject_arrays(payload):
    """Yield (subject, split, X ndarray (n,ch,T), y ndarray) for every subject, sorted by subject."""
    by_subject = {}
    for split in ("train", "val", "test"):
        X, y, subs = payload[split]
        subs = np.array([str(s) for s in subs])
        y = np.asarray(y)
        for s in np.unique(subs):
            idx = np.where(subs == s)[0]
            by_subject.setdefault(s, []).append((split, X, y, idx))
    for s in sorted(by_subject):
        for split, X, y, idx in by_subject[s]:
            yield s, split, np.asarray(X[idx]), y[idx], idx


def build_paradigm(key, out_root, n_jobs):
    from joblib import Parallel, delayed
    t0 = time.time()
    payload = load_payload(key)
    ch_names, sfreq = list(payload["ch_names"]), float(payload["sfreq"])
    chunks = list(subject_arrays(payload))
    print(f"{key}: {sum(len(c[3]) for c in chunks)} trials, {len(set(c[0] for c in chunks))} subjects, "
          f"{len(ch_names)} channels, sfreq={sfreq}", flush=True)
    feats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(features_for_chunk)(c[2], sfreq, ch_names) for c in chunks)
    X = np.concatenate(feats)
    y = np.concatenate([c[3] for c in chunks]).astype(np.int64)
    subjects = np.concatenate([[c[0]] * len(c[3]) for c in chunks])
    split = np.concatenate([[c[1]] * len(c[3]) for c in chunks])
    row_in_split = np.concatenate([c[4] for c in chunks])
    out = out_root / key
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "subjects.npy", subjects)
    np.save(out / "split.npy", split)
    np.save(out / "row_in_split.npy", row_in_split)
    (out / "meta.json").write_text(json.dumps({
        "paradigm": key, "ch_names": ch_names, "sfreq": sfreq,
        "feature_blocks": block_boundaries(len(ch_names)),
        "n_trials": int(len(y)), "class_counts": np.bincount(y).tolist(),
        "note": "features are NOT normalised; fit scalers on training folds only"}, indent=1))
    n_bad = int((~np.isfinite(X)).sum())
    print(f"[DONE] {key}: X={X.shape} nonfinite={n_bad} classes={np.bincount(y).tolist()} "
          f"time={time.time() - t0:.0f}s", flush=True)


def check_equivalence(key="ERP_CORE_P3", n_check=12):
    """Raw epochs (via the verified loader) vs the prepared payload: same trials, same features."""
    from dl_prep.explicit_epoch_loader import load_explicit
    ds, task = "ERP_CORE", key[len("ERP_CORE_"):]
    X_raw, y_raw, subs_raw, ch_names, sfreq = load_explicit(Path("results_batch"), ds, task)
    payload = load_payload(key)
    s0 = sorted(set(subs_raw))[0]
    raw = X_raw[np.array(subs_raw) == s0][:n_check].astype(np.float64)
    pay = [(sp, x, yy) for (s, sp, x, yy, _) in subject_arrays(payload) if s == s0]
    if not pay:  # first raw subject may not be in the payload; use the first payload subject
        s0 = next(subject_arrays(payload))[0]
        raw = X_raw[np.array(subs_raw) == s0][:n_check].astype(np.float64)
        pay = [(sp, x, yy) for (s, sp, x, yy, _) in subject_arrays(payload) if s == s0]
    Xp = np.concatenate([p[1] for p in pay]).astype(np.float64)

    def zs(a):  # per-channel standardise a single trial so scaling differences cancel
        a = a - a.mean(axis=-1, keepdims=True)
        return a / (a.std(axis=-1, keepdims=True) + 1e-12)

    match = []
    for r in raw:
        c = [np.mean(zs(r) * zs(p)) for p in Xp]
        match.append((int(np.argmax(c)), float(np.max(c))))
    print(f"subject {s0}: best signal correlation per trial: min={min(m[1] for m in match):.6f}")
    f_raw = features_for_chunk(raw, sfreq, ch_names)
    f_pay = features_for_chunk(Xp[[m[0] for m in match]], sfreq, ch_names)
    bounds = block_boundaries(len(ch_names))
    ok = min(m[1] for m in match) > 0.999
    for k in BLOCK_ORDER:
        a, b = bounds[k]
        diff = np.abs(f_raw[:, a:b] - f_pay[:, a:b]).max()
        corr = np.corrcoef(f_raw[:, a:b].ravel(), f_pay[:, a:b].ravel())[0, 1]
        print(f"  {k:11s} max|diff|={diff:.3e}  corr={corr:.6f}")
        ok &= corr > 0.999
    print("EQUIVALENT" if ok else "NOT EQUIVALENT: do not use the payload features")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigms", default="ERP_CORE_P3,ERP_CORE_N170,ERP_CORE_N400,ERP_CORE_ERN,ERP_CORE_N2pc,ERP_CORE_MMN")
    ap.add_argument("--out", default="features_v2")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--check-equivalence", action="store_true")
    a = ap.parse_args()
    if a.check_equivalence:
        check_equivalence()
    else:
        for k in a.paradigms.split(","):
            build_paradigm(k, Path(a.out), a.n_jobs)
