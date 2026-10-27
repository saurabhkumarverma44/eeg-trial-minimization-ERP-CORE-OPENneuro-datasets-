"""
verify_cache.py  (run from D:\\erp_work)

Checks that each graph cache in results_dl_2 is complete and consistent with
the trusted prepared datasets in results_dl_1, and saves the electrode names.

For every paradigm it checks:
  1. the cache pickle loads (a truncated copy cannot be unpickled)
  2. the node features, edge index and edge attributes have consistent shapes and no NaN/inf
  3. the number of trials per (subject, class) equals the number in
     results_dl_1\\datasets\\<paradigm>_prepared.pt  (train + val + test together)
  4. the electrode names stored in the .pt file match the cache's channel count

It writes:
  results_gat\\cache_verification.csv
  results_gat\\ch_names\\<paradigm>.json     (electrode names in cache node order)

It never modifies the backup.
"""
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(r"E:\erp_trial_min_backup_20260626\erp_trial_min")
CACHE_DIR = SRC / "results_dl_2" / "graph_cache"
DATASET_DIR = SRC / "results_dl_1" / "datasets"
OUT_DIR = Path("results_gat")
NAMES_DIR = OUT_DIR / "ch_names"


def count_pairs(subject_ids, y):
    """Counter of (subject, class) -> number of trials."""
    return Counter(zip([str(s) for s in subject_ids], [int(v) for v in y]))


def compare_counts(cache_counts, ref_counts):
    """Return a list of human-readable differences (empty list means identical)."""
    diffs = []
    for key in sorted(set(cache_counts) | set(ref_counts)):
        a, b = cache_counts.get(key, 0), ref_counts.get(key, 0)
        if a != b:
            diffs.append(f"{key[0]} class{key[1]}: cache={a} dataset={b}")
    return diffs


def check_arrays(cache, sample_every=1):
    """Shape and finiteness checks. Returns a list of problems."""
    problems = []
    n = len(cache["node_features"])
    for k in ("edge_index", "edge_attr"):
        if len(cache[k]) != n:
            problems.append(f"{k} length {len(cache[k])} != {n}")
    if len(cache["y"]) != n or len(cache["subject_ids"]) != n:
        problems.append("y / subject_ids length differs from node_features")
    n_ch = int(cache["n_channels"])
    bad_shape = bad_nan = 0
    for i in range(0, n, sample_every):
        nf = cache["node_features"][i]
        if nf.shape != (n_ch, 5):
            bad_shape += 1
        if not np.isfinite(nf).all() or not np.isfinite(cache["edge_attr"][i]).all():
            bad_nan += 1
    if bad_shape:
        problems.append(f"{bad_shape} graphs with node_features shape != ({n_ch}, 5)")
    if bad_nan:
        problems.append(f"{bad_nan} graphs with NaN/inf")
    return problems


def main():
    import torch  # imported here so the helper functions can be tested without torch

    NAMES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pkl in sorted(CACHE_DIR.glob("*.pkl")):
        key = pkl.stem
        row = {"paradigm": key, "status": "", "n_graphs": "", "n_channels": "", "details": ""}
        try:
            with open(pkl, "rb") as f:
                cache = pickle.load(f)
        except Exception as exc:  # truncated or corrupt
            row.update(status="FAIL", details=f"cannot load pickle: {exc!r}")
            rows.append(row)
            print(f"[FAIL] {key}: {row['details']}")
            continue

        row["n_graphs"] = len(cache["node_features"])
        row["n_channels"] = int(cache["n_channels"])
        problems = check_arrays(cache)

        ds_path = DATASET_DIR / f"{key}_prepared.pt"
        if not ds_path.exists():
            problems.append(f"no reference dataset {ds_path.name}")
        else:
            payload = torch.load(ds_path, mmap=True, weights_only=False)
            ys, subs = [], []
            for split in ("train", "val", "test"):
                _, y_s, sub_s = payload[split]
                ys.append(y_s.numpy())
                subs.extend(list(sub_s))
            ref_counts = count_pairs(subs, np.concatenate(ys))
            cache_counts = count_pairs(cache["subject_ids"], cache["y"])
            diffs = compare_counts(cache_counts, ref_counts)
            if diffs:
                problems.append(f"{len(diffs)} (subject,class) counts differ, e.g. " + "; ".join(diffs[:3]))
            ch_names = list(payload["ch_names"])
            if len(ch_names) != int(cache["n_channels"]):
                problems.append(f"dataset has {len(ch_names)} channels, cache has {cache['n_channels']}")
            else:
                (NAMES_DIR / f"{key}.json").write_text(json.dumps(ch_names))
            del payload

        row["status"] = "OK" if not problems else "FAIL"
        row["details"] = " | ".join(problems)
        rows.append(row)
        print(f"[{row['status']}] {key}: graphs={row['n_graphs']} channels={row['n_channels']} {row['details']}")

    with (OUT_DIR / "cache_verification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "OK" for r in rows)
    print(f"\n{n_ok}/{len(rows)} caches verified. Table: {OUT_DIR / 'cache_verification.csv'}")


if __name__ == "__main__":
    main()
