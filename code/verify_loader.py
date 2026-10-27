"""
verify_loader.py  (run from D:\\erp_work)

Proves that the reconstructed dl_prep\\explicit_epoch_loader.py reproduces the data the
verified graph caches were built from. For each paradigm it loads the epochs with
load_explicit() and checks, against the cache in results_dl_2:
  * same number of trials
  * same number of trials for every (subject, class)
  * same electrode names (results_gat\\ch_names\\<paradigm>.json) and channel count

Usage:  python verify_loader.py ERP_CORE_P3 ERP_CORE_ERN
Reads epochs through the results_batch junction. Writes nothing.
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl_prep.explicit_epoch_loader import load_explicit  # noqa: E402

SRC = Path(r"E:\erp_trial_min_backup_20260626\erp_trial_min")
CACHE_DIR = SRC / "results_dl_2" / "graph_cache"
KNOWN = ["ERP_CORE", "DS006018", "DS002680"]


def split_key(key):
    for ds in KNOWN:
        if key.startswith(ds + "_"):
            return ds, key[len(ds) + 1:]
    raise ValueError(key)


def counts(subjects, y):
    return Counter(zip([str(s) for s in subjects], [int(v) for v in y]))


def main(keys):
    ok_all = True
    for key in keys:
        ds, task = split_key(key)
        X, y, subs, ch_names, sfreq = load_explicit(Path("results_batch"), ds, task)
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            cache = pickle.load(f)
        problems = []
        if X.shape[0] != len(cache["y"]):
            problems.append(f"trials: loader={X.shape[0]} cache={len(cache['y'])}")
        a, b = counts(subs, y), counts(cache["subject_ids"], cache["y"])
        diff = [f"{k}: loader={a.get(k, 0)} cache={b.get(k, 0)}" for k in sorted(set(a) | set(b)) if a.get(k, 0) != b.get(k, 0)]
        if diff:
            problems.append(f"{len(diff)} (subject,class) counts differ, e.g. " + "; ".join(diff[:3]))
        saved = Path("results_gat") / "ch_names" / f"{key}.json"
        if saved.exists() and json.loads(saved.read_text()) != list(ch_names):
            problems.append("electrode names differ from the saved list")
        if X.shape[1] != int(cache["n_channels"]):
            problems.append(f"channels: loader={X.shape[1]} cache={cache['n_channels']}")
        status = "OK" if not problems else "FAIL"
        ok_all &= not problems
        print(f"[{status}] {key}: trials={X.shape[0]} channels={X.shape[1]} sfreq={sfreq} {' | '.join(problems)}", flush=True)
        del X, cache
    print("\nALL MATCH" if ok_all else "\nMISMATCH FOUND: do not use the reconstructed loader yet")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ERP_CORE_P3"])
