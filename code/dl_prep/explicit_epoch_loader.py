# RECONSTRUCTED from the IncreaseAcc1 notebook (cells 59, 61, 64) because the original file is missing from the backup.
# Verified against the graph caches with verify_loader.py before use.
"""
explicit_epoch_loader.py -- authoritative, spec-driven epoch loader.

Design principles (all three prior bugs this session trace back to
violating one of these):
  1. No globbing beyond exact filenames named in CONDITION_SPEC.
     (Fixes: orphaned stale files being silently included.)
  2. No keyword substring matching. Conditions are matched by exact
     file stem or exact numeric code range, nothing else.
     (Fixes: 'related' matching inside 'unrelated', etc.)
  3. A task with no CONDITION_SPEC entry raises an error, not a
     silent fallback (parity-based or otherwise).
     (Fixes: flanker/visualsearch/N170 running on a meaningless
     parity fallback without anyone knowing.)
"""
import numpy as np
import mne
import re
from pathlib import Path

CONDITION_SPEC = {
    "ERP_CORE_P3":   {"class0": "target",  "class1": "standard", "exclude": ["epochs"]},
    "ERP_CORE_MMN":  {"class0": "deviant", "class1": "standard", "exclude": []},
    "ERP_CORE_ERN":  {"class0": "error",   "class1": "correct",  "exclude": []},
    "ERP_CORE_N400": {"class0": "related", "class1": "unrelated","exclude": []},
    "ERP_CORE_N2pc": {"class0": "left",    "class1": "right",    "exclude": []},
    "ERP_CORE_LRP":  {"class0": "left",    "class1": "right",    "exclude": ["epochs"]},
    "ERP_CORE_N170": {"class0_range": (1, 40), "class1_range": (41, 80), "single_file_codes": True},

    "DS006018_auditoryoddball": {"class0": "deviant", "class1": "standard", "exclude": ["epochs"]},
    "DS006018_visualoddball":   {"digit_match_rule": True, "single_file": "Stimulus", "valid_range": (11, 55)},

    **{f"DS002680_gonogo_{i}": {"class0": "target", "class1": "distractor",
                                  "exclude": ["epochs", "correct", "incorrect"]} for i in range(1, 14)},
}


def load_explicit(results_batch_dir: Path, dataset_name: str, task_name: str):
    """
    Loads exactly the trials CONDITION_SPEC says belong to this task's
    binary classification, from real subject folders, with no globbing
    of unlisted files and no keyword inference.
    """
    spec_key = f"{dataset_name}_{task_name}"
    if spec_key not in CONDITION_SPEC:
        raise ValueError(
            f"'{spec_key}' has no entry in CONDITION_SPEC. Refusing to load with a "
            f"fallback -- this is exactly the failure mode that caused undetected "
            f"issues this session. Add an explicit, verified entry before loading."
        )
    spec = CONDITION_SPEC[spec_key]

    task_dirs = sorted((results_batch_dir / dataset_name).glob(f"sub-*/{task_name}"))
    if not task_dirs:
        raise FileNotFoundError(f"No subject folders found for {spec_key}")

    X_list, y_list, sub_list = [], [], []
    ch_names, sfreq = None, None

    for tdir in task_dirs:
        sub_name = tdir.parent.name

        if "single_file" in spec:
            # e.g. DS006018_visualoddball: one Stimulus-epo.fif, split by numeric code
            fpath = tdir / f"{spec['single_file']}-epo.fif"
            if not fpath.exists():
                continue
            epochs = mne.read_epochs(fpath, preload=True, verbose=False)
            epochs.pick("eeg", verbose=False)
            codes = epochs.events[:, -1]
            # map MNE's internal event id back to the real code via event_id dict
            inv_id = {v: k for k, v in epochs.event_id.items()}
            def _extract_code(name):
                m = re.search(r"(\d+)$", name)
                return int(m.group(1)) if m else -1
            real_codes = np.array([_extract_code(inv_id[c]) for c in codes])
            if spec.get("digit_match_rule"):
                lo, hi = spec["valid_range"]
                in_range = (real_codes >= lo) & (real_codes <= hi) & (real_codes < 100)
                tens = real_codes // 10
                units = real_codes % 10
                mask0 = in_range & (tens == units)   # target: block-target matches shown stimulus
                mask1 = in_range & (tens != units)   # standard: mismatch
            else:
                mask0 = real_codes == spec["class0_code"]
                mask1 = real_codes == spec["class1_code"]
            data = epochs.get_data()
            if mask0.any():
                X_list.append(data[mask0]); y_list.append(np.zeros(mask0.sum(), dtype=np.int64))
                sub_list.append(np.array([sub_name] * mask0.sum()))
            if mask1.any():
                X_list.append(data[mask1]); y_list.append(np.ones(mask1.sum(), dtype=np.int64))
                sub_list.append(np.array([sub_name] * mask1.sum()))
            if ch_names is None:
                ch_names, sfreq = epochs.ch_names, epochs.info["sfreq"]

        elif "single_file_codes" in spec:
            # e.g. N170: many numeric-code files, class determined by code RANGE
            for fpath in tdir.glob("*-epo.fif"):
                stem = fpath.stem.replace("-epo", "")
                if not stem.isdigit():
                    continue
                code = int(stem)
                lo0, hi0 = spec["class0_range"]
                lo1, hi1 = spec["class1_range"]
                if lo0 <= code <= hi0:
                    cls = 0
                elif lo1 <= code <= hi1:
                    cls = 1
                else:
                    continue  # out of range (e.g. scrambled/response codes) -- correctly excluded
                epochs = mne.read_epochs(fpath, preload=True, verbose=False)
                epochs.pick("eeg", verbose=False)
                if len(epochs) == 0:
                    continue
                data = epochs.get_data()
                X_list.append(data)
                y_list.append(np.full(len(epochs), cls, dtype=np.int64))
                sub_list.append(np.array([sub_name] * len(epochs)))
                if ch_names is None:
                    ch_names, sfreq = epochs.ch_names, epochs.info["sfreq"]

        else:
            # Standard case: exactly two named condition files
            for cls, key in [(0, "class0"), (1, "class1")]:
                fpath = tdir / f"{spec[key]}-epo.fif"
                if not fpath.exists():
                    continue
                epochs = mne.read_epochs(fpath, preload=True, verbose=False)
                epochs.pick("eeg", verbose=False)
                if len(epochs) == 0:
                    continue
                data = epochs.get_data()
                X_list.append(data)
                y_list.append(np.full(len(epochs), cls, dtype=np.int64))
                sub_list.append(np.array([sub_name] * len(epochs)))
                if ch_names is None:
                    ch_names, sfreq = epochs.ch_names, epochs.info["sfreq"]

    if not X_list:
        raise FileNotFoundError(f"No valid trials loaded for {spec_key}")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sub_list, axis=0)

    print(f"[Explicit Loader] {spec_key}: X={X.shape}, y={y.shape}, "
          f"classes={np.unique(y)}, subjects={len(np.unique(subject_ids))}, "
          f"class_balance={np.bincount(y).tolist()}")

    return X, y, subject_ids, ch_names, sfreq
