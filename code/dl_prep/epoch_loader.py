import os
import glob
import numpy as np
import mne

def map_event_ids_to_binary(epochs, task_name=None):
    """
    Parses MNE event_id dictionary to map raw trigger codes into binary/grouped ERP classes (0 vs 1).
    """
    events = epochs.events[:, -1]
    event_id = epochs.event_id

    if not event_id:
        unique_events = np.unique(events)
        half = len(unique_events) // 2
        label_map = {ev: (0 if idx < half else 1) for idx, ev in enumerate(unique_events)}
        return np.array([label_map[e] for e in events], dtype=np.int64)

    # Keywords for Class 0 vs Class 1 across ERP CORE, ds006018, ds002680
    class0_keywords = ['target', 'face', 'related', 'deviant', 'error', 'left', 'congruent', 'go']
    class1_keywords = ['standard', 'non-target', 'nontarget', 'car', 'unrelated', 'correct', 'right', 'incongruent', 'nogo', 'no-go']

    label_map = {}
    for name, code in event_id.items():
        name_lower = name.lower()
        if any(kw in name_lower for kw in class0_keywords):
            label_map[code] = 0
        elif any(kw in name_lower for kw in class1_keywords):
            label_map[code] = 1
        else:
            # Fallback grouping by code value parity
            label_map[code] = 0 if (code % 2 == 1) else 1

    mapped_y = np.array([label_map.get(e, 0) for e in events], dtype=np.int64)
    return mapped_y

def discover_tasks(dataset_dir):
    task_dirs = set()
    sub_dirs = glob.glob(os.path.join(dataset_dir, "sub-*"))
    for sub in sub_dirs:
        for item in os.listdir(sub):
            item_path = os.path.join(sub, item)
            if os.path.isdir(item_path) and item != "grand_averages":
                task_dirs.add(item)
    return sorted(list(task_dirs))

def load_dataset_epochs(dataset_dir, task_name=None):
    if task_name and task_name.lower() != "all":
        pattern = os.path.join(dataset_dir, "sub-*", task_name, "*-epo.fif")
        fif_files = glob.glob(pattern, recursive=True)
        if not fif_files:
            pattern = os.path.join(dataset_dir, "sub-*", f"*{task_name}*", "*-epo.fif")
            fif_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(dataset_dir, "sub-*", "*", "*-epo.fif")
        fif_files = glob.glob(pattern, recursive=True)
        if not fif_files:
            pattern = os.path.join(dataset_dir, "sub-*", "*-epo.fif")
            fif_files = glob.glob(pattern, recursive=True)

    print(f"[Phase 3 Loader] Dataset Dir: {dataset_dir} | Task: {task_name or 'All'}")
    print(f"[Phase 3 Loader] Found {len(fif_files)} epoch files.")

    X_list, y_list, sub_id_list = [], [], []
    ch_names, sfreq = None, None

    for fpath in sorted(fif_files):
        parts = fpath.split(os.sep)
        sub_candidates = [p for p in parts if p.startswith("sub-")]
        sub_name = sub_candidates[0] if sub_candidates else "unknown_sub"

        try:
            epochs = mne.read_epochs(fpath, preload=True, verbose=False)
            if len(epochs) == 0:
                continue

            epochs.pick("eeg", verbose=False)
            data = epochs.get_data().astype(np.float32)
            labels = map_event_ids_to_binary(epochs, task_name=task_name)
            
            if ch_names is None:
                ch_names = epochs.ch_names
                sfreq = epochs.info['sfreq']

            n_trials = data.shape[0]
            if n_trials > 0:
                X_list.append(data)
                y_list.append(labels)
                sub_id_list.append(np.array([sub_name] * n_trials, dtype=str))
        except Exception as e:
            print(f"[Warning] Error reading {fpath}: {e}")

    if not X_list:
        raise FileNotFoundError(f"No valid epoch files found for dataset {dataset_dir} with task={task_name}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sub_id_list, axis=0)

    print(f"[Phase 3 Data Prep] Loaded Shape: X={X.shape}, y={y.shape}, Classes={len(np.unique(y))}, Subjects={len(np.unique(subject_ids))}")
    return X, y, subject_ids, ch_names, sfreq
