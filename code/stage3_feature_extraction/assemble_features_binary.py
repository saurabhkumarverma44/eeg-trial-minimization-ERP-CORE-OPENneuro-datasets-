
"""
assemble_features_binary.py -- Phase 3.7, rebuilt to assign real
binary labels via stage3_labeling.get_binary_label() (which wraps
the verified CONDITION_SPEC). Rows with no valid label (excluded
files, out-of-range codes) are dropped, not silently kept unlabeled.
"""
import csv, json, re
import numpy as np
import mne

import config3 as C
from stage3_utils import ensure_all_output_dirs, get_logger, iter_epoch_files
from stage3_labeling import get_binary_label

log = get_logger("phase37_assemble_binary")
CONN_BANDS = list(C.CONNECTIVITY_BANDS.keys())


def _load_feature_blocks(dataset, file_key):
    bp_path = C.BANDPOWER_DIR / dataset / f"{file_key}_bandpower.npy"
    ent_path = C.ENTROPY_DIR / dataset / f"{file_key}_entropy.npy"
    if not (bp_path.exists() and ent_path.exists()):
        return None
    blocks = {}
    bp = np.load(bp_path)
    blocks["band_power"] = bp.reshape(bp.shape[0], -1)
    blocks["entropy"] = np.load(ent_path)
    n_epochs = bp.shape[0]
    for band in CONN_BANDS:
        plv_path = C.PLV_DIR / dataset / f"{file_key}_{band}_plv_flat.npy"
        wpli_path = C.WPLI_DIR / dataset / f"{file_key}_{band}_wpli_flat.npy"
        if not (plv_path.exists() and wpli_path.exists()):
            return None
        plv, wpli = np.load(plv_path), np.load(wpli_path)
        n_epochs = min(n_epochs, plv.shape[0], wpli.shape[0])
        blocks[f"plv_{band}"], blocks[f"wpli_{band}"] = plv, wpli
    for k in blocks:
        blocks[k] = blocks[k][:n_epochs]
    return blocks, n_epochs


def _feature_vector(blocks, idx):
    parts = [blocks["band_power"][idx]]
    for band in CONN_BANDS: parts.append(blocks[f"plv_{band}"][idx])
    for band in CONN_BANDS: parts.append(blocks[f"wpli_{band}"][idx])
    parts.append(blocks["entropy"][idx])
    return np.concatenate(parts)


def _boundaries(blocks):
    boundaries, pos = {}, 0
    order = ["band_power"] + [f"plv_{b}" for b in CONN_BANDS] + [f"wpli_{b}" for b in CONN_BANDS] + ["entropy"]
    for name in order:
        size = blocks[name].shape[1]
        boundaries[name] = [pos, pos + size]
        pos += size
    boundaries["_total_features"] = pos
    return boundaries


def assemble_paradigm(dataset: str, task: str):
    paradigm_dir_name = f"{dataset}__{task}"
    refs = [r for r in iter_epoch_files(dataset) if r.task == task]
    if not refs:
        log.warning(f"[SKIP] {paradigm_dir_name}: no epoch files found")
        return

    feature_rows, rows_meta, block_boundaries = [], [], None
    n_excluded = 0

    for ref in refs:
        loaded = _load_feature_blocks(dataset, ref.key)
        if loaded is None:
            continue
        blocks, n_epochs = loaded

        try:
            epochs = mne.read_epochs(ref.path, preload=False, verbose="ERROR")
        except Exception as exc:
            log.warning(f"  [SKIP] {ref.stem}: read error {exc}")
            continue
        if len(epochs) == 0:
            continue

        inv_id = {v: k for k, v in epochs.event_id.items()}
        codes = epochs.events[:, -1][:n_epochs]

        if block_boundaries is None:
            block_boundaries = _boundaries(blocks)

        for i, code in enumerate(codes):
            raw_label = inv_id.get(code, "")
            m = re.search(r"(\d+)$", raw_label.replace(" ", ""))
            numeric_code = int(m.group(1)) if m else int(code)
            label = get_binary_label(dataset, task, ref.file_id, raw_label, numeric_code)
            if label is None:
                n_excluded += 1
                continue
            feature_rows.append(_feature_vector(blocks, i))
            rows_meta.append({"row_index": len(feature_rows)-1, "subject": ref.subject,
                               "file_id": ref.file_id, "epoch_index_in_subject": i, "label": label})

    if not feature_rows:
        log.warning(f"[SKIP] {paradigm_dir_name}: no labeled rows assembled (excluded={n_excluded})")
        return

    X = np.vstack(feature_rows)
    y = np.array([r["label"] for r in rows_meta], dtype=np.int64)

    out_dir = C.ML_DIR / paradigm_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    with (out_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index","subject","file_id","epoch_index_in_subject","label"])
        writer.writeheader()
        writer.writerows(rows_meta)

    with (out_dir / "feature_block_boundaries.json").open("w") as f:
        json.dump(block_boundaries, f, indent=2)

    class_counts = np.bincount(y, minlength=2).tolist()
    log.info(f"[OK] {paradigm_dir_name}: X={X.shape} y={y.shape} "
             f"class_balance={class_counts} excluded={n_excluded} "
             f"subjects={len(set(r['subject'] for r in rows_meta))}")


def run_assembly_binary():
    ensure_all_output_dirs()
    for dataset, tasks in [("ERP_CORE", ["P3","MMN","ERN","N400","N2pc","LRP","N170"]),
                            ("DS006018", ["auditoryoddball","visualoddball"]),
                            ("DS002680", [f"gonogo_{i}" for i in range(1,14)])]:
        for task in tasks:
            try:
                assemble_paradigm(dataset, task)
            except Exception as exc:
                log.warning(f"[ERROR] {dataset}_{task}: {exc}")
    log.info("Phase 3.7 (binary) assembly complete.")


if __name__ == "__main__":
    run_assembly_binary()
