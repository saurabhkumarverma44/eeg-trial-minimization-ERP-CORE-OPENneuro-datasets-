# preprocessing_batch/erp_pipeline.py
# Steps 2.4–2.7: Epoch Segmentation, Baseline Correction,
#                Epoch Rejection, Grand Average ERP + Topography

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths & parameters (matching your config.py)
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results_batch")
TMIN         = -0.2          # epoch start (s)
TMAX         =  0.8          # epoch end   (s)
BASELINE     = (None, 0.0)   # −200 to 0 ms
REJECT_UV    = 100e-6        # 100 µV peak-to-peak (per your table 2.6)
MIN_EPOCHS   = 30            # flag if below this per condition
FLAG_REJECT  = 0.30          # flag if >30% epochs rejected
TOPO_TIMES   = [0.3, 0.4, 0.5]

# ──────────────────────────────────────────────────────────────────────────────
# Event mappings (identical to your event_registry.py)
# ──────────────────────────────────────────────────────────────────────────────

ERP_CORE_EVENTS = {
    "P3":   {"target":   [11,22,33,44,55],
             "standard": [12,13,14,15,21,23,24,25,31,32,34,35,
                          41,42,43,45,51,52,53,54,201,202]},
    "MMN":  {"deviant":  [70], "standard": [80]},
    "ERN":  {"correct":  [111,121], "error": [212,221,222]},
    "N400": {"related":  [111,112,121,122], "unrelated": [211,212,221,222]},
    "N2pc": {"left":     [111,112,121,122], "right": [211,212,221,222]},
    "LRP":  {"left":     [11,12,21,22],     "right": [111,121,212,221,222]},
    "N170": {},   # all events
}

DS006018_EVENTS = {
    "auditoryoddball": {"deviant": [70], "standard": [80]},
    "visualoddball":   {"deviant": [70], "standard": [80]},
    "flanker":     {},
    "visualsearch": {},
}

DS002680_EVENTS = {
    "gonogo": {
        "target":      [2],
        "distractor":  [1],
        "correct":     [3],
        "incorrect":   [4],
    },
}

DATASET_EVENTS = {
    "ERP_CORE": ERP_CORE_EVENTS,
    "DS006018": DS006018_EVENTS,
    "DS002680": DS002680_EVENTS,
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

log_dir = RESULTS_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_dir / "erp_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("erp_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# QC CSV helper
# ──────────────────────────────────────────────────────────────────────────────

QC_FIELDS = [
    "dataset", "subject", "task", "run",
    "condition", "n_kept", "n_total", "reject_rate", "flagged",
]

def _write_qc(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QC_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# Core per-recording function  (2.4 → 2.6)
# ──────────────────────────────────────────────────────────────────────────────

def run_epochs_erp(
    cleaned_fif: Path,
    output_dir: Path,
    dataset_name: str,
    subject: str,
    task: str,
    run_id: str,
    overwrite: bool = False,
) -> Optional[Dict[str, mne.Evoked]]:
    """
    Load cleaned raw, epoch, baseline-correct, reject, compute ERPs.
    Returns dict of condition → Evoked, or None if skipped / no events.
    """
    done_flag = output_dir / "ERP_DONE"
    if done_flag.exists() and not overwrite:
        log.info(f"[SKIP] {dataset_name} {subject} {task} {run_id}")
        return None

    if not cleaned_fif.exists():
        log.warning(f"[MISSING] {cleaned_fif} — skipping")
        return None

    log.info(f"[START] {dataset_name} {subject} {task} {run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = mne.io.read_raw_fif(cleaned_fif, preload=True, verbose="ERROR")
    except Exception as exc:
        log.error(
            f"[CORRUPT] {dataset_name} {subject} {task} {run_id} "
            f"— failed to read {cleaned_fif}: {exc}"
        )
        return None

    # ------------------------------------------------------------------
    # 2.4  Epoch segmentation
    # ------------------------------------------------------------------
    event_map = DATASET_EVENTS.get(dataset_name, {}).get(task, {})

    try:
        all_events, all_event_id = mne.events_from_annotations(
            raw, verbose="ERROR"
        )
    except Exception as exc:
        log.warning(f"[WARN] events_from_annotations failed: {exc}")
        return None

    present_codes = set(all_event_id.values())

    # Build reduced event_id with only codes present in this recording
    epoch_event_id: Dict[str, int] = {}
    for condition, codes in event_map.items():
        for code in codes:
            if code in present_codes:
                label = f"{condition}"
                # MNE epoch_event_id maps label→code (int); if multiple codes
                # per condition, merge by naming them label/code
                epoch_event_id[f"{condition}/{code}"] = code

    # Fallback: if no mapping or empty map, epoch on every event
    if not epoch_event_id:
        epoch_event_id = all_event_id

    if not epoch_event_id:
        log.info(f"[INFO] No events for {dataset_name} {subject} {task} — skip")
        return None

    reject = {"eeg": REJECT_UV}

    # 2.4 + 2.5: epoch + baseline correction
    epochs = mne.Epochs(
        raw,
        all_events,
        event_id=epoch_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,          # −200 to 0 ms
        reject=reject,              # 2.6 peak-to-peak rejection
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 2.6  Epoch rejection statistics
    # ------------------------------------------------------------------
    qc_csv = RESULTS_ROOT / dataset_name / "qc_epochs.csv"
    evokeds: Dict[str, mne.Evoked] = {}

    # Group sub-condition labels back to parent condition for ERP
    from collections import defaultdict
    condition_epochs: Dict[str, list] = defaultdict(list)
    condition_n_total: Dict[str, int] = defaultdict(int)
    for key in epochs.event_id:
        parent = key.split("/")[0]
        try:
            ep = epochs[key]
            # Track total attempted events for this sub-condition even if
            # all of them were rejected (drop_log length == events found,
            # regardless of how many survived).
            condition_n_total[parent] += len(ep.drop_log)
            if len(ep) > 0:
                condition_epochs[parent].append(ep)
        except Exception:
            pass

    for condition in list(condition_n_total.keys()):
        ep_list = condition_epochs.get(condition, [])

        if not ep_list:
            # Every event for this condition was rejected (0 kept) —
            # concatenate_epochs cannot handle an all-empty list, so
            # log it as a fully-rejected condition and move on.
            n_total = condition_n_total[condition]
            n_kept = 0
            reject_rate = 1.0 if n_total > 0 else 0.0
            flagged = True
            log.warning(
                f"  {condition}: 0/{n_total} kept (100.0% rejected) "
                f"⚠ FLAGGED — skipping ERP"
            )
            _write_qc(qc_csv, {
                "dataset":      dataset_name,
                "subject":      subject,
                "task":         task,
                "run":          run_id,
                "condition":    condition,
                "n_kept":       n_kept,
                "n_total":      n_total,
                "reject_rate":  f"{reject_rate:.4f}",
                "flagged":      int(flagged),
            })
            continue

        ep_cat = mne.concatenate_epochs(ep_list)
        n_total  = condition_n_total[condition]
        n_kept   = len(ep_cat)
        n_dropped = n_total - n_kept
        reject_rate = n_dropped / n_total if n_total > 0 else 0.0
        flagged  = (reject_rate > FLAG_REJECT) or (n_kept < MIN_EPOCHS)

        log.info(
            f"  {condition}: {n_kept}/{n_total} kept "
            f"({reject_rate*100:.1f}% rejected) "
            f"{'⚠ FLAGGED' if flagged else ''}"
        )

        _write_qc(qc_csv, {
            "dataset":      dataset_name,
            "subject":      subject,
            "task":         task,
            "run":          run_id,
            "condition":    condition,
            "n_kept":       n_kept,
            "n_total":      n_total,
            "reject_rate":  f"{reject_rate:.4f}",
            "flagged":      int(flagged),
        })

        if n_kept == 0:
            log.warning(f"  [WARN] 0 epochs for {condition} — skipping ERP")
            continue

        # 2.4 epoch shape log
        log.info(
            f"  Epoch shape ({condition}): "
            f"{ep_cat.get_data().shape}  "
            f"(n_epochs × n_channels × n_times)"
        )

        # Save epochs
        ep_cat.save(
            output_dir / f"{condition}-epo.fif",
            overwrite=True,
        )

        # ------------------------------------------------------------------
        # 2.7  ERP (average across epochs)
        # ------------------------------------------------------------------
        evoked = ep_cat.average()
        evoked.save(output_dir / f"{condition}-ave.fif", overwrite=True)
        evokeds[condition] = evoked

        # ERP butterfly plot
        fig = evoked.plot(
            spatial_colors=True, show=False,
            titles={"eeg": f"{dataset_name} {subject} {task} {condition}"},
        )
        fig.savefig(output_dir / f"{condition}_erp.png", dpi=150)
        plt.close(fig)

        # Topomap at 300 / 400 / 500 ms
        try:
            fig = evoked.plot_topomap(
                times=TOPO_TIMES, ch_type="eeg", show=False,
            )
            fig.savefig(output_dir / f"{condition}_topomap.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            log.warning(f"  topomap failed ({condition}): {exc}")

    # Difference wave for oddball/P3 designs
    if "target" in evokeds and "standard" in evokeds:
        diff = mne.combine_evoked(
            [evokeds["target"], evokeds["standard"]], weights=[1, -1]
        )
        diff.save(output_dir / "difference-ave.fif", overwrite=True)
        fig = diff.plot(spatial_colors=True, show=False,
                        titles={"eeg": f"{dataset_name} {subject} {task} diff"})
        fig.savefig(output_dir / "difference_wave.png", dpi=150)
        plt.close(fig)
        try:
            fig = diff.plot_topomap(
                times=TOPO_TIMES, ch_type="eeg", show=False,
            )
            fig.savefig(output_dir / "difference_topomap.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    if evokeds:
        done_flag.touch()
        log.info(f"[DONE] {dataset_name} {subject} {task} {run_id}")

    return evokeds if evokeds else None


# ──────────────────────────────────────────────────────────────────────────────
# Grand average helper  (2.7)
# ──────────────────────────────────────────────────────────────────────────────

def compute_grand_averages(dataset_name: str) -> None:
    """
    Scan all subject/task/condition-ave.fif files and compute grand averages.
    Output: results_batch/<DATASET>/grand_averages/<task>/<condition>_grand-ave.fif
    """
    root = RESULTS_ROOT / dataset_name
    if not root.exists():
        return

    # Discover all available tasks from folder structure
    tasks = set()
    for ave_file in root.rglob("*-ave.fif"):
        parts = ave_file.relative_to(root).parts
        if len(parts) >= 2:
            tasks.add(parts[1])   # sub-XXX / task / file

    for task in sorted(tasks):
        # For each condition in this task across all subjects
        condition_files: Dict[str, List[Path]] = {}
        for ave_file in sorted(root.glob(f"sub-*/{task}/*-ave.fif")):
            cond = ave_file.stem.replace("-ave", "")
            if cond not in condition_files:
                condition_files[cond] = []
            condition_files[cond].append(ave_file)

        for condition, files in condition_files.items():
            if len(files) < 2:
                continue

            try:
                evokeds = [
                    mne.read_evokeds(f, condition=0, verbose="ERROR")
                    for f in files
                ]
                grand = mne.grand_average(evokeds)

                out_dir = root / "grand_averages" / task
                out_dir.mkdir(parents=True, exist_ok=True)

                grand.save(
                    out_dir / f"{condition}_grand-ave.fif",
                    overwrite=True,
                )

                # Grand average ERP plot
                fig = grand.plot(
                    spatial_colors=True, show=False,
                    titles={"eeg": f"Grand Average — {dataset_name} {task} {condition} "
                          f"(N={len(files)})"},
                )
                fig.savefig(out_dir / f"{condition}_grand_erp.png", dpi=150)
                plt.close(fig)

                # Grand average topomap
                try:
                    fig = grand.plot_topomap(
                        times=TOPO_TIMES, ch_type="eeg", show=False,
                    )
                    fig.savefig(
                        out_dir / f"{condition}_grand_topomap.png", dpi=150
                    )
                    plt.close(fig)
                except Exception as exc:
                    log.warning(
                        f"  Grand topomap failed "
                        f"{dataset_name} {task} {condition}: {exc}"
                    )

                log.info(
                    f"[GA] {dataset_name} {task} {condition} "
                    f"— grand avg of {len(files)} subjects saved"
                )

            except Exception as exc:
                log.warning(
                    f"[GA WARN] {dataset_name} {task} {condition}: {exc}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific runners
# ──────────────────────────────────────────────────────────────────────────────

def run_erpcore(overwrite: bool = False) -> None:
    root = RESULTS_ROOT / "ERP_CORE"
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for task_dir in sorted(subj_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            cleaned = task_dir / "cleaned_raw.fif"
            try:
                run_epochs_erp(
                    cleaned_fif=cleaned,
                    output_dir=task_dir,
                    dataset_name="ERP_CORE",
                    subject=subject,
                    task=task,
                    run_id=task,
                    overwrite=overwrite,
                )
            except Exception as exc:
                log.error(
                    f"[FATAL-SKIP] ERP_CORE {subject} {task}: {exc}"
                )
    compute_grand_averages("ERP_CORE")


def run_ds006018(overwrite: bool = False) -> None:
    root = RESULTS_ROOT / "DS006018"
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for task_dir in sorted(subj_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            cleaned = task_dir / "cleaned_raw.fif"
            try:
                run_epochs_erp(
                    cleaned_fif=cleaned,
                    output_dir=task_dir,
                    dataset_name="DS006018",
                    subject=subject,
                    task=task,
                    run_id=task,
                    overwrite=overwrite,
                )
            except Exception as exc:
                log.error(
                    f"[FATAL-SKIP] DS006018 {subject} {task}: {exc}"
                )
    compute_grand_averages("DS006018")


def run_ds002680(overwrite: bool = False) -> None:
    root = RESULTS_ROOT / "DS002680"
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for run_dir in sorted(subj_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name   # e.g. gonogo_run-1
            task = "gonogo"
            cleaned = run_dir / "cleaned_raw.fif"
            try:
                run_epochs_erp(
                    cleaned_fif=cleaned,
                    output_dir=run_dir,
                    dataset_name="DS002680",
                    subject=subject,
                    task=task,
                    run_id=run_id,
                    overwrite=overwrite,
                )
            except Exception as exc:
                log.error(
                    f"[FATAL-SKIP] DS002680 {subject} {run_id}: {exc}"
                )
    compute_grand_averages("DS002680")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 2 STEPS 2.4–2.7: ERP PIPELINE")
    log.info("=" * 60)

    run_erpcore(overwrite=False)
    run_ds006018(overwrite=False)
    run_ds002680(overwrite=False)

    log.info("=" * 60)
    log.info("ALL DONE")
    log.info("=" * 60)