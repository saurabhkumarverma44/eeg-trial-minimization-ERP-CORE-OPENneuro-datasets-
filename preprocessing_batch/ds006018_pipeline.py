# preprocessing_batch/ds006018_pipeline.py

from pathlib import Path

import mne

from config import DS006018_ROOT, RESULTS_ROOT
from common_preprocessing import preprocess_raw, compute_grand_averages
from dataset_registry import DS006018_TASKS
from event_registry import DS006018 as DS006018_EVENTS


def iter_ds006018_files(root: Path):
    """
    Yield (subject_id, task, eeg_file) for all ds006018 recordings.
    Assumes file pattern: sub-XXX_task-<task>_eeg.vhdr
    """
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        eeg_files = sorted(subj_dir.rglob("*.vhdr"))
        for eeg_file in eeg_files:
            name = eeg_file.name
            if "task-" not in name:
                continue
            task = name.split("task-")[1].split("_")[0]
            if task not in DS006018_TASKS:
                continue
            yield subject, task, eeg_file


def run_ds006018_preprocessing(overwrite: bool = False) -> None:
    out_root = RESULTS_ROOT / "DS006018"
    out_root.mkdir(parents=True, exist_ok=True)

    for subject, task, eeg_file in iter_ds006018_files(DS006018_ROOT):
        output_dir = out_root / subject / task
        try:
            raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
            preprocess_raw(
                raw=raw,
                output_dir=output_dir,
                dataset_name="DS006018",
                subject=subject,
                task=task,
                run_id=task,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] DS006018 {subject} {task} {eeg_file}: {exc}")

    compute_grand_averages("DS006018", DS006018_EVENTS)


if __name__ == "__main__":
    run_ds006018_preprocessing(overwrite=False)