# preprocessing_batch/erpcore_pipeline.py

from pathlib import Path

import mne

from config import ERP_CORE_ROOT, RESULTS_ROOT
from common_preprocessing import preprocess_raw, compute_grand_averages
from dataset_registry import ERP_CORE_TASKS
from event_registry import ERP_CORE as ERP_CORE_EVENTS


def iter_erpcore_files(root: Path):
    """
    Yield (subject_id, task, eeg_file) for all ERP CORE recordings.
    """
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for ses_dir in sorted(subj_dir.glob("ses-*")):
            task = ses_dir.name.replace("ses-", "")
            if task not in ERP_CORE_TASKS:
                continue

            eeg_files = sorted(ses_dir.rglob("*_eeg.set"))
            if not eeg_files:
                continue

            for eeg_file in eeg_files:
                yield subject, task, eeg_file


def run_erpcore_preprocessing(overwrite: bool = False) -> None:
    out_root = RESULTS_ROOT / "ERP_CORE"
    out_root.mkdir(parents=True, exist_ok=True)

    for subject, task, eeg_file in iter_erpcore_files(ERP_CORE_ROOT):
        output_dir = out_root / subject / task
        try:
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
            preprocess_raw(
                raw=raw,
                output_dir=output_dir,
                dataset_name="ERP_CORE",
                subject=subject,
                task=task,
                run_id=task,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] ERP_CORE {subject} {task} {eeg_file}: {exc}")

    # Grand averages across subjects
    compute_grand_averages("ERP_CORE", ERP_CORE_EVENTS)


if __name__ == "__main__":
    run_erpcore_preprocessing(overwrite=False)