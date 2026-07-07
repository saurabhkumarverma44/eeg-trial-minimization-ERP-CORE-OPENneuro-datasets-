# preprocessing_batch/ds002680_pipeline.py

from pathlib import Path

import mne

from config import DS002680_ROOT, RESULTS_ROOT
from common_preprocessing import preprocess_raw, compute_grand_averages
from event_registry import DS002680 as DS002680_EVENTS


def iter_ds002680_files(root: Path):
    """
    Yield (subject_id, task, eeg_file, run_id) for ds002680.
    Assumes BIDS-like naming with 'task-gonogo' and 'run-<n>' in filename.
    """
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for ses_dir in sorted(subj_dir.glob("ses-*")):
            eeg_files = sorted(ses_dir.rglob("*_eeg.set"))
            for eeg_file in eeg_files:
                name = eeg_file.name
                if "task-gonogo" not in name:
                    continue
                task = "gonogo"
                run_id = "run-1"
                if "run-" in name:
                    run_id = name.split("run-")[1].split("_")[0]
                yield subject, task, eeg_file, run_id


def run_ds002680_preprocessing(overwrite: bool = False) -> None:
    out_root = RESULTS_ROOT / "DS002680"
    out_root.mkdir(parents=True, exist_ok=True)

    for subject, task, eeg_file, run_id in iter_ds002680_files(DS002680_ROOT):
        output_dir = out_root / subject / f"{task}_{run_id}"
        try:
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
            preprocess_raw(
                raw=raw,
                output_dir=output_dir,
                dataset_name="DS002680",
                subject=subject,
                task=task,
                run_id=run_id,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] DS002680 {subject} {task} {run_id} {eeg_file}: {exc}"
            )

    compute_grand_averages("DS002680", DS002680_EVENTS)


if __name__ == "__main__":
    run_ds002680_preprocessing(overwrite=False)