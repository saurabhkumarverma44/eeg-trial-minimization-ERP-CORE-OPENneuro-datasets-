from pathlib import Path
import json
import pandas as pd
import mne

ROOT = Path("data/raw")

records = []

datasets = {
    "ERP_CORE": {
        "path": ROOT / "erp_core",
        "example": next((ROOT / "erp_core").rglob("*.set"))
    },

    "ds006018": {
        "path": ROOT / "ds006018",
        "example": next((ROOT / "ds006018").rglob("*.vhdr"))
    },

    "ds002680": {
        "path": ROOT / "ds002680",
        "example": next((ROOT / "ds002680").rglob("*.set"))
    }
}

for dataset_name, info in datasets.items():

    dataset_path = info["path"]

    dataset_json = dataset_path / "dataset_description.json"

    row = {
        "Dataset": dataset_name,
        "Name": "NA",
        "BIDSVersion": "NA",
        "Subjects": 0,
        "Channels": "NA",
        "SamplingRate": "NA"
    }

    if dataset_json.exists():
        with open(dataset_json) as f:
            meta = json.load(f)

        row["Name"] = meta.get("Name", "NA")
        row["BIDSVersion"] = meta.get("BIDSVersion", "NA")

    subjects = [
        x.name
        for x in dataset_path.iterdir()
        if x.is_dir() and x.name.startswith("sub-")
    ]

    row["Subjects"] = len(subjects)

    eeg_file = info["example"]

    try:

        if eeg_file.suffix == ".vhdr":
            raw = mne.io.read_raw_brainvision(
                eeg_file,
                preload=False,
                verbose=False
            )

        else:
            raw = mne.io.read_raw_eeglab(
                eeg_file,
                preload=False,
                verbose=False
            )

        row["Channels"] = len(raw.ch_names)
        row["SamplingRate"] = raw.info["sfreq"]

    except Exception as e:
        print(dataset_name, e)

    records.append(row)

df = pd.DataFrame(records)

print(df)

audit_path = Path(
    "data/audit/dataset_audit.csv"
)

df.to_csv(
    audit_path,
    index=False
)

print("\nSaved:", audit_path)