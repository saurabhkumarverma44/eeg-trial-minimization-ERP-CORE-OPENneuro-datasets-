import mne
import pandas as pd
from pathlib import Path

# ==========================================================
# PHASE 2.4 EPOCH SEGMENTATION
# ERP CORE P3 Subject 021
# ==========================================================

INPUT_FILE = "results/rereference/rereferenced_raw.fif"

EVENT_FILE = (
    "data/raw/erp_core/sub-021/"
    "ses-P3/eeg/sub-021_ses-P3_task-P3_events.tsv"
)

OUTPUT_DIR = Path("results/epochs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.4 EPOCH SEGMENTATION")
print("=" * 60)

# ----------------------------------------------------------
# Load EEG
# ----------------------------------------------------------

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

print("\nLoaded Re-Referenced EEG")

# ----------------------------------------------------------
# Read Events
# ----------------------------------------------------------

events_df = pd.read_csv(
    EVENT_FILE,
    sep="\t"
)

print("\nEvents Loaded")
print("Total Events :", len(events_df))

# ----------------------------------------------------------
# Build MNE Events Array
# ----------------------------------------------------------

events = []

for _, row in events_df.iterrows():

    sample = int(row["sample"])

    event_code = int(row["value"])

    events.append(
        [sample, 0, event_code]
    )

events = mne.events_from_annotations(raw)[0] \
    if len(events) == 0 else events

events = pd.DataFrame(
    events,
    columns=["sample", "dummy", "event"]
).to_numpy()

print("Events Converted :", len(events))

# ----------------------------------------------------------
# Event Dictionary
# ----------------------------------------------------------

unique_codes = sorted(
    list(events_df["value"].unique())
)

event_dict = {}

for code in unique_codes:
    event_dict[str(code)] = int(code)

print("\nUnique Event Codes")
print(unique_codes)

# ----------------------------------------------------------
# Epoching
# ----------------------------------------------------------

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.8,
    preload=True,
    baseline=None,
    verbose=False
)

print("\nEpochs Created")

print("Shape:")
print(epochs.get_data().shape)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

epochs.save(
    OUTPUT_DIR / "epochs-epo.fif",
    overwrite=True
)

with open(
    OUTPUT_DIR / "epoch_report.txt",
    "w"
) as f:

    f.write("PHASE 2.4 EPOCH SEGMENTATION\n\n")
    f.write(f"Epoch Shape : {epochs.get_data().shape}\n")
    f.write(f"Event Count : {len(events)}\n")

print("\nOutputs Saved")
print("results/epochs/epochs-epo.fif")
print("results/epochs/epoch_report.txt")

print("\nPHASE 2.4 COMPLETE")