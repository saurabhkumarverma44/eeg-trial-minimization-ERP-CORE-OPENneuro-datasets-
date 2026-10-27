import mne
import pandas as pd
from pathlib import Path

# ==========================================================
# PHASE 2.6 EPOCH REJECTION
# ERP CORE P3 Subject 021
# ==========================================================

INPUT_FILE = "results/baseline/baseline_epochs-epo.fif"

OUTPUT_DIR = Path("results/rejection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.6 EPOCH REJECTION")
print("=" * 60)

# ----------------------------------------------------------
# Load Epochs
# ----------------------------------------------------------

epochs = mne.read_epochs(
    INPUT_FILE,
    preload=True
)

n_before = len(epochs)

print("\nEpochs Before Rejection:")
print(n_before)

# ----------------------------------------------------------
# Peak-to-Peak Rejection
# WBS Threshold = 100 µV
# ----------------------------------------------------------

reject_criteria = dict(
    eeg=100e-6
)

epochs.drop_bad(
    reject=reject_criteria
)

n_after = len(epochs)

rejected = n_before - n_after

rejection_rate = (
    rejected / n_before
) * 100

print("\nEpochs After Rejection:")
print(n_after)

print("\nRejected Epochs:")
print(rejected)

print("\nRejection Rate:")
print(f"{rejection_rate:.2f}%")

# ----------------------------------------------------------
# Flag if >30%
# ----------------------------------------------------------

flag = "NO"

if rejection_rate > 30:
    flag = "YES"

print("\nFlag (>30% Rejected):")
print(flag)

# ----------------------------------------------------------
# Save Clean Epochs
# ----------------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "clean_epochs-epo.fif"
)

epochs.save(
    output_file,
    overwrite=True
)

# ----------------------------------------------------------
# QC Report
# ----------------------------------------------------------

report = pd.DataFrame(
    {
        "Subject": ["sub-021"],
        "Total_Epochs": [n_before],
        "Clean_Epochs": [n_after],
        "Rejected_Epochs": [rejected],
        "Rejection_Rate": [round(rejection_rate, 2)],
        "Flag_30Percent": [flag]
    }
)

report.to_csv(
    OUTPUT_DIR / "rejection_report.csv",
    index=False
)

with open(
    OUTPUT_DIR / "rejection_summary.txt",
    "w"
) as f:

    f.write("PHASE 2.6 EPOCH REJECTION\n\n")
    f.write(f"Total Epochs: {n_before}\n")
    f.write(f"Clean Epochs: {n_after}\n")
    f.write(f"Rejected Epochs: {rejected}\n")
    f.write(f"Rejection Rate: {rejection_rate:.2f}%\n")
    f.write(f"Flag (>30%): {flag}\n")

print("\nOutputs Saved")
print(output_file)
print("results/rejection/rejection_report.csv")
print("results/rejection/rejection_summary.txt")

print("\nPHASE 2.6 COMPLETE")