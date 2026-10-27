import mne
from pathlib import Path

# ==========================================================
# PHASE 2.5 BASELINE CORRECTION
# ERP CORE P3 Subject 021
# ==========================================================

INPUT_FILE = "results/epochs/epochs-epo.fif"

OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.5 BASELINE CORRECTION")
print("=" * 60)

# ----------------------------------------------------------
# Load Epochs
# ----------------------------------------------------------

epochs = mne.read_epochs(
    INPUT_FILE,
    preload=True
)

print("\nLoaded Epochs")
print("Shape :", epochs.get_data().shape)

print("\nBaseline Before:")
print(epochs.baseline)

# ----------------------------------------------------------
# Apply Baseline Correction
# ----------------------------------------------------------

epochs.apply_baseline(
    baseline=(-0.2, 0.0)
)

print("\nBaseline Applied")
print("Window : -200 ms to 0 ms")

# ----------------------------------------------------------
# Save Corrected Epochs
# ----------------------------------------------------------

output_file = OUTPUT_DIR / "baseline_epochs-epo.fif"

epochs.save(
    output_file,
    overwrite=True
)

# ----------------------------------------------------------
# Report
# ----------------------------------------------------------

with open(
    OUTPUT_DIR / "baseline_report.txt",
    "w"
) as f:

    f.write("PHASE 2.5 BASELINE CORRECTION\n\n")
    f.write("Baseline Window: -0.2 to 0.0 sec\n")
    f.write(f"Epoch Shape: {epochs.get_data().shape}\n")
    f.write("Status: SUCCESS\n")

print("\nOutputs Saved")
print(output_file)
print("results/baseline/baseline_report.txt")

print("\nPHASE 2.5 COMPLETE")