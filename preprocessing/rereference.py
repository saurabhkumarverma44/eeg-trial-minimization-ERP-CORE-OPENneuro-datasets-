import mne
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================================
# PHASE 2.3 RE-REFERENCING
# ERP CORE P3 Subject 021
# ==========================================================

INPUT_FILE = "results/ica/cleaned_raw.fif"

OUTPUT_DIR = Path("results/rereference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.3 RE-REFERENCING")
print("=" * 60)

# ----------------------------------------------------------
# Load ICA-cleaned EEG
# ----------------------------------------------------------

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

print("\nLoaded ICA-Cleaned Data")
print("Channels :", len(raw.ch_names))

# ----------------------------------------------------------
# Apply Standard 10-20 Montage
# ----------------------------------------------------------

raw.set_montage(
    "standard_1020",
    on_missing="ignore"
)

print("\n10-20 Montage Applied")

# ----------------------------------------------------------
# Save Montage Plot
# ----------------------------------------------------------

fig = raw.plot_sensors(
    kind="topomap",
    show=False
)

fig.savefig(
    OUTPUT_DIR / "montage_plot.png"
)

plt.close("all")

# ----------------------------------------------------------
# Average Reference
# ----------------------------------------------------------

raw.set_eeg_reference(
    ref_channels="average"
)

print("\nAverage Reference Applied")

# ----------------------------------------------------------
# Save Re-referenced EEG
# ----------------------------------------------------------

output_file = OUTPUT_DIR / "rereferenced_raw.fif"

raw.save(
    output_file,
    overwrite=True
)

# ----------------------------------------------------------
# Create Report
# ----------------------------------------------------------

with open(
    OUTPUT_DIR / "rereference_report.txt",
    "w"
) as f:

    f.write("PHASE 2.3 RE-REFERENCING\n")
    f.write("=" * 40 + "\n\n")

    f.write("Dataset : ERP CORE\n")
    f.write("Subject : sub-021\n")
    f.write("Paradigm : P3\n\n")

    f.write("Reference : Average Reference\n")
    f.write("Montage : standard_1020\n\n")

    f.write("Outputs:\n")
    f.write("- rereferenced_raw.fif\n")
    f.write("- montage_plot.png\n\n")

    f.write("Status : SUCCESS\n")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------

print("\nOutputs Saved")

print(output_file)
print(OUTPUT_DIR / "montage_plot.png")
print(OUTPUT_DIR / "rereference_report.txt")

print("\nPHASE 2.3 COMPLETE")