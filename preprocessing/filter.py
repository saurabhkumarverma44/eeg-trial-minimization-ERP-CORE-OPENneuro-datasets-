from pathlib import Path
import mne
import matplotlib.pyplot as plt

# --------------------------------------------------
# Phase 2.1 : Bandpass Filter
# ERP CORE P3 Subject 021
# --------------------------------------------------

RAW_FILE = (
    "data/raw/erp_core/"
    "sub-021/ses-P3/eeg/"
    "sub-021_ses-P3_task-P3_eeg.set"
)

OUTPUT_DIR = Path(
    "results/filter"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("PHASE 2.1 BANDPASS FILTER")
print("=" * 60)

# Load data
raw = mne.io.read_raw_eeglab(
    RAW_FILE,
    preload=True
)

print("\nSampling Rate :", raw.info["sfreq"])
print("Channels      :", len(raw.ch_names))

# PSD BEFORE
fig = raw.compute_psd().plot(show=False)

fig.savefig(
    OUTPUT_DIR / "psd_before_filter.png"
)

plt.close("all")

# Apply FIR Filter
raw.filter(
    l_freq=0.1,
    h_freq=40.0,
    method="fir"
)

# Save filtered raw
raw.save(
    OUTPUT_DIR / "filtered_raw.fif",
    overwrite=True
)

# PSD AFTER
fig = raw.compute_psd().plot(show=False)

fig.savefig(
    OUTPUT_DIR / "psd_after_filter.png"
)

plt.close("all")

print("\nFilter Type : FIR")
print("Passband    : 0.1 - 40 Hz")

print("\nOutputs Saved")

print(
    OUTPUT_DIR / "psd_before_filter.png"
)

print(
    OUTPUT_DIR / "psd_after_filter.png"
)

print(
    OUTPUT_DIR / "filtered_raw.fif"
)

print("\nPhase 2.1 Complete")