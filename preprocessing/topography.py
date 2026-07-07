import mne
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("results/evoked")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load Difference Wave
# --------------------------------------------------

evoked = mne.read_evokeds(
    "results/evoked/difference-ave.fif",
    condition=0
)

# --------------------------------------------------
# Rename channels to standard 10-20 names
# --------------------------------------------------

rename_dict = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FCZ": "FCz",
    "CPZ": "CPz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "FZ": "Fz",
}

existing = {}

for ch in evoked.ch_names:
    upper = ch.upper()
    if upper in rename_dict:
        existing[ch] = rename_dict[upper]

if existing:
    evoked.rename_channels(existing)

# --------------------------------------------------
# Apply standard montage
# --------------------------------------------------

montage = mne.channels.make_standard_montage(
    "standard_1020"
)

evoked.set_montage(
    montage,
    on_missing="ignore"
)

# --------------------------------------------------
# Use only EEG channels
# --------------------------------------------------

evoked.pick("eeg")

# --------------------------------------------------
# Plot topographies
# --------------------------------------------------

times = [0.30, 0.40, 0.50]

fig = evoked.plot_topomap(
    times=times,
    ch_type="eeg",
    time_unit="s",
    show=False
)

fig.savefig(
    OUTPUT_DIR / "grand_average_topomap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

print("Grand Average Topography Saved")