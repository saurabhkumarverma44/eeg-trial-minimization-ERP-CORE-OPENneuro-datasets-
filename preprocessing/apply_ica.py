import mne
from pathlib import Path

INPUT_RAW = "results/filter/filtered_raw.fif"
ICA_FILE = "results/ica/ica_solution.fif"

OUTPUT_DIR = Path("results/ica")
OUTPUT_DIR.mkdir(exist_ok=True)

raw = mne.io.read_raw_fif(
    INPUT_RAW,
    preload=True
)

ica = mne.preprocessing.read_ica(
    ICA_FILE
)

bad_components = []

# ------------------------------------------------
# EOG
# ------------------------------------------------

eog_channels = [
    ch for ch in raw.ch_names
    if "eog" in ch.lower()
]

for ch in eog_channels:

    inds, scores = ica.find_bads_eog(
        raw,
        ch_name=ch
    )

    bad_components.extend(inds)

# ------------------------------------------------
# ECG
# ------------------------------------------------

try:

    ecg_inds, ecg_scores = ica.find_bads_ecg(
        raw,
        method="correlation"
    )

    bad_components.extend(ecg_inds)

except:

    print("No ECG channel found")

bad_components = list(
    set(bad_components)
)

ica.exclude = bad_components

clean_raw = raw.copy()

ica.apply(clean_raw)

clean_raw.save(
    OUTPUT_DIR / "cleaned_raw.fif",
    overwrite=True
)

print("Removed Components")
print(bad_components)