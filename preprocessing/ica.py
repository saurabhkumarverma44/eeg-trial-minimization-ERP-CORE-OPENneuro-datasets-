import mne
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA

# ==========================================================
# PHASE 2.2 ICA ARTEFACT REMOVAL
# ERP CORE P3 Subject 021
# ==========================================================

INPUT_FILE = "results/filter/filtered_raw.fif"

OUTPUT_DIR = Path("results/ica")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.2 ICA ARTEFACT REMOVAL")
print("=" * 60)

# ----------------------------------------------------------
# Load filtered EEG
# ----------------------------------------------------------

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

print("\nLoaded Filtered Data")
print("Channels :", len(raw.ch_names))

# ----------------------------------------------------------
# Set EOG channel types
# Supervisor specifically requested this
# ----------------------------------------------------------

eog_map = {}

for ch in raw.ch_names:

    lower = ch.lower()

    if "heog" in lower:
        eog_map[ch] = "eog"

    elif "veog" in lower:
        eog_map[ch] = "eog"

if len(eog_map) > 0:
    raw.set_channel_types(eog_map)

print("\nEOG Channels Found:")
print(list(eog_map.keys()))

# ----------------------------------------------------------
# Apply standard montage
# ----------------------------------------------------------

raw.set_montage(
    "standard_1020",
    on_missing="ignore"
)

# ----------------------------------------------------------
# Run Extended Infomax ICA
# ----------------------------------------------------------

print("\nRunning Extended Infomax ICA...")

ica = ICA(
    n_components=0.999,
    method="infomax",
    fit_params=dict(extended=True),
    random_state=42,
    max_iter="auto"
)

ica.fit(
    raw,
    picks="eeg"
)

print("ICA Fit Complete")

print("\nNumber of ICA Components:")
print(ica.n_components_)

# ----------------------------------------------------------
# Detect EOG Components
# ----------------------------------------------------------

bad_components = []

for eog_channel in eog_map.keys():

    inds, scores = ica.find_bads_eog(
        raw,
        ch_name=eog_channel
    )

    bad_components.extend(inds)

bad_components = sorted(
    list(set(bad_components))
)

ica.exclude = bad_components

print("\nComponents Marked For Removal:")
print(bad_components)

# ----------------------------------------------------------
# Apply ICA
# ----------------------------------------------------------

raw_clean = raw.copy()

ica.apply(raw_clean)

print("\nICA Artefact Removal Applied")

# ----------------------------------------------------------
# Save cleaned EEG
# ----------------------------------------------------------

clean_file = OUTPUT_DIR / "cleaned_raw.fif"

raw_clean.save(
    clean_file,
    overwrite=True
)

# ----------------------------------------------------------
# Create ICA QC Report
# ----------------------------------------------------------

qc = pd.DataFrame(
    {
        "Subject": ["sub-021"],
        "Dataset": ["ERP_CORE"],
        "ICA_Components": [ica.n_components_],
        "Removed_ICs": [len(bad_components)],
        "Removed_Component_List": [str(bad_components)]
    }
)

report_file = OUTPUT_DIR / "ica_report.csv"

qc.to_csv(
    report_file,
    index=False
)

# ----------------------------------------------------------
# Create Text Report
# ----------------------------------------------------------

with open(
    OUTPUT_DIR / "ica_summary.txt",
    "w"
) as f:

    f.write("PHASE 2.2 ICA ARTEFACT REMOVAL\n")
    f.write("=" * 40 + "\n\n")

    f.write("Dataset : ERP CORE\n")
    f.write("Subject : sub-021\n")
    f.write("Paradigm : P3\n\n")

    f.write("ICA Method : Extended Infomax\n")
    f.write(f"ICA Components : {ica.n_components_}\n")
    f.write(f"Removed ICs : {len(bad_components)}\n")
    f.write(f"Removed Components : {bad_components}\n\n")

    f.write("Status : SUCCESS\n")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------

print("\nRemoved IC Count :", len(bad_components))

print("\nSaved Files")
print(clean_file)
print(report_file)
print(OUTPUT_DIR / "ica_summary.txt")

print("\nPHASE 2.2 COMPLETE")