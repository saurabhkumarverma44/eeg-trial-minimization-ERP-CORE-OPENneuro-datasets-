import mne
from pathlib import Path
from mne.preprocessing import ICA

INPUT_FILE = "results/filter/filtered_raw.fif"

OUTPUT_DIR = Path("results/ica")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

raw.set_eeg_reference(
    "average",
    projection=False
)

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

ica.save(
    OUTPUT_DIR / "ica_solution.fif"
)

print("ICA Solution Saved")