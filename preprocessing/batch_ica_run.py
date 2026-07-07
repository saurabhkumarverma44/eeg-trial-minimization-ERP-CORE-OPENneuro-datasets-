import mne
from pathlib import Path
from mne.preprocessing import ICA

ROOT = Path("data/preprocessed")

ROOT.mkdir(exist_ok=True)

subjects = sorted(
    ROOT.glob("*/filtered_raw.fif")
)

for fif_file in subjects:

    print("\nProcessing:", fif_file)

    raw = mne.io.read_raw_fif(
        fif_file,
        preload=True
    )

    ica = ICA(
        n_components=0.99,
        random_state=42,
        max_iter="auto"
    )

    ica.fit(raw)

    out_file = (
        fif_file.parent /
        "ica_solution-ica.fif"
    )

    ica.save(out_file)

    print("Saved:", out_file)