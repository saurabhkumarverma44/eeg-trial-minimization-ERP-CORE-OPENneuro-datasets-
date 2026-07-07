import mne
from pathlib import Path

subjects = sorted(
    Path("data/preprocessed").glob(
        "*/cleaned_raw.fif"
    )
)

for fif_file in subjects:

    raw = mne.io.read_raw_fif(
        fif_file,
        preload=True
    )

    events = mne.find_events(raw)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2,0),
        preload=True
    )

    epochs.save(
        fif_file.parent /
        "epochs-epo.fif",
        overwrite=True
    )