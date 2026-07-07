import mne
from pathlib import Path

subjects = sorted(
    Path("data/preprocessed").glob(
        "*/epochs-epo.fif"
    )
)

for ep_file in subjects:

    epochs = mne.read_epochs(
        ep_file,
        preload=True
    )

    epochs.drop_bad(
        reject=dict(
            eeg=150e-6
        )
    )

    epochs.save(
        ep_file.parent /
        "clean_epochs-epo.fif",
        overwrite=True
    )