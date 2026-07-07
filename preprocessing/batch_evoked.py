import mne
from pathlib import Path

all_evokeds = []

files = sorted(
    Path("data/preprocessed").glob(
        "*/clean_epochs-epo.fif"
    )
)

for ep_file in files:

    epochs = mne.read_epochs(
        ep_file,
        preload=True
    )

    evoked = epochs.average()

    all_evokeds.append(
        evoked
    )

grand_average = mne.grand_average(
    all_evokeds
)

grand_average.save(
    "results/grand_average-ave.fif",
    overwrite=True
)