import mne
from pathlib import Path

subjects = sorted(
    Path("data/preprocessed").glob(
        "*/filtered_raw.fif"
    )
)

for fif_file in subjects:

    print("\nSubject:", fif_file)

    raw = mne.io.read_raw_fif(
        fif_file,
        preload=True
    )

    ica_file = (
        fif_file.parent /
        "ica_solution-ica.fif"
    )

    ica = mne.preprocessing.read_ica(
        ica_file
    )

    eog_inds = []

    for ch in raw.ch_names:

        if "EOG" in ch.upper():

            inds, scores = ica.find_bads_eog(
                raw,
                ch_name=ch
            )

            eog_inds.extend(inds)

    ica.exclude = list(set(eog_inds))

    cleaned = raw.copy()

    ica.apply(cleaned)

    cleaned.save(
        fif_file.parent /
        "cleaned_raw.fif",
        overwrite=True
    )