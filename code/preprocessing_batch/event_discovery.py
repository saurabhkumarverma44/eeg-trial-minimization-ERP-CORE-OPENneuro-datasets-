from pathlib import Path
import mne

# --------------------------------------------------
# DS006018
# --------------------------------------------------

print("\nDS006018")

sample = sorted(
    Path("data/raw/ds006018").rglob("*.vhdr")
)[0]

raw = mne.io.read_raw_brainvision(
    sample,
    preload=False
)

events, event_id = mne.events_from_annotations(
    raw
)

print(event_id)

# --------------------------------------------------
# DS002680
# --------------------------------------------------

print("\nDS002680")

sample = sorted(
    Path("data/raw/ds002680").rglob("*.set")
)[0]

raw = mne.io.read_raw_eeglab(
    sample,
    preload=False
)

try:

    events, event_id = mne.events_from_annotations(
        raw
    )

    print(event_id)

except Exception as e:

    print(e)
