import mne
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================================
# PHASE 2.7 ERP GENERATION
# ERP CORE P3
# ==========================================================

INPUT_FILE = "results/rejection/clean_epochs-epo.fif"

OUTPUT_DIR = Path("results/evoked")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2.7 ERP GENERATION")
print("=" * 60)

# ----------------------------------------------------------
# Load epochs
# ----------------------------------------------------------

epochs = mne.read_epochs(
    INPUT_FILE,
    preload=True
)

print("\nEpoch Count:", len(epochs))

# ----------------------------------------------------------
# Define Target / NonTarget
# ----------------------------------------------------------

critical_codes = [
    "11",
    "22",
    "33",
    "44",
    "55"
]

all_codes = list(
    epochs.event_id.keys()
)

standard_codes = [
    code
    for code in all_codes
    if code not in critical_codes
]

print("\nCritical Codes:")
print(critical_codes)

print("\nStandard Codes:")
print(standard_codes)

# ----------------------------------------------------------
# Create ERP Conditions
# ----------------------------------------------------------

critical_epochs = mne.concatenate_epochs(
    [epochs[code] for code in critical_codes]
)

standard_epochs = mne.concatenate_epochs(
    [epochs[code] for code in standard_codes]
)

print(
    "\nCritical Trials:",
    len(critical_epochs)
)

print(
    "Standard Trials:",
    len(standard_epochs)
)

# ----------------------------------------------------------
# ERP Averages
# ----------------------------------------------------------

critical_evoked = critical_epochs.average()

standard_evoked = standard_epochs.average()

difference_evoked = mne.combine_evoked(
    [critical_evoked, standard_evoked],
    weights=[1, -1]
)

# ----------------------------------------------------------
# Save Evoked Files
# ----------------------------------------------------------

critical_evoked.save(
    OUTPUT_DIR / "critical-ave.fif",
    overwrite=True
)

standard_evoked.save(
    OUTPUT_DIR / "standard-ave.fif",
    overwrite=True
)

difference_evoked.save(
    OUTPUT_DIR / "difference-ave.fif",
    overwrite=True
)

# ----------------------------------------------------------
# ERP Figures
# ----------------------------------------------------------

fig = critical_evoked.plot(
    spatial_colors=True,
    show=False
)

fig.savefig(
    OUTPUT_DIR / "critical_erp.png"
)

fig = standard_evoked.plot(
    spatial_colors=True,
    show=False
)

fig.savefig(
    OUTPUT_DIR / "standard_erp.png"
)

fig = difference_evoked.plot(
    spatial_colors=True,
    show=False
)

fig.savefig(
    OUTPUT_DIR / "difference_wave.png"
)

plt.close("all")

# ----------------------------------------------------------
# P300 Metrics
# ----------------------------------------------------------

metrics = []

for ch in ["Pz", "Cz", "Fz"]:

    if ch in difference_evoked.ch_names:

        idx = difference_evoked.ch_names.index(ch)

        data = difference_evoked.data[idx]

        times = difference_evoked.times

        mask = (
            (times >= 0.25)
            & (times <= 0.60)
        )

        peak_amp = np.max(
            data[mask]
        )

        peak_latency = (
            times[mask][
                np.argmax(
                    data[mask]
                )
            ]
        )

        metrics.append(
            [
                ch,
                peak_amp,
                peak_latency
            ]
        )

metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Channel",
        "Peak_Amplitude_uV",
        "Peak_Latency_s"
    ]
)

metrics_df.to_csv(
    OUTPUT_DIR / "p300_metrics.csv",
    index=False
)

# ----------------------------------------------------------
# Report
# ----------------------------------------------------------

with open(
    OUTPUT_DIR / "evoked_report.txt",
    "w"
) as f:

    f.write(
        "PHASE 2.7 ERP GENERATION\n\n"
    )

    f.write(
        f"Critical Trials : {len(critical_epochs)}\n"
    )

    f.write(
        f"Standard Trials : {len(standard_epochs)}\n"
    )

    f.write(
        "\nOutputs Generated:\n"
    )

    f.write(
        "- critical_erp.png\n"
    )

    f.write(
        "- standard_erp.png\n"
    )

    f.write(
        "- difference_wave.png\n"
    )

    f.write(
        "- p300_metrics.csv\n"
    )

print("\nERP Generation Complete")
print("Outputs saved to results/evoked")