import mne
import matplotlib.pyplot as plt

evoked = mne.read_evokeds(
    "results/grand_average-ave.fif",
    condition=0
)

fig = evoked.plot_joint(
    show=False
)

plt.savefig(
    "results/grand_average_topography.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved")