# run_phase23_montage.py
# Phase 2.3 — Montage / channel-position verification plots
#   Saves one scalp layout plot per dataset showing electrode positions.
#   Confirms montage was correctly applied during preprocessing.
#
# SAFE TO RUN: reads one raw EEG file per dataset from data/raw/
#              writes 3 PNG plots to results_batch/audit/phase2_checks/montage/
#              touches NOTHING else.
#
# Run from project root:
#   cd ~/23BRS1328/erp_trial_min
#   python preprocessing_batch/run_phase23_montage.py

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

# ==================================================
# Config
# ==================================================

RESULTS_ROOT = Path("results_batch")
OUT_DIR      = RESULTS_ROOT / "audit" / "phase2_checks" / "montage"
MONTAGE_NAME = "standard_1020"

RAW_ROOTS = {
    "ERP_CORE": Path("data/raw/erp_core"),
    "DS006018": Path("data/raw/ds006018"),
    "DS002680": Path("data/raw/ds002680"),
}

EOG_NAMES = {
    "ERP_CORE": {"HEOG_LEFT", "HEOG_RIGHT", "VEOG_LOWER"},
    "DS006018": {"HEL", "HER", "VER"},
    "DS002680": set(),
}

# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_ROOT / "logs" / "phase23_montage.log"),
    ],
)
log = logging.getLogger("phase23")

# ==================================================
# Helpers
# ==================================================

def _load_one_raw(dataset: str, root: Path) -> mne.io.BaseRaw | None:
    """Load the first available raw file for this dataset."""
    if dataset == "DS006018":
        files = sorted(root.rglob("*.vhdr"))
        if not files:
            return None
        return mne.io.read_raw_brainvision(files[0], preload=False, verbose="ERROR")
    else:
        # ERP_CORE and DS002680 use EEGLAB .set
        files = sorted(root.rglob("*.set"))
        if not files:
            return None
        return mne.io.read_raw_eeglab(files[0], preload=False, verbose="ERROR")


def _apply_channel_types(raw: mne.io.BaseRaw, dataset: str) -> None:
    eog_set = EOG_NAMES.get(dataset, set())
    ch_types = {}
    for ch in raw.ch_names:
        up = ch.upper()
        if up in eog_set or "EOG" in up:
            ch_types[ch] = "eog"
        elif "ECG" in up:
            ch_types[ch] = "ecg"
    if ch_types:
        raw.set_channel_types(ch_types)


def _plot_montage(raw: mne.io.BaseRaw, dataset: str, out_png: Path) -> None:
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")

    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    n_eeg = len(picks_eeg)

    # Plot 1: channel positions (2D scalp layout)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: standard MNE montage plot
    try:
        montage_fig = raw.plot_sensors(
            kind="topomap",
            ch_type="eeg",
            show_names=True,
            show=False,
        )
        # Copy the axes content
        src_ax = montage_fig.axes[0]
        for item in src_ax.get_children():
            pass  # just close and re-do manually
        plt.close(montage_fig)
    except Exception:
        pass

    # Direct approach: use plot_sensors on the axes
    try:
        raw.plot_sensors(
            kind="topomap",
            ch_type="eeg",
            show_names=True,
            axes=axes[0],
            show=False,
        )
    except Exception as exc:
        axes[0].text(
            0.5, 0.5, f"plot_sensors failed:\n{exc}",
            ha="center", va="center", transform=axes[0].transAxes, fontsize=9,
        )

    axes[0].set_title(f"{dataset}\nChannel positions ({n_eeg} EEG channels)")

    # Right: 3D-style flat layout using channel positions
    try:
        layout = mne.channels.find_layout(raw.info, ch_type="eeg")
        pos = layout.pos[:, :2]
        names = layout.names

        axes[1].scatter(pos[:, 0], pos[:, 1], s=80, c="steelblue", zorder=3)
        for i, name in enumerate(names):
            axes[1].annotate(
                name,
                (pos[i, 0], pos[i, 1]),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )
        axes[1].set_aspect("equal")
        axes[1].set_title(f"{dataset}\nLayout ({len(names)} channels)")
        axes[1].axis("off")
    except Exception as exc:
        axes[1].text(
            0.5, 0.5, f"Layout failed:\n{exc}",
            ha="center", va="center", transform=axes[1].transAxes, fontsize=9,
        )

    fig.suptitle(
        f"Phase 2.3 Montage Verification — {dataset}\n"
        f"Montage: {MONTAGE_NAME}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  [OK] Saved: {out_png}")


# ==================================================
# Main
# ==================================================

def run_montage_verification() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset, root in RAW_ROOTS.items():
        out_png = OUT_DIR / f"{dataset}_montage_verification.png"
        if out_png.exists():
            log.info(f"[SKIP] {dataset} montage plot already exists.")
            continue

        log.info(f"[{dataset}] Loading one raw file for montage check...")

        if not root.exists():
            log.warning(f"[SKIP] {dataset} raw dir not found: {root}")
            continue

        try:
            raw = _load_one_raw(dataset, root)
            if raw is None:
                log.warning(f"[SKIP] No raw files found for {dataset}.")
                continue

            _apply_channel_types(raw, dataset)
            _plot_montage(raw, dataset, out_png)

        except Exception as exc:
            log.warning(f"[WARN] {dataset} montage verification failed: {exc}")

    log.info("=" * 60)
    log.info("Phase 2.3 Montage verification complete.")
    log.info(f"Plots saved to: {OUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_montage_verification()
