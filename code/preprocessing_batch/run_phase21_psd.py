# run_phase21_psd.py
# Phase 2.1 — PSD plots confirming no ringing/filtering artifacts
#
# SAFE TO RUN: only reads cleaned_raw.fif from results_batch/
#              writes PNG plots to results_batch/audit/phase2_checks/psd/
#              touches NOTHING else.
#
# Run from project root:
#   cd ~/23BRS1328/erp_trial_min
#   python preprocessing_batch/run_phase21_psd.py

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

# ==================================================
# Config — must match your config.py
# ==================================================

RESULTS_ROOT  = Path("results_batch")
DATASETS      = ["ERP_CORE", "DS006018", "DS002680"]
LOW_CUT       = 0.1
HIGH_CUT      = 40.0
OUT_DIR       = RESULTS_ROOT / "audit" / "phase2_checks" / "psd"

# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_ROOT / "logs" / "phase21_psd.log"),
    ],
)
log = logging.getLogger("phase21")

# ==================================================
# Main
# ==================================================

def run_psd_checks() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        ds_root = RESULTS_ROOT / dataset
        if not ds_root.exists():
            log.warning(f"[SKIP] {dataset} results dir not found.")
            continue

        cleaned_fifs = sorted(ds_root.rglob("cleaned_raw.fif"))
        if not cleaned_fifs:
            log.warning(f"[SKIP] No cleaned_raw.fif found for {dataset}.")
            continue

        log.info(f"[{dataset}] Found {len(cleaned_fifs)} cleaned files.")

        ds_out = OUT_DIR / dataset
        ds_out.mkdir(parents=True, exist_ok=True)

        # Per-subject PSD plots
        for fif in cleaned_fifs:
            # e.g. results_batch/ERP_CORE/sub-001/P3/cleaned_raw.fif
            parts = fif.relative_to(ds_root).parts
            subject = parts[0] if len(parts) >= 1 else "unknown"
            task    = parts[1] if len(parts) >= 2 else "unknown"
            label   = f"{subject}_{task}"

            out_png = ds_out / f"{label}_psd.png"
            if out_png.exists():
                log.info(f"  [SKIP] {label} — PSD already exists.")
                continue

            try:
                raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")

                picks_eeg = mne.pick_types(
                    raw.info, meg=False, eeg=True, eog=False, exclude="bads"
                )

                # compute_psd returns a Spectrum object in MNE >= 1.2
                spectrum = raw.compute_psd(
                    method="welch",
                    fmin=0.5,
                    fmax=50.0,
                    picks=picks_eeg,
                    verbose="ERROR",
                )

                fig, ax = plt.subplots(figsize=(10, 4))
                spectrum.plot(axes=ax, show=False, picks=picks_eeg)

                # Mark filter cutoffs
                ax.axvline(LOW_CUT,  color="red",   linestyle="--",
                           linewidth=1, label=f"HP {LOW_CUT} Hz")
                ax.axvline(HIGH_CUT, color="blue",  linestyle="--",
                           linewidth=1, label=f"LP {HIGH_CUT} Hz")
                ax.legend(fontsize=8)
                ax.set_title(f"{dataset} | {label} | PSD (post-filter)")

                fig.tight_layout()
                fig.savefig(out_png, dpi=120)
                plt.close(fig)

                log.info(f"  [OK] {label}")

            except Exception as exc:
                log.warning(f"  [WARN] {label}: {exc}")
                plt.close("all")

        # Grand-average PSD per dataset (one overlay of all subjects)
        log.info(f"[{dataset}] Computing grand-average PSD overlay...")
        _grand_psd(dataset, cleaned_fifs, ds_out)

    log.info("Phase 2.1 PSD checks complete.")
    log.info(f"Plots saved to: {OUT_DIR}")


def _grand_psd(
    dataset: str,
    fif_list: list[Path],
    out_dir: Path,
    max_subjects: int = 20,
) -> None:
    """
    Overlay PSD from up to max_subjects subjects on one figure.
    Sampling every file if there are many — just for visual QC.
    """
    out_png = out_dir / f"{dataset}_grand_psd_overlay.png"
    if out_png.exists():
        log.info(f"  [SKIP] Grand PSD overlay already exists for {dataset}.")
        return

    # Sample evenly if too many
    step = max(1, len(fif_list) // max_subjects)
    sampled = fif_list[::step][:max_subjects]

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = 0

    for fif in sampled:
        try:
            raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
            picks_eeg = mne.pick_types(
                raw.info, meg=False, eeg=True, eog=False, exclude="bads"
            )
            spectrum = raw.compute_psd(
                method="welch",
                fmin=0.5,
                fmax=50.0,
                picks=picks_eeg,
                verbose="ERROR",
            )
            # Get PSD data for manual plotting (overlay)
            psds, freqs = spectrum.get_data(return_freqs=True)
            import numpy as np
            mean_psd = psds.mean(axis=0)  # average across channels
            ax.semilogy(freqs, mean_psd, alpha=0.4, linewidth=0.8)
            plotted += 1
        except Exception as exc:
            log.warning(f"  [WARN] grand PSD skip {fif.name}: {exc}")

    ax.axvline(LOW_CUT,  color="red",  linestyle="--", linewidth=1.2,
               label=f"HP {LOW_CUT} Hz")
    ax.axvline(HIGH_CUT, color="blue", linestyle="--", linewidth=1.2,
               label=f"LP {HIGH_CUT} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (V²/Hz)")
    ax.set_title(f"{dataset} — Grand PSD overlay ({plotted} subjects sampled)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log.info(f"  [OK] Grand PSD overlay saved ({plotted} subjects).")


if __name__ == "__main__":
    run_psd_checks()
