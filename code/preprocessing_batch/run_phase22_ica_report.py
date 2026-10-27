# run_phase22_ica_report.py
# Phase 2.2 — ICA variance report
#   - How many ICs were excluded per subject/task
#   - Whether any subject loses >40% variance after ICA
#   - Logs IC count per subject (supervisor requirement)
#
# SAFE TO RUN: only reads ica.fif from results_batch/
#              writes CSV + summary to results_batch/audit/phase2_checks/ica/
#              touches NOTHING else.
#
# Run from project root:
#   cd ~/23BRS1328/erp_trial_min
#   python preprocessing_batch/run_phase22_ica_report.py

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import mne
from mne.preprocessing import ICA

# ==================================================
# Config
# ==================================================

RESULTS_ROOT     = Path("results_batch")
DATASETS         = ["ERP_CORE", "DS006018", "DS002680"]
OUT_DIR          = RESULTS_ROOT / "audit" / "phase2_checks" / "ica"
VARIANCE_THRESH  = 0.40   # flag if ICA removes >40% variance

# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_ROOT / "logs" / "phase22_ica.log"),
    ],
)
log = logging.getLogger("phase22")

# ==================================================
# Helpers
# ==================================================

FIELDNAMES = [
    "dataset", "subject", "task",
    "n_components_total",
    "n_components_excluded",
    "excluded_indices",
    "variance_explained_excluded",
    "variance_flag",          # True if >40% variance removed
]


def _append_row(csv_path: Path, row: dict) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _variance_of_excluded(ica: ICA) -> float:
    """
    Estimate fraction of total EEG variance explained by excluded components.
    Uses the mixing matrix column norms as a proxy.
    Returns 0.0 if no components excluded or mixing matrix unavailable.
    """
    if not ica.exclude:
        return 0.0
    try:
        # mixing_matrix_ shape: (n_channels, n_components)
        mixing = ica.mixing_matrix_
        # variance proxy: sum of squared column norms for excluded vs all
        col_norms_sq = np.sum(mixing ** 2, axis=0)
        total         = col_norms_sq.sum()
        excluded_var  = col_norms_sq[ica.exclude].sum()
        if total == 0:
            return 0.0
        return float(excluded_var / total)
    except Exception:
        return float("nan")


# ==================================================
# Main
# ==================================================

def run_ica_report() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ica_variance_report.csv"

    # Fresh CSV each run (re-runnable)
    if csv_path.exists():
        csv_path.unlink()

    flagged_count = 0
    total_count   = 0

    for dataset in DATASETS:
        ds_root = RESULTS_ROOT / dataset
        if not ds_root.exists():
            log.warning(f"[SKIP] {dataset} results dir not found.")
            continue

        ica_fifs = sorted(ds_root.rglob("ica.fif"))
        if not ica_fifs:
            log.warning(f"[SKIP] No ica.fif found for {dataset}.")
            continue

        log.info(f"[{dataset}] Found {len(ica_fifs)} ICA files.")

        for fif in ica_fifs:
            parts   = fif.relative_to(ds_root).parts
            subject = parts[0] if len(parts) >= 1 else "unknown"
            task    = parts[1] if len(parts) >= 2 else "unknown"

            try:
                ica = mne.preprocessing.read_ica(fif, verbose="ERROR")

                n_total    = ica.n_components_
                n_excluded = len(ica.exclude)
                excl_idx   = str(ica.exclude)
                var_frac   = _variance_of_excluded(ica)
                flagged    = (
                    not np.isnan(var_frac) and var_frac > VARIANCE_THRESH
                )

                if flagged:
                    flagged_count += 1
                    log.warning(
                        f"  [FLAG >40% VAR] {dataset} {subject} {task}: "
                        f"{n_excluded}/{n_total} ICs excluded, "
                        f"variance removed={var_frac*100:.1f}%"
                    )
                else:
                    log.info(
                        f"  [OK] {dataset} {subject} {task}: "
                        f"{n_excluded}/{n_total} ICs excluded, "
                        f"variance removed={var_frac*100:.1f}%"
                    )

                _append_row(csv_path, {
                    "dataset":                    dataset,
                    "subject":                    subject,
                    "task":                       task,
                    "n_components_total":         n_total,
                    "n_components_excluded":      n_excluded,
                    "excluded_indices":           excl_idx,
                    "variance_explained_excluded": f"{var_frac:.4f}",
                    "variance_flag":              flagged,
                })
                total_count += 1

            except Exception as exc:
                log.warning(f"  [WARN] {dataset} {subject} {task}: {exc}")

    # Summary
    log.info("=" * 60)
    log.info(f"Phase 2.2 ICA Report complete.")
    log.info(f"Total recordings processed : {total_count}")
    log.info(f"Flagged (>40% variance)    : {flagged_count}")
    log.info(f"Report saved to            : {csv_path}")
    log.info("=" * 60)

    # Print quick summary to terminal
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print("\n--- ICA Summary by Dataset ---")
        print(df.groupby("dataset")[
            ["n_components_excluded", "variance_explained_excluded"]
        ].describe().round(3))
        print(f"\nSubjects with >40% variance removed: {flagged_count}")
    except Exception:
        pass


if __name__ == "__main__":
    run_ica_report()
