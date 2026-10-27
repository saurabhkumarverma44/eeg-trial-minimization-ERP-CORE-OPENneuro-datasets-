# stage3_feature_extraction/run_stage3_all.py
#
# Runs the full Phase 3 grid in spec order: 3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5
# -> 3.6 -> 3.7. Each step is independently re-runnable/resumable (skips
# work already done unless overwrite=True), matching the resume pattern
# used in Phase 2's run_*_all.py scripts.
#
# Run from project root:
#   cd /lp-dev/23BRS1328/erp_trial_min
#   python stage3_feature_extraction/run_stage3_all.py
#
# Does NOT touch preprocessing_batch/ or any Phase 2 output file — only
# reads results_batch/<DATASET>/.../*-epo.fif and writes under
# results_batch/stage3/.

from __future__ import annotations

from subsample_trials import run_subsampling
from bootstrap_grid import run_bootstrap_grid
from features_band_power import run_band_power
from features_plv import run_plv
from features_wpli import run_wpli
from features_entropy import run_entropy
from assemble_features import run_assembly

from stage3_utils import get_logger

log = get_logger("run_stage3_all")


def main(overwrite: bool = False) -> None:
    log.info("#" * 70)
    log.info("PHASE 3 — Trial Subsampling Grid & Feature Extraction")
    log.info("#" * 70)

    log.info(">>> 3.1 Subsampling Function")
    run_subsampling(overwrite=overwrite)

    log.info(">>> 3.2 Bootstrap Wrapper")
    run_bootstrap_grid(overwrite=overwrite)

    log.info(">>> 3.3 Band Power Features")
    run_band_power(overwrite=overwrite)

    log.info(">>> 3.4 PLV Matrices")
    run_plv(overwrite=overwrite)

    log.info(">>> 3.5 wPLI Matrices")
    run_wpli(overwrite=overwrite)

    log.info(">>> 3.6 Spectral Entropy")
    run_entropy(overwrite=overwrite)

    log.info(">>> 3.7 Feature Assembly")
    run_assembly()

    log.info("#" * 70)
    log.info("PHASE 3 COMPLETE. See results_batch/stage3/ for all outputs.")
    log.info("#" * 70)


if __name__ == "__main__":
    main(overwrite=False)

