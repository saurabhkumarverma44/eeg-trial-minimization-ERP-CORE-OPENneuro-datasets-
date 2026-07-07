# preprocessing_batch/common_preprocessing.py

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

from config import (
    LOW_CUT,
    HIGH_CUT,
    ICA_COMPONENTS,
    TMIN,
    TMAX,
    BASELINE,
    REJECT_UV,
    RESULTS_ROOT,
    DEFAULT_TOPO_TIMES,
    MONTAGE_NAME,
)
from event_registry import ERP_CORE as ERP_CORE_EVENTS
from event_registry import DS006018 as DS006018_EVENTS
from event_registry import DS002680 as DS002680_EVENTS


# ==================================================
# Logging
# ==================================================

_LOGGER = logging.getLogger("preprocessing_batch")
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    log_dir = RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "preprocessing.log")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    _LOGGER.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    _LOGGER.addHandler(sh)


# ==================================================
# Helpers
# ==================================================

def _mark_eog_ecg_channels(raw: mne.io.BaseRaw, dataset_name: str) -> None:
    """Set channel types for EOG/ECG based on names + dataset-specific heuristics."""
    eog_chs: List[str] = []
    ecg_chs: List[str] = []

    for ch in raw.ch_names:
        up = ch.upper()

        # Generic EOG pattern
        if "EOG" in up:
            eog_chs.append(ch)
            continue

        # Dataset-specific EOGs
        if dataset_name == "ERP_CORE":
            if up in {"HEOG_LEFT", "HEOG_RIGHT", "VEOG_LOWER"}:
                eog_chs.append(ch)
        elif dataset_name == "DS006018":
            # BrainVision: HEL, HER, VER are horizontal/vertical EOG
            if up in {"HEL", "HER", "VER"}:
                eog_chs.append(ch)

        # ECG: generic naming
        if "ECG" in up:
            ecg_chs.append(ch)

    ch_types = {}
    for ch in eog_chs:
        ch_types[ch] = "eog"
    for ch in ecg_chs:
        ch_types[ch] = "ecg"

    if ch_types:
        raw.set_channel_types(ch_types)


def _apply_montage(raw: mne.io.BaseRaw, dataset_name: str, subject: str,
                    task: str, run_id: str) -> None:
    """
    Apply a standard montage so EEG channels get real per-channel 3-D
    positions (ch['loc']). Without this, MNE falls back to legacy
    digitization points for topomap plotting, which can disagree with
    the post-retyping EEG channel count (EOG/ECG channels removed from
    the EEG count but not from the old dig-point list) and raise
    "Number of EEG digitization points doesn't match the number of EEG
    channels". on_missing="ignore" means channels that don't match the
    standard montage (EOG/ECG/non-standard names) are simply skipped,
    not treated as an error.
    """
    try:
        montage = mne.channels.make_standard_montage(MONTAGE_NAME)
        raw.set_montage(
            montage,
            match_case=False,
            on_missing="ignore",
            verbose="ERROR",
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            f"[WARN] Montage application failed "
            f"{dataset_name} {subject} {task} {run_id}: {exc}"
        )


def _get_event_mapping(dataset: str, task: str) -> Dict[str, List[int]]:
    if dataset == "ERP_CORE":
        return ERP_CORE_EVENTS.get(task, {})
    if dataset == "DS006018":
        return DS006018_EVENTS.get(task, {})
    if dataset == "DS002680":
        return DS002680_EVENTS.get(task, {})
    return {}


def _append_qc_row(
    csv_path: Path,
    row: Dict[str, object],
    fieldnames: List[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _is_valid_fif(fif_path: Path) -> bool:
    """
    Verify a saved .fif file is actually readable (header + full data
    buffer), not just present on disk. Guards against a previous run
    being killed mid-write or leaving a truncated/corrupted file behind
    while the DONE flag was still (incorrectly) touched — which would
    otherwise cause that file to be skipped forever on every resume.
    """
    try:
        test_raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        del test_raw
        return True
    except Exception:
        return False


# ==================================================
# Core per-file preprocessing
# ==================================================

def preprocess_raw(
    raw: mne.io.BaseRaw,
    output_dir: Path,
    dataset_name: str,
    subject: str,
    task: str,
    run_id: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[mne.io.BaseRaw]:
    """
    Full preprocessing for a single raw recording:
    - EOG/ECG channel typing
    - Band-pass filter
    - ICA (EOG + ECG)
    - Average reference
    - Epoching (if event mappings are known)
    - ERP + topography + QC

    Returns cleaned raw or None if skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = task

    done_flag = output_dir / "DONE"
    cleaned_fif = output_dir / "cleaned_raw.fif"

    if done_flag.exists() and cleaned_fif.exists() and not overwrite:
        if _is_valid_fif(cleaned_fif):
            _LOGGER.info(
                f"[SKIP] {dataset_name} {subject} {task} {run_id} – DONE flag present."
            )
            return None
        else:
            _LOGGER.warning(
                f"[REDO] {dataset_name} {subject} {task} {run_id} – "
                f"DONE flag present but {cleaned_fif.name} is corrupted/unreadable; "
                f"re-running this recording."
            )
            done_flag.unlink(missing_ok=True)

    _LOGGER.info(
        f"[START] {dataset_name} {subject} {task} {run_id}"
    )

    # --------------------------------------------------
    # Channel typing
    # --------------------------------------------------
    _mark_eog_ecg_channels(raw, dataset_name)

    # --------------------------------------------------
    # Montage (electrode positions) — must come after channel typing,
    # so EEG channel count used by plot_topomap matches the montage
    # positions actually applied.
    # --------------------------------------------------
    _apply_montage(raw, dataset_name, subject, task, run_id)

    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------
    raw.filter(LOW_CUT, HIGH_CUT, fir_design="firwin")

    # --------------------------------------------------
    # ICA (EOG + ECG)
    # --------------------------------------------------
    picks_eeg = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        eog=False,
        ecg=False,
        exclude="bads",
    )
    ica = ICA(
        n_components=ICA_COMPONENTS,
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks_eeg)

    # EOG components
    try:
        eog_epochs = None
        eog_picks = mne.pick_types(raw.info, eog=True)
        if eog_picks.size > 0:
            eog_epochs = create_eog_epochs(raw, reject=None)
        eog_inds: List[int] = []
        if eog_epochs is not None and len(eog_epochs) > 0:
            eog_inds, _ = ica.find_bads_eog(eog_epochs)
        # ECG components
        ecg_inds: List[int] = []
        ecg_picks = mne.pick_types(raw.info, ecg=True)
        if ecg_picks.size > 0:
            ecg_epochs = create_ecg_epochs(raw, reject=None)
            if len(ecg_epochs) > 0:
                ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
        ica.exclude = list(sorted(set(eog_inds + ecg_inds)))
        _LOGGER.info(
            f"[ICA] {dataset_name} {subject} {task} {run_id} — "
            f"excluding {len(ica.exclude)} component(s): {ica.exclude} "
            f"(EOG ch: {eog_picks.size}, ECG ch: {ecg_picks.size})"
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            f"[WARN] ICA artifact detection failed "
            f"{dataset_name} {subject} {task} {run_id}: {exc}"
        )
        ica.exclude = []

    ica.save(output_dir / "ica.fif", overwrite=True)

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # --------------------------------------------------
    # Average reference
    # --------------------------------------------------
    raw_clean.set_eeg_reference("average")

    # Save cleaned raw
    raw_clean.save(cleaned_fif, overwrite=True)

    # --------------------------------------------------
    # Epoching + ERPs (if events known)
    # --------------------------------------------------
    event_mapping = _get_event_mapping(dataset_name, task)
    epochs = None
    if event_mapping:
        try:
            events, event_id_all = mne.events_from_annotations(
                raw_clean,
                verbose="ERROR",
            )
            present_codes = set(event_id_all.values())

            # Build event_id as condition/code sub-labels, one integer
            # code each — mne.Epochs requires event_id values to be a
            # single int per label, NOT a list of ints. A condition with
            # multiple underlying codes (e.g. P3 target=[11,22,33,44,55])
            # gets split into "target/11", "target/22", etc.; these are
            # re-grouped back into the parent condition in
            # _compute_erps_and_qc for averaging/ERP purposes.
            epoch_event_id: Dict[str, int] = {}
            for condition, codes in event_mapping.items():
                for code in codes:
                    if code in present_codes:
                        epoch_event_id[f"{condition}/{code}"] = code

            if epoch_event_id:
                reject = {"eeg": REJECT_UV}
                epochs = mne.Epochs(
                    raw_clean,
                    events,
                    event_id=epoch_event_id,
                    tmin=TMIN,
                    tmax=TMAX,
                    baseline=BASELINE,
                    reject=reject,
                    preload=True,
                )
                epochs.save(output_dir / "epochs-epo.fif", overwrite=True)

                # --------------------------------------------------
                # ERPs, difference waves, QC and topography
                # --------------------------------------------------
                _compute_erps_and_qc(
                    epochs,
                    output_dir,
                    dataset_name,
                    subject,
                    task,
                    run_id,
                )
            else:
                _LOGGER.info(
                    f"[INFO] No matching event codes for {dataset_name} "
                    f"{subject} {task} {run_id}; skipping epoching."
                )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f"[WARN] Epoching failed {dataset_name} {subject} {task} "
                f"{run_id}: {exc}"
            )

    done_flag.touch()
    _LOGGER.info(
        f"[DONE] {dataset_name} {subject} {task} {run_id}"
    )
    return raw_clean


def _compute_erps_and_qc(
    epochs: mne.Epochs,
    output_dir: Path,
    dataset_name: str,
    subject: str,
    task: str,
    run_id: str,
) -> None:
    """
    Save condition ERPs, optional difference waves, topomaps, and QC CSV.

    epochs.event_id may contain "condition/code" sub-labels (e.g.
    "target/11", "target/22") when a condition has multiple underlying
    event codes — these are grouped back into their parent condition
    ("target") here for averaging, since mne.Epochs itself only allows
    a single int per event_id label.
    """
    qc_csv = RESULTS_ROOT / dataset_name / "qc_epochs.csv"
    fieldnames = [
        "dataset",
        "subject",
        "task",
        "run",
        "condition",
        "n_epochs",
    ]

    # Group sub-labels ("condition/code") back to their parent condition.
    from collections import defaultdict
    condition_epochs: Dict[str, list] = defaultdict(list)
    condition_n_total: Dict[str, int] = defaultdict(int)
    for key in epochs.event_id:
        parent = key.split("/")[0]
        try:
            ep = epochs[key]
            condition_n_total[parent] += len(ep.drop_log)
            if len(ep) > 0:
                condition_epochs[parent].append(ep)
        except Exception:
            pass

    evokeds: Dict[str, mne.Evoked] = {}

    # Per-condition ERPs + plots
    for condition in list(condition_n_total.keys()):
        ep_list = condition_epochs.get(condition, [])
        n_total = condition_n_total[condition]
        n_epochs = sum(len(e) for e in ep_list)

        # QC row (written regardless of whether ERP could be computed)
        row = {
            "dataset": dataset_name,
            "subject": subject,
            "task": task,
            "run": run_id,
            "condition": condition,
            "n_epochs": n_epochs,
        }
        _append_qc_row(qc_csv, row, fieldnames)

        if not ep_list:
            _LOGGER.warning(
                f"[WARN] 0/{n_total} epochs survived for {dataset_name} "
                f"{subject} {task} {run_id} condition={condition} — "
                f"skipping ERP for this condition only (other conditions "
                f"still run)."
            )
            continue

        ep_cat = ep_list[0] if len(ep_list) == 1 else mne.concatenate_epochs(ep_list)
        evoked = ep_cat.average()
        evokeds[condition] = evoked

        cond_dir = output_dir
        evoked_fname = cond_dir / f"{condition}-ave.fif"
        evoked.save(evoked_fname, overwrite=True)

        # ERP plot
        fig = evoked.plot(
            spatial_colors=True,
            show=False,
        )
        fig.savefig(cond_dir / f"{condition}_erp.png", dpi=150)
        plt.close(fig)

        # Topography at default times
        try:
            fig = evoked.plot_topomap(
                times=DEFAULT_TOPO_TIMES,
                ch_type="eeg",
                show=False,
            )
            fig.savefig(cond_dir / f"{condition}_topomap.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f"[WARN] Topomap failed {dataset_name} {subject} {task} "
                f"{run_id} condition={condition}: {exc}"
            )

    # Difference wave for typical oddball designs
    if "target" in evokeds and "standard" in evokeds:
        diff = mne.combine_evoked(
            [evokeds["target"], evokeds["standard"]],
            weights=[1, -1],
        )
        diff.save(output_dir / "difference-ave.fif", overwrite=True)

        fig = diff.plot(spatial_colors=True, show=False)
        fig.savefig(output_dir / "difference_wave.png", dpi=150)
        plt.close(fig)

        try:
            fig = diff.plot_topomap(
                times=DEFAULT_TOPO_TIMES,
                ch_type="eeg",
                show=False,
            )
            fig.savefig(output_dir / "difference_topomap.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f"[WARN] Difference topomap failed "
                f"{dataset_name} {subject} {task} {run_id}: {exc}"
            )
    elif "target" in condition_n_total and "standard" in condition_n_total:
        _LOGGER.warning(
            f"[WARN] Cannot compute difference wave for "
            f"{dataset_name} {subject} {task} {run_id} — "
            f"target or standard had 0 surviving epochs."
        )


# ==================================================
# Grand-average helper
# ==================================================

def compute_grand_averages(
    dataset_name: str,
    task_to_events: Dict[str, Dict[str, List[int]]],
) -> None:
    """
    Compute grand-average ERPs and topomaps for each condition across subjects.

    Looks for:
        results_batch/{dataset}/sub-*/{task}/{condition}-ave.fif
    """
    root = RESULTS_ROOT / dataset_name
    if not root.exists():
        _LOGGER.info(f"[GA] No results dir for {dataset_name}, skipping.")
        return

    for task, cond_map in task_to_events.items():
        if not cond_map:
            continue

        for condition in cond_map.keys():
            pattern = f"sub-*/{task}/{condition}-ave.fif"
            files = sorted(root.glob(pattern))
            if not files:
                _LOGGER.info(
                    f"[GA] No evoked files for {dataset_name} {task} {condition}"
                )
                continue

            try:
                evokeds = [
                    mne.read_evokeds(f, condition=0, verbose="ERROR")
                    for f in files
                ]
                grand = mne.grand_average(evokeds)

                out_dir = root / "grand_averages" / task
                out_dir.mkdir(parents=True, exist_ok=True)

                grand_fname = out_dir / f"{condition}_grand-ave.fif"
                grand.save(grand_fname, overwrite=True)

                # ERP plot
                fig = grand.plot(spatial_colors=True, show=False)
                fig.savefig(out_dir / f"{condition}_grand_erp.png", dpi=150)
                plt.close(fig)

                # Topomap
                try:
                    fig = grand.plot_topomap(
                        times=DEFAULT_TOPO_TIMES,
                        ch_type="eeg",
                        show=False,
                    )
                    fig.savefig(out_dir / f"{condition}_grand_topomap.png", dpi=150)
                    plt.close(fig)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        f"[WARN] Grand topomap failed "
                        f"{dataset_name} {task} {condition}: {exc}"
                    )

                _LOGGER.info(
                    f"[GA] Saved grand average for {dataset_name} {task} "
                    f"{condition} ({len(files)} subjects)"
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    f"[GA WARN] {dataset_name} {task} {condition}: {exc}"
                )