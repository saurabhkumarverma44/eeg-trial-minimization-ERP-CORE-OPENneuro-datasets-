#!/usr/bin/env python3

"""
Stage 1 Classical ML status report.

Generates a plain-text report you can copy-paste, covering:
1. Paradigms completed so far and their names.
2. Full table of paradigm | step | model | accuracy | balanced_accuracy | AUC.
3. NFS status and values per step for at least one paradigm, if available.
4. LRP handling (classification + NFS N/A).
5. Exact file paths where results are saved.
6. Any errors/warnings/anomalies encountered so far.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
OUT_ROOT = PROJECT_ROOT / "results_stage4_classical_ml"

MODEL_DIRS = {
    "random_forest": OUT_ROOT / "random_forest",
    "xgboost": OUT_ROOT / "xgboost",
    "svm": OUT_ROOT / "svm",
}

NFS_DIR = OUT_ROOT / "nfs"
MARKER_DIR = OUT_ROOT / "markers"
ERROR_DIR = OUT_ROOT / "errors"

# Import Stage 4 modules to get paradigms and steps
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from stage4_classical_ml.dataset import get_paradigms, STEPS  # type: ignore

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_all_summaries():
    """Load all per-model per-paradigm summary CSVs into one DataFrame."""
    rows = []

    for model_name, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            continue

        for paradigm_dir in model_dir.iterdir():
            if not paradigm_dir.is_dir():
                continue

            summary_csv = paradigm_dir / f"{paradigm_dir.name}_summary.csv"
            if not summary_csv.exists():
                continue

            df = pd.read_csv(summary_csv)
            df["model"] = model_name
            df["paradigm"] = paradigm_dir.name
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def list_completed_paradigms(all_df):
    """
    Paradigms 'completed' = all (model, step) combinations have markers/summary rows.
    """
    completed = []
    all_paradigms = get_paradigms()

    for paradigm in all_paradigms:
        sub = all_df[all_df["dataset_task"] == paradigm]
        ok = True
        for model_name in MODEL_DIRS.keys():
            for step in STEPS:
                mask = (
                    (sub["model"] == model_name)
                    & (sub["step"].astype(str) == str(step))
                )
                if mask.sum() == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            completed.append(paradigm)

    return completed, all_paradigms


def load_nfs_status():
    """Load NFS CSVs if they exist."""
    nfs_rows = []
    if not NFS_DIR.exists():
        return pd.DataFrame()

    for csv in NFS_DIR.glob("*_nfs.csv"):
        try:
            df = pd.read_csv(csv)
            df["model"] = csv.name.replace("_nfs.csv", "")
            nfs_rows.append(df)
        except Exception:
            continue

    if not nfs_rows:
        return pd.DataFrame()

    return pd.concat(nfs_rows, ignore_index=True)


def list_errors():
    """List error files and a brief note."""
    out = []
    if not ERROR_DIR.exists():
        return out

    for path in ERROR_DIR.glob("*.error.txt"):
        try:
            first_line = path.read_text().splitlines()[0] if path.read_text() else ""
        except Exception:
            first_line = "<could not read>"
        out.append((path.name, first_line))
    return out


def marker_count():
    if not MARKER_DIR.exists():
        return 0
    return len(list(MARKER_DIR.glob("*.done")))


# ---------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------


def main():
    all_df = load_all_summaries()
    nfs_df = load_nfs_status()
    errors = list_errors()
    n_markers = marker_count()

    print("Stage 1 Classical ML Status Report")
    print("==================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results root: {OUT_ROOT}")
    print(f"Markers dir:  {MARKER_DIR}")
    print(f"Errors dir:   {ERROR_DIR}")
    print(f"NFS dir:      {NFS_DIR}")
    print(f"Total marker files currently: {n_markers}")
    print()

    # 1. Paradigms completed so far
    print("1) Paradigms completed so far (all models, all steps):")
    if all_df.empty:
        print("   No summary CSVs found; no paradigms appear completed yet.")
    else:
        completed, all_paradigms = list_completed_paradigms(all_df)
        print(f"   Total paradigms configured: {len(all_paradigms)}")
        print(f"   Paradigms fully completed so far: {len(completed)}")
        for name in completed:
            print(f"   - {name}")
    print()

    # 2. Full table of per-step metrics
    print("2) Per-model, per-paradigm, per-step metrics (completed so far):")
    if all_df.empty:
        print("   No metrics found.")
    else:
        print("   Columns: paradigm | model | step | accuracy_mean | balanced_accuracy_mean | auc_mean")
        for _, row in all_df.sort_values(
            ["dataset_task", "model", "step"]
        ).iterrows():
            parad = str(row.get("dataset_task", row.get("paradigm", "")))
            model = str(row.get("model", ""))
            step = str(row.get("step", ""))
            acc = float(row.get("accuracy_mean", np.nan))
            bal = float(row.get("balanced_accuracy_mean", np.nan))
            auc = float(row.get("auc_mean", np.nan))
            print(
                f"   {parad} | {model} | step={step} | "
                f"accuracy={acc:.4f} | "
                f"balanced_accuracy={bal:.4f} | "
                f"auc={auc:.4f}"
            )
    print()

    # 3. NFS status
    print("3) NFS status:")
    if nfs_df.empty:
        print("   NFS has not been computed yet, or no NFS CSVs are present.")
    else:
        paradigms_with_nfs = sorted(nfs_df["dataset_task"].unique())
        print(f"   NFS computed for paradigms: {', '.join(paradigms_with_nfs)}")
        # Show NFS values per step for the first paradigm with data
        first_paradigm = paradigms_with_nfs[0]
        sub = nfs_df[nfs_df["dataset_task"] == first_paradigm]
        print(f"   Example: NFS for {first_paradigm}:")
        for _, row in sub.sort_values(["model", "step"]).iterrows():
            model = row["model"]
            step = row["step"]
            nfs_val = row.get("nfs", np.nan)
            ci_low = row.get("ci_low", np.nan)
            ci_high = row.get("ci_high", np.nan)
            note = str(row.get("note", ""))
            print(
                f"      {model} | step={step} | "
                f"NFS={nfs_val:.4f} | CI=[{ci_low:.4f}, {ci_high:.4f}] | note={note}"
            )
    print()

    # 4. LRP handling
    print("4) LRP handling:")
    if all_df.empty:
        print("   Cannot confirm; no summary CSVs loaded.")
    else:
        lrp_rows = all_df[all_df["dataset_task"] == "ERP_CORE__LRP"]
        if lrp_rows.empty:
            print("   ERP_CORE__LRP has no summary rows; it may not have been run yet.")
        else:
            print("   ERP_CORE__LRP is being classified normally (RF/XGB/SVM summary rows present).")
            if not nfs_df.empty:
                lrp_nfs = nfs_df[nfs_df["dataset_task"] == "ERP_CORE__LRP"]
                if lrp_nfs.empty:
                    print("   NFS for LRP is correctly marked N/A (no NFS rows).")
                else:
                    print("   WARNING: NFS rows exist for LRP; they should typically be N/A.")
            else:
                print("   NFS CSVs not present; LRP NFS is effectively N/A.")
    print()

    # 5. Exact file paths
    print("5) Exact file paths where results are saved:")
    print(f"   Summary CSVs per model/paradigm:")
    for model_name, model_dir in MODEL_DIRS.items():
        print(f"   - {model_name}: {model_dir}/<PARADIGM>/{'<PARADIGM>_summary.csv'}")
    print(f"   Markers (completion flags): {MARKER_DIR}/<model>__<paradigm>__step<STEP>.done")
    print(f"   NFS results: {NFS_DIR}/<model>_nfs.csv")
    print(f"   Errors: {ERROR_DIR}/*.error.txt")
    print()

    # 6. Errors, warnings, anomalies
    print("6) Errors, warnings, anomalies encountered so far:")
    if not errors:
        print("   No error files found.")
    else:
        for name, first_line in errors:
            print(f"   - {name}: {first_line}")
    print()

if __name__ == "__main__":
    main()