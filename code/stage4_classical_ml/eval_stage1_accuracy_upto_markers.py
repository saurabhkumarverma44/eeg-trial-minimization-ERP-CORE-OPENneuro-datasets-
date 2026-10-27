#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
OUT_ROOT = PROJECT_ROOT / "results_stage4_classical_ml"

MODEL_DIRS = {
    "random_forest": OUT_ROOT / "random_forest",
    "xgboost": OUT_ROOT / "xgboost",
    "svm": OUT_ROOT / "svm",
}

def main():
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
            rows.append(df)

    if not rows:
        print("No summary CSVs found under results_stage4_classical_ml.")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # Print per-model aggregate accuracy / balanced accuracy
    print("\nPer-model accuracy up to current markers:\n")
    for model_name in sorted(all_df["model"].unique()):
        sub = all_df[all_df["model"] == model_name]
        acc = sub["accuracy_mean"].mean()
        bal = sub["balanced_accuracy_mean"].mean()
        print(
            f"{model_name}: "
            f"accuracy_mean={acc:.4f}, "
            f"balanced_accuracy_mean={bal:.4f}"
        )

    # Optional: save full table for later analysis
    out_csv = OUT_ROOT / "stage4_accuracy_upto_current_markers.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed table to {out_csv}")

if __name__ == "__main__":
    main()