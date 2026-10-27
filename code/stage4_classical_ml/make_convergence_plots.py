#!/usr/bin/env python3
"""
Generates accuracy/balanced-accuracy convergence plots and NFS convergence
plots from results_stage4_classical_ml/. Safe to run anytime, including
mid-way through a still-running training job (reads whatever CSVs exist).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
OUT_ROOT = PROJECT_ROOT / "results_stage4_classical_ml"
CONV_DIR = OUT_ROOT / "convergence_plots"
CONV_DIR.mkdir(parents=True, exist_ok=True)

STEP_ORDER = ["5", "10", "20", "30", "50", "75", "100", "150", "200", "full"]


def plot_classification_convergence(model_name):
    model_dir = OUT_ROOT / model_name
    if not model_dir.exists():
        return
    for paradigm_dir in sorted(model_dir.iterdir()):
        if not paradigm_dir.is_dir():
            continue
        summary_csv = paradigm_dir / f"{paradigm_dir.name}_summary.csv"
        if not summary_csv.exists():
            continue
        df = pd.read_csv(summary_csv)
        df["step"] = pd.Categorical(df["step"].astype(str), categories=STEP_ORDER, ordered=True)
        df = df.sort_values("step")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["step"].astype(str), df["accuracy_mean"], marker="o", label="Accuracy")
        ax.plot(df["step"].astype(str), df["balanced_accuracy_mean"], marker="s", label="Balanced accuracy")
        ax.set_xlabel("Trial count step")
        ax.set_ylabel("Score")
        ax.set_title(f"{model_name} — {paradigm_dir.name}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = CONV_DIR / f"{model_name}__{paradigm_dir.name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_nfs_convergence():
    nfs_dir = OUT_ROOT / "nfs"
    if not nfs_dir.exists():
        return
    for nfs_csv in sorted(nfs_dir.glob("*_nfs.csv")):
        model_name = nfs_csv.stem.replace("_nfs", "")
        df = pd.read_csv(nfs_csv)
        df = df[pd.to_numeric(df["nfs"], errors="coerce").notna()]
        if df.empty:
            continue
        for paradigm, g in df.groupby("dataset_task"):
            g = g.copy()
            g["step"] = pd.Categorical(g["step"].astype(str), categories=STEP_ORDER, ordered=True)
            g = g.sort_values("step")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(g["step"].astype(str), g["nfs"].astype(float), marker="o", color="darkred")
            ax.fill_between(
                g["step"].astype(str),
                g["ci_low"].astype(float),
                g["ci_high"].astype(float),
                alpha=0.2,
                color="darkred",
            )
            ax.set_xlabel("Trial count step")
            ax.set_ylabel("NFS (Pearson r vs reference topography)")
            ax.set_title(f"NFS convergence — {model_name} — {paradigm}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            out_path = CONV_DIR / f"NFS__{model_name}__{paradigm}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    for m in ["random_forest", "xgboost", "svm"]:
        plot_classification_convergence(m)
    plot_nfs_convergence()
    print("Convergence plots complete.")