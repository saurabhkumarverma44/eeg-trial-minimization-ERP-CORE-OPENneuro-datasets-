# dl_models/minimization_analyzer.py

from typing import List, Dict, Tuple

import os
import json

import numpy as np
import matplotlib.pyplot as plt


def _compute_ci(
    values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute mean, standard error, and symmetric (1-alpha)*100% confidence interval.

    For K runs per fraction, we use normal approximation: mean ± z * SE.
    """
    if values.size == 0:
        return np.nan, np.nan, (np.nan, np.nan)

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    se = std / np.sqrt(values.size) if values.size > 1 else 0.0

    # z-score for two-sided alpha (0.05 -> 1.96)
    from math import sqrt
    if alpha == 0.05:
        z = 1.96
    else:
        # crude approximation using inverse CDF would require scipy; we keep alpha fixed in config
        z = 1.96

    ci_low = mean - z * se
    ci_high = mean + z * se
    return mean, se, (ci_low, ci_high)


def analyze_minimization_results(
    results_list: List[Dict],
    output_dir,
    target_name: str,
    threshold_pct: float,
    ci_alpha: float,
) -> Tuple[Dict, "np.ndarray"]:
    """
    Analyze trial minimization results with multiple seeds per fraction.

    Parameters
    ----------
    results_list : list of dicts with keys:
        "trial_fraction", "n_train_trials", "seed", "test_acc"
    output_dir : directory for plots + summary JSON
    target_name : e.g. ERP_CORE_P3
    threshold_pct : e.g. 0.95 (95% of baseline accuracy)
    ci_alpha : e.g. 0.05 (95% CI)

    Returns
    -------
    summary_dict : dict with baseline metrics and minimum stable fraction
    df_res : numpy structured array for further analysis/plotting
    """

    os.makedirs(output_dir, exist_ok=True)

    # Group accuracies by fraction
    fractions = sorted(set(r["trial_fraction"] for r in results_list))
    frac_to_accs = {f: [] for f in fractions}
    frac_to_ntrials = {f: None for f in fractions}

    for r in results_list:
        f = r["trial_fraction"]
        frac_to_accs[f].append(r["test_acc"])
        if frac_to_ntrials[f] is None:
            frac_to_ntrials[f] = r["n_train_trials"]

    # Compute baseline (fraction==1.0)
    if 1.0 not in frac_to_accs or len(frac_to_accs[1.0]) == 0:
        raise ValueError("Baseline fraction 1.0 not found in results_list.")

    baseline_mean, baseline_se, baseline_ci = _compute_ci(
        np.asarray(frac_to_accs[1.0], dtype=float),
        alpha=ci_alpha,
    )

    # Collect per-fraction stats
    table_rows = []
    for f in fractions:
        accs_f = np.asarray(frac_to_accs[f], dtype=float)
        mean_f, se_f, ci_f = _compute_ci(accs_f, alpha=ci_alpha)

        row = {
            "trial_fraction": f,
            "n_train_trials": frac_to_ntrials[f],
            "mean_test_acc": mean_f,
            "se_test_acc": se_f,
            "ci_low": ci_f[0],
            "ci_high": ci_f[1],
        }
        table_rows.append(row)

    # Determine minimum stable fraction:
    # - mean_test_acc >= threshold_pct * baseline_mean
    # - ci_low >= threshold_pct * baseline_mean (optional stronger condition)
    threshold_acc = threshold_pct * baseline_mean

    min_stable_fraction = None
    for row in sorted(table_rows, key=lambda r: r["trial_fraction"], reverse=True):
        f = row["trial_fraction"]
        mean_f = row["mean_test_acc"]
        ci_low_f = row["ci_low"]

        if mean_f >= threshold_acc and ci_low_f >= threshold_acc:
            min_stable_fraction = f  # smallest f encountered when iterating descending
        else:
            # Once below threshold, keep looking at higher fractions only
            pass

    # Create numpy structured array for convenience/plotting
    dtype = [
        ("trial_fraction", float),
        ("n_train_trials", int),
        ("mean_test_acc", float),
        ("se_test_acc", float),
        ("ci_low", float),
        ("ci_high", float),
    ]
    df_res = np.zeros(len(table_rows), dtype=dtype)
    for i, r in enumerate(sorted(table_rows, key=lambda r: r["trial_fraction"], reverse=True)):
        df_res[i]["trial_fraction"] = r["trial_fraction"]
        df_res[i]["n_train_trials"] = r["n_train_trials"]
        df_res[i]["mean_test_acc"] = r["mean_test_acc"]
        df_res[i]["se_test_acc"] = r["se_test_acc"]
        df_res[i]["ci_low"] = r["ci_low"]
        df_res[i]["ci_high"] = r["ci_high"]

    # Save summary JSON
    summary_dict = {
        "target_name": target_name,
        "baseline_mean_acc": baseline_mean,
        "baseline_ci_low": baseline_ci[0],
        "baseline_ci_high": baseline_ci[1],
        "threshold_pct": threshold_pct,
        "threshold_acc": threshold_acc,
        "min_stable_fraction": float(min_stable_fraction) if min_stable_fraction is not None else None,
        "min_stable_n_trials": (
            int(frac_to_ntrials[min_stable_fraction]) if min_stable_fraction is not None else None
        ),
    }

    summary_path = os.path.join(output_dir, f"{target_name}_minimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[Phase 5] Minimization summary saved -> {summary_path}")

    # Plot fraction vs mean accuracy (+ CI)
    plt.figure(figsize=(7, 5))
    fracs_sorted = [r["trial_fraction"] for r in sorted(table_rows, key=lambda r: r["trial_fraction"])]
    mean_sorted = [r["mean_test_acc"] for r in sorted(table_rows, key=lambda r: r["trial_fraction"])]
    ci_low_sorted = [r["ci_low"] for r in sorted(table_rows, key=lambda r: r["trial_fraction"])]
    ci_high_sorted = [r["ci_high"] for r in sorted(table_rows, key=lambda r: r["trial_fraction"])]

    plt.plot(fracs_sorted, mean_sorted, marker="o", label="Mean test accuracy")
    plt.fill_between(fracs_sorted, ci_low_sorted, ci_high_sorted, alpha=0.2, label="95% CI")
    plt.axhline(baseline_mean, color="gray", linestyle="--", label="Baseline mean")
    plt.axhline(threshold_acc, color="red", linestyle="--", label=f"{threshold_pct*100:.0f}% of baseline")
    plt.xlabel("Trial fraction")
    plt.ylabel("Test accuracy")
    plt.title(f"Trial minimization: {target_name}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, f"{target_name}_trial_minimization.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Phase 5] Minimization curve saved -> {plot_path}")

    return summary_dict, df_res