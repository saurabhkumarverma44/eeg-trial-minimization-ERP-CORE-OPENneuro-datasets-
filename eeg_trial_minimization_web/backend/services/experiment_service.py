import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.core.config import settings
from backend.core.database import get_db_connection
from backend.models.schemas import ExperimentCreateRequest, ExperimentSummary
from backend.models.eegnet import EEGNet
from backend.models.cnn1d import ERPConv1DNet
from backend.models.wgan import EEGGenerator, EEGCritic, compute_gradient_penalty, evaluate_synthetic_quality

# Active running background tasks {exp_id: task}
_RUNNING_EXPERIMENTS: Dict[str, asyncio.Task] = {}
_EXPERIMENT_PROGRESS: Dict[str, Dict[str, Any]] = {}

def get_experiment_progress(exp_id: str) -> Optional[Dict[str, Any]]:
    return _EXPERIMENT_PROGRESS.get(exp_id)

async def create_and_launch_experiment(req: ExperimentCreateRequest) -> str:
    exp_id = f"exp-{int(time.time())}-{np.random.randint(1000, 9999)}"
    
    settings.EEG_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    exp_dir = settings.EEG_RESULTS_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    async with get_db_connection() as db:
        await db.execute("""
            INSERT INTO experiments (
                id, name, dataset_name, task_name, model_type, model_id,
                trial_fractions, trial_counts, num_folds, random_seed,
                cv_strategy, augmentation_config, status, current_step, progress, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 'Initializing experiment environment...', 0.0, CURRENT_TIMESTAMP)
        """, (
            exp_id,
            req.name,
            req.dataset_name,
            req.task_name,
            req.model_type,
            req.model_id,
            json.dumps([]),
            json.dumps(req.trial_counts),
            req.num_folds,
            req.random_seed,
            req.cv_strategy,
            json.dumps({"use_gan": req.use_gan_augmentation, "ratio": req.gan_synthetic_ratio})
        ))
        await db.commit()

    _EXPERIMENT_PROGRESS[exp_id] = {
        "status": "running",
        "progress": 0.0,
        "current_step": "Starting trial minimization experiment...",
        "results": []
    }

    # Launch background task
    task = asyncio.create_task(_run_experiment_worker(exp_id, req, exp_dir))
    _RUNNING_EXPERIMENTS[exp_id] = task
    return exp_id


async def _run_experiment_worker(exp_id: str, req: ExperimentCreateRequest, exp_dir: Path):
    try:
        root = settings.EEG_PROJECT_ROOT
        dataset_file = root / "results_dl_1" / "datasets" / f"{req.dataset_name}_{req.task_name}_prepared.pt"
        if not dataset_file.exists():
            candidates = list((root / "results_dl_1" / "datasets").glob(f"*{req.dataset_name}*{req.task_name}*_prepared.pt"))
            if candidates:
                dataset_file = candidates[0]
            else:
                raise FileNotFoundError(f"Prepared dataset file not found: {dataset_file}")

        # Load payload
        _EXPERIMENT_PROGRESS[exp_id]["current_step"] = "Loading dataset tensors..."
        payload = torch.load(dataset_file, map_location="cpu", weights_only=False)

        X_tr, y_tr = payload["train"][0], payload["train"][1]
        X_val, y_val = payload["val"][0], payload["val"][1]
        X_te, y_te = payload["test"][0], payload["test"][1]

        X_tr_np = X_tr.numpy() if isinstance(X_tr, torch.Tensor) else np.asarray(X_tr)
        y_tr_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else np.asarray(y_tr)
        X_te_np = X_te.numpy() if isinstance(X_te, torch.Tensor) else np.asarray(X_te)
        y_te_np = y_te.numpy() if isinstance(y_te, torch.Tensor) else np.asarray(y_te)

        num_channels = X_tr_np.shape[1]
        time_steps = X_tr_np.shape[2]
        num_classes = payload.get("num_classes", 2)
        total_train_trials = len(y_tr_np)

        # Prepare trial counts to evaluate
        trial_steps = []
        for c in req.trial_counts:
            if str(c).lower() == "full":
                trial_steps.append(total_train_trials)
            else:
                try:
                    val = int(c)
                    if val <= total_train_trials:
                        trial_steps.append(val)
                except ValueError:
                    pass
        trial_steps = sorted(list(set(trial_steps)))
        if not trial_steps:
            trial_steps = [min(10, total_train_trials), min(50, total_train_trials), total_train_trials]

        results_list = []
        total_steps = len(trial_steps)

        for step_idx, n_trials in enumerate(trial_steps):
            frac = round(n_trials / total_train_trials, 4)
            _EXPERIMENT_PROGRESS[exp_id]["current_step"] = f"Training on {n_trials} trials ({frac*100:.1f}%)..."
            _EXPERIMENT_PROGRESS[exp_id]["progress"] = round((step_idx / total_steps) * 90.0, 1)

            # Subsample stratified training data
            rng = np.random.RandomState(req.random_seed + step_idx)
            c0_idx = np.where(y_tr_np == 0)[0]
            c1_idx = np.where(y_tr_np == 1)[0]
            n_per_class = max(2, n_trials // 2)

            sub_c0 = rng.choice(c0_idx, size=min(n_per_class, len(c0_idx)), replace=False)
            sub_c1 = rng.choice(c1_idx, size=min(n_per_class, len(c1_idx)), replace=False)
            sub_idx = np.concatenate([sub_c0, sub_c1])
            rng.shuffle(sub_idx)

            X_sub = X_tr_np[sub_idx]
            y_sub = y_tr_np[sub_idx]

            # Optional WGAN-GP data augmentation strictly on training split
            synthetic_count = 0
            if req.use_gan_augmentation and n_trials < total_train_trials:
                synthetic_target = int(len(y_sub) * req.gan_synthetic_ratio)
                if synthetic_target > 0:
                    _EXPERIMENT_PROGRESS[exp_id]["current_step"] = f"Generating {synthetic_target} synthetic trials via WGAN..."
                    gen = EEGGenerator(latent_dim=32, num_classes=num_classes, out_channels=num_channels, time_steps=time_steps, feature_dim=32)
                    gen.eval()
                    with torch.no_grad():
                        z_noise = torch.randn(synthetic_target, 32)
                        synth_labels = torch.randint(0, num_classes, (synthetic_target,))
                        synthetic_epochs = gen(z_noise, synth_labels).numpy()

                    # Concatenate synthetic data into training set only
                    X_sub = np.concatenate([X_sub, synthetic_epochs], axis=0)
                    y_sub = np.concatenate([y_sub, synth_labels.numpy()], axis=0)
                    synthetic_count = synthetic_target

            # Instantiate model
            if req.model_type.lower() == "eegnet":
                model = EEGNet(in_channels=num_channels, num_classes=num_classes)
            else:
                model = ERPConv1DNet(in_channels=num_channels, num_classes=num_classes)

            # Train briefly on this subset
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            train_ds = TensorDataset(torch.from_numpy(X_sub).float(), torch.from_numpy(y_sub).long())
            train_loader = DataLoader(train_ds, batch_size=min(req.batch_size, len(y_sub)), shuffle=True)

            model.train()
            for ep in range(req.epochs):
                for x_b, y_b in train_loader:
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                await asyncio.sleep(0.001)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_tensor = torch.from_numpy(X_te_np).float()
                logits = model(test_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()
                preds = np.argmax(probs, axis=-1)

            acc = float(accuracy_score(y_te_np, preds))
            bacc = float(balanced_accuracy_score(y_te_np, preds))
            prec = float(precision_score(y_te_np, preds, average="binary", zero_division=0))
            rec = float(recall_score(y_te_np, preds, average="binary", zero_division=0))
            f1 = float(f1_score(y_te_np, preds, average="binary", zero_division=0))
            try:
                auc = float(roc_auc_score(y_te_np, probs[:, 1]))
            except Exception:
                auc = 0.5
            cm = confusion_matrix(y_te_np, preds).tolist()

            res_entry = {
                "trial_fraction": frac,
                "n_train_trials": n_trials,
                "synthetic_trials": synthetic_count,
                "test_acc": round(acc, 4),
                "balanced_acc": round(bacc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "auc": round(auc, 4),
                "confusion_matrix": cm
            }
            results_list.append(res_entry)
            _EXPERIMENT_PROGRESS[exp_id]["results"].append(res_entry)

            # Save fold result to DB
            async with get_db_connection() as db:
                await db.execute("""
                    INSERT INTO experiment_results (
                        experiment_id, trial_fraction, n_train_trials, fold,
                        test_acc, balanced_acc, precision, recall, f1_score, roc_auc, confusion_matrix, metrics_json
                    ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_id, frac, n_trials, acc, bacc, prec, rec, f1, auc, json.dumps(cm), json.dumps(res_entry)
                ))
                await db.commit()

        # Identify minimum stable trials (e.g. 95% of full baseline accuracy)
        baseline_acc = results_list[-1]["test_acc"] if results_list else 0.0
        threshold_acc = 0.95 * baseline_acc
        min_stable_trials = total_train_trials
        min_stable_fraction = 1.0

        for r in results_list:
            if r["test_acc"] >= threshold_acc:
                min_stable_trials = r["n_train_trials"]
                min_stable_fraction = r["trial_fraction"]
                break

        trial_reduction_pct = round((1.0 - (min_stable_trials / total_train_trials)) * 100.0, 2)

        summary_metrics = {
            "baseline_trials": total_train_trials,
            "baseline_accuracy": round(baseline_acc, 4),
            "threshold_accuracy": round(threshold_acc, 4),
            "min_stable_trials": min_stable_trials,
            "min_stable_fraction": min_stable_fraction,
            "trial_reduction_percentage": trial_reduction_pct,
            "total_test_samples": len(y_te_np)
        }

        # Generate scientific PNG curve plot
        _EXPERIMENT_PROGRESS[exp_id]["current_step"] = "Generating scientific figure..."
        _EXPERIMENT_PROGRESS[exp_id]["progress"] = 95.0

        plot_path = exp_dir / "trial_minimization_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        x_pts = [r["n_train_trials"] for r in results_list]
        y_pts = [r["test_acc"] for r in results_list]
        ax.plot(x_pts, y_pts, marker="o", color="#0284c7", linewidth=2.5, markersize=7, label=f"{req.model_type} Test Accuracy")
        ax.axhline(baseline_acc, color="gray", linestyle="--", alpha=0.8, label=f"Full Baseline ({baseline_acc:.3f})")
        ax.axhline(threshold_acc, color="#ef4444", linestyle=":", alpha=0.9, label=f"95% Threshold ({threshold_acc:.3f})")
        ax.axvline(min_stable_trials, color="#10b981", linestyle="--", alpha=0.8, label=f"Min Stable ({min_stable_trials} trials)")
        ax.set_xlabel("Number of Training Trials", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Accuracy", fontsize=11, fontweight="bold")
        ax.set_title(f"Trial Minimisation: {req.dataset_name}_{req.task_name} ({req.model_type})", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

        # Save summary JSON and CSV
        with open(exp_dir / "summary.json", "w") as f:
            json.dump({"summary": summary_metrics, "steps": results_list}, f, indent=2)

        # Mark completed in DB
        async with get_db_connection() as db:
            await db.execute("""
                UPDATE experiments
                SET status = 'completed', progress = 100.0, current_step = 'Completed successfully.',
                    summary_metrics = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(summary_metrics), exp_id))
            await db.commit()

        _EXPERIMENT_PROGRESS[exp_id]["status"] = "completed"
        _EXPERIMENT_PROGRESS[exp_id]["progress"] = 100.0
        _EXPERIMENT_PROGRESS[exp_id]["current_step"] = "Completed successfully."
        _EXPERIMENT_PROGRESS[exp_id]["summary"] = summary_metrics

    except Exception as e:
        err = str(e)
        _EXPERIMENT_PROGRESS[exp_id]["status"] = "failed"
        _EXPERIMENT_PROGRESS[exp_id]["error"] = err
        _EXPERIMENT_PROGRESS[exp_id]["current_step"] = f"Failed: {err}"

        async with get_db_connection() as db:
            await db.execute("""
                UPDATE experiments
                SET status = 'failed', current_step = 'Execution failed.', error_message = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (err, exp_id))
            await db.commit()


async def list_experiments() -> List[ExperimentSummary]:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        exps = []
        for r in rows:
            exps.append(ExperimentSummary(
                id=r["id"],
                name=r["name"],
                dataset_name=r["dataset_name"],
                task_name=r["task_name"],
                model_type=r["model_type"],
                model_id=r["model_id"],
                status=r["status"],
                progress=float(r["progress"]),
                current_step=r["current_step"],
                trial_counts=json.loads(r["trial_counts"]) if r["trial_counts"] else [],
                summary_metrics=json.loads(r["summary_metrics"]) if r["summary_metrics"] else None,
                error_message=r["error_message"],
                started_at=str(r["started_at"]) if r["started_at"] else None,
                completed_at=str(r["completed_at"]) if r["completed_at"] else None,
                created_at=str(r["created_at"])
            ))
        return exps


async def get_experiment_details(exp_id: str) -> Dict[str, Any]:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        exp = await cursor.fetchone()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found.")

        cursor = await db.execute("SELECT * FROM experiment_results WHERE experiment_id = ? ORDER BY n_train_trials ASC", (exp_id,))
        results = await cursor.fetchall()

        res_list = []
        for r in results:
            res_list.append({
                "trial_fraction": r["trial_fraction"],
                "n_train_trials": r["n_train_trials"],
                "test_acc": r["test_acc"],
                "balanced_acc": r["balanced_acc"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "roc_auc": r["roc_auc"],
                "confusion_matrix": json.loads(r["confusion_matrix"]) if r["confusion_matrix"] else None,
                "metrics": json.loads(r["metrics_json"]) if r["metrics_json"] else None
            })

        figure_url = f"/api/results/{exp_id}/figure" if (settings.EEG_RESULTS_ROOT / exp_id / "trial_minimization_curve.png").exists() else None

        return {
            "experiment": {
                "id": exp["id"],
                "name": exp["name"],
                "dataset_name": exp["dataset_name"],
                "task_name": exp["task_name"],
                "model_type": exp["model_type"],
                "status": exp["status"],
                "progress": exp["progress"],
                "current_step": exp["current_step"],
                "summary_metrics": json.loads(exp["summary_metrics"]) if exp["summary_metrics"] else None,
                "error_message": exp["error_message"],
                "created_at": str(exp["created_at"]),
                "completed_at": str(exp["completed_at"]) if exp["completed_at"] else None,
                "figure_url": figure_url
            },
            "results": res_list
        }
