import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from configs.dl_config import TRAIN_CONFIG, RESULTS_DL_DIR, DATASET_TASKS
from dl_prep.subsampler import subsample_training_data
from dl_prep.eeg_dataset import create_dataloaders
from dl_models.cnn1d import ERPConv1DNet
from dl_models.trainer import Trainer, LabelSmoothingCrossEntropy
from dl_models.evaluator import evaluate_model
from dl_models.minimization_analyzer import analyze_minimization_results
TRIAL_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
def run_minimization_experiment(dataset_name, task_name, epochs=30, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_name = f"{dataset_name}_{task_name}"
    dataset_file = RESULTS_DL_DIR / "datasets" / f"{target_name}_prepared.pt"
    if not dataset_file.exists():
        print(f"[Warning] Prepared dataset file not found: {dataset_file}. Skipping.")
        return None
    payload = torch.load(dataset_file, weights_only=False)
    X_tr_orig, y_tr_orig, sub_tr_orig = payload["train"]
    val_data = payload["val"]
    test_data = payload["test"]
    num_classes = payload["num_classes"]
    X_tr_np = X_tr_orig.numpy() if isinstance(X_tr_orig, torch.Tensor) else X_tr_orig
    y_tr_np = y_tr_orig.numpy() if isinstance(y_tr_orig, torch.Tensor) else y_tr_orig
    in_channels = X_tr_np.shape[1]
    time_samples = X_tr_np.shape[2]
    out_minimization_dir = RESULTS_DL_DIR / "trial_minimization" / target_name
    os.makedirs(out_minimization_dir, exist_ok=True)
    results_list = []
    print(f"\n=======================================================")
    print(f"  Phase 5 Trial Minimization: {target_name} on {device}")
    print(f"=======================================================")
    for frac in TRIAL_FRACTIONS:
        X_sub, y_sub, sub_sub = subsample_training_data(X_tr_np, y_tr_np, sub_tr_orig, fraction=frac)
        n_trials = len(y_sub)
        print(f"\n---> Training 1D CNN with {frac*100:.0f}% trials ({n_trials} trials)...")
        train_data_sub = (X_sub, y_sub, sub_sub)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data_sub, val_data, test_data,
            batch_size=batch_size,
            num_workers=TRAIN_CONFIG["num_workers"]
        )
        model = ERPConv1DNet(in_channels=in_channels, time_samples=time_samples, num_classes=num_classes).to(device)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        ckpt_path = out_minimization_dir / f"cnn1d_frac_{int(frac*100)}.pt"
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            save_path=ckpt_path,
            log_dir=None,
            patience=10
        )
        trainer.fit(train_loader, val_loader, epochs=epochs)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        metrics, _, _, _ = evaluate_model(model, test_loader, device, output_dir=out_minimization_dir)
        results_list.append({
            "trial_fraction": frac,
            "n_train_trials": n_trials,
            "test_acc": metrics["Test_Accuracy"],
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
            "f1_score": metrics["F1_Score"],
            "auc": metrics["AUC"]
        })
    curves_dir = RESULTS_DL_DIR / "trial_minimization" / "curves"
    summary_dict, df_res = analyze_minimization_results(results_list, output_dir=curves_dir, target_name=target_name)
    return summary_dict
def main():
    parser = argparse.ArgumentParser(description="Phase 5: Trial Minimization Framework using 1D CNN")
    parser.add_argument("--dataset", type=str, default="ERP_CORE", choices=["ERP_CORE", "DS006018", "DS002680"])
    parser.add_argument("--task", type=str, default="P3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--all_datasets", action="store_true")
    args = parser.parse_args()
    summaries = []
    if args.all_datasets:
        for ds_name, tasks in DATASET_TASKS.items():
            for task in tasks:
                res = run_minimization_experiment(ds_name, task, epochs=args.epochs, batch_size=args.batch_size)
                if res:
                    summaries.append(res)
    else:
        res = run_minimization_experiment(args.dataset, args.task, epochs=args.epochs, batch_size=args.batch_size)
        if res:
            summaries.append(res)
    if summaries:
        df_summary = pd.DataFrame(summaries)
        summary_csv = RESULTS_DL_DIR / "trial_minimization" / "summary_minimization.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n[Phase 5 Complete] Final Trial Minimization Summary saved to: {summary_csv}")
if __name__ == "__main__":
    main()