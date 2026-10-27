import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from configs.dl_config import TRAIN_CONFIG, RESULTS_DL_DIR, DATASET_TASKS
from dl_prep.eeg_dataset import create_dataloaders
from dl_models.cnn1d import ERPConv1DNet
from dl_models.trainer import Trainer, LabelSmoothingCrossEntropy
from dl_models.evaluator import evaluate_model
def train_and_eval_task(dataset_name, task_name, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_name = f"{dataset_name}_{task_name}"
    print(f"\n=======================================================")
    print(f"  Phase 4 1D CNN Training: {target_name} on {device}")
    print(f"=======================================================")
    dataset_file = RESULTS_DL_DIR / "datasets" / f"{target_name}_prepared.pt"
    if not dataset_file.exists():
        print(f"[Warning] Prepared dataset file not found: {dataset_file}. Skipping.")
        return
    payload = torch.load(dataset_file, weights_only=False)
    train_data = payload["train"]
    val_data = payload["val"]
    test_data = payload["test"]
    num_classes = payload["num_classes"]
    in_channels = train_data[0].shape[1]
    time_samples = train_data[0].shape[2]
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=batch_size,
        num_workers=TRAIN_CONFIG["num_workers"]
    )
    # Instantiate High-Generalization 1D CNN
    model = ERPConv1DNet(in_channels=in_channels, time_samples=time_samples, num_classes=num_classes).to(device)
    
    # Label Smoothing + L2 Regularization (Weight Decay 1e-3)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ckpt_dir = RESULTS_DL_DIR / "checkpoints"
    figures_dir = RESULTS_DL_DIR / "figures" / target_name
    log_dir = RESULTS_DL_DIR / "logs" / target_name
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    save_path = ckpt_dir / f"cnn1d_{target_name}_best.pt"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        save_path=save_path,
        log_dir=log_dir,
        patience=15
    )
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    # Plot Learning Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curves ({target_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"Accuracy Curves ({target_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "learning_curves.png", dpi=300)
    plt.close()
    # Final Evaluation
    print(f"\n--- Holdout Test Evaluation ({target_name}) ---")
    model.load_state_dict(torch.load(save_path, weights_only=False))
    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, output_dir=figures_dir)
    print(f"[Phase 4 Complete] Results saved for {target_name} to: {figures_dir}")
def main():
    parser = argparse.ArgumentParser(description="Phase 4: High-Generalization 1D CNN Baseline Training")
    parser.add_argument("--dataset", type=str, default="ERP_CORE", choices=["ERP_CORE", "DS006018", "DS002680"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--all_datasets", action="store_true")
    args = parser.parse_args()
    if args.all_datasets:
        for ds_name, tasks in DATASET_TASKS.items():
            for task in tasks:
                train_and_eval_task(ds_name, task_name=task, epochs=args.epochs, batch_size=args.batch_size)
    elif args.task:
        train_and_eval_task(args.dataset, task_name=args.task, epochs=args.epochs, batch_size=args.batch_size)
    else:
        for task in DATASET_TASKS[args.dataset]:
            train_and_eval_task(args.dataset, task_name=task, epochs=args.epochs, batch_size=args.batch_size)
if __name__ == "__main__":
    main()