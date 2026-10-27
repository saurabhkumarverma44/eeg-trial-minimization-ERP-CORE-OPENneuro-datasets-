import argparse
import torch
import os
import numpy as np
from configs.dl_config import DATASET_PATHS, DATASET_TASKS, PREP_CONFIG, RESULTS_DL_DIR
from dl_prep.epoch_loader import load_dataset_epochs, discover_tasks
from dl_prep.data_splitter import subject_wise_split
from dl_prep.scaler import EEGChannelScaler

def process_dataset_task(dataset_name, task_name):
    dataset_path = DATASET_PATHS[dataset_name]
    print(f"\n=======================================================")
    print(f"  Phase 3 Data Prep: {dataset_name} | Task Paradigm: {task_name}")
    print(f"=======================================================")

    try:
        # 1. Load Clean Epochs & Labels
        X, y, subject_ids, ch_names, sfreq = load_dataset_epochs(dataset_path, task_name=task_name)

        # 2. Subject-wise Split (No Data Leakage)
        train_tuple, val_tuple, test_tuple = subject_wise_split(
            X, y, subject_ids,
            train_ratio=PREP_CONFIG["train_ratio"],
            val_ratio=PREP_CONFIG["val_ratio"],
            test_ratio=PREP_CONFIG["test_ratio"],
            seed=PREP_CONFIG["random_seed"]
        )

        X_train, y_train, sub_train = train_tuple
        X_val, y_val, sub_val = val_tuple
        X_test, y_test, sub_test = test_tuple

        # 3. Fit Scaler ONLY on Training Set
        scaler = EEGChannelScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # 4. Save Payload
        output_dir = RESULTS_DL_DIR / "datasets"
        os.makedirs(output_dir, exist_ok=True)
        save_file = output_dir / f"{dataset_name}_{task_name}_prepared.pt"

        payload = {
            "dataset_name": dataset_name,
            "task_name": task_name,
            "train": (torch.from_numpy(X_train_scaled), torch.from_numpy(y_train), list(sub_train)),
            "val": (torch.from_numpy(X_val_scaled), torch.from_numpy(y_val), list(sub_val)),
            "test": (torch.from_numpy(X_test_scaled), torch.from_numpy(y_test), list(sub_test)),
            "ch_names": ch_names,
            "sfreq": sfreq,
            "num_classes": len(set(y))
        }

        torch.save(payload, save_file, pickle_protocol=5)
        print(f"[Phase 3 Complete] Saved: {save_file}\n")
    except Exception as e:
        print(f"[Error] Failed to process {dataset_name} task {task_name}: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Deep Learning Data Preparation")
    parser.add_argument("--dataset", type=str, default="ERP_CORE", choices=["ERP_CORE", "DS006018", "DS002680"])
    parser.add_argument("--task", type=str, default=None, help="Specific task paradigm name")
    parser.add_argument("--all_datasets", action="store_true", help="Process ALL tasks for ALL 3 datasets")
    args = parser.parse_args()

    if args.all_datasets:
        for ds_name, tasks in DATASET_TASKS.items():
            print(f"\n>>> PROCESSING ALL TASKS FOR DATASET: {ds_name} <<<")
            for task in tasks:
                process_dataset_task(ds_name, task)
    elif args.task:
        process_dataset_task(args.dataset, args.task)
    else:
        # Auto-discover tasks for selected dataset
        tasks = discover_tasks(DATASET_PATHS[args.dataset])
        print(f"Discovered tasks for {args.dataset}: {tasks}")
        for task in tasks:
            process_dataset_task(args.dataset, task)

if __name__ == "__main__":
    main()
