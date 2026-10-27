from pathlib import Path
from typing import List, Dict, Any
from backend.core.config import settings
from backend.models.schemas import InventoryItem, InventoryResponse

def scan_project_inventory() -> InventoryResponse:
    root = settings.EEG_PROJECT_ROOT
    items: List[InventoryItem] = []
    total_size = 0.0

    if not root.exists():
        return InventoryResponse(
            project_root=str(root),
            items=[],
            total_files=0,
            total_size_mb=0.0
        )

    # Key checkpoints
    checkpoints_dir = root / "results_dl_1" / "checkpoints_eegnet"
    if checkpoints_dir.exists():
        for f in checkpoints_dir.glob("*.pt"):
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            items.append(InventoryItem(
                name=f.name,
                category="Checkpoint",
                path=str(f),
                size_mb=round(sz, 3),
                file_type=f.suffix,
                status="Verified",
                description="Trained EEGNet checkpoint for paradigm"
            ))

    # Prepared datasets
    datasets_dir = root / "results_dl_1" / "datasets"
    if datasets_dir.exists():
        for f in datasets_dir.glob("*_prepared.pt"):
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            items.append(InventoryItem(
                name=f.name,
                category="Prepared Dataset",
                path=str(f),
                size_mb=round(sz, 2),
                file_type=f.suffix,
                status="Ready",
                description="Pre-extracted (train, val, test) PyTorch tensor dataset"
            ))

    # MNE raw and epochs
    fif_candidates = [
        root / "results" / "epochs" / "epochs-epo.fif",
        root / "results" / "rereference" / "rereferenced_raw.fif",
    ]
    for f in fif_candidates:
        if f.exists():
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            items.append(InventoryItem(
                name=f.name,
                category="MNE FIF Recording",
                path=str(f),
                size_mb=round(sz, 2),
                file_type=f.suffix,
                status="Verified",
                description="Preprocessed raw/epochs recording file"
            ))

    # Summary and comparison CSVs
    csv_candidates = [
        (root / "results_dl_1" / "cnn_vs_eegnet_comparison.csv", "Model Comparison Summary"),
        (root / "results_dl_1" / "phase4_eegnet_final_summary.csv", "Phase 4 Final Accuracy Summary"),
        (root / "results_dl_1" / "trial_minimization" / "summary_minimization.csv", "Trial Minimization Stability Summary"),
    ]
    for f, desc in csv_candidates:
        if f.exists():
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            items.append(InventoryItem(
                name=f.name,
                category="Research CSV Table",
                path=str(f),
                size_mb=round(sz, 4),
                file_type=f.suffix,
                status="Verified",
                description=desc
            ))

    # Visualizations
    fig_candidates = [
        root / "results_dl_1" / "jetson_visualizations" / "eeg_all_channels.png",
        root / "results_dl_1" / "jetson_visualizations" / "eeg_selected_channels.png",
        root / "results_dl_1" / "figures_eegnet" / "DS002680_gonogo_4" / "learning_curves_eegnet.png",
        root / "results" / "rereference" / "montage_plot.png",
    ]
    for f in fig_candidates:
        if f.exists():
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            items.append(InventoryItem(
                name=f.name,
                category="Scientific Figure",
                path=str(f),
                size_mb=round(sz, 3),
                file_type=f.suffix,
                status="Available",
                description="Research plot artifact"
            ))

    return InventoryResponse(
        project_root=str(root),
        items=items,
        total_files=len(items),
        total_size_mb=round(total_size, 2)
    )
