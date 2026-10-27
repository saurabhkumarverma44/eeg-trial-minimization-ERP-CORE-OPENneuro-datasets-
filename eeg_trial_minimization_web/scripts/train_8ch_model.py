import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set path
sys.path.insert(0, str(Path(".").resolve()))
from backend.models.eegnet import EEGNet

def train_8ch_model():
    dataset_path = Path("../results_dl_1/datasets/DS002680_gonogo_4_prepared.pt")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    print("Loading prepared dataset...")
    d = torch.load(dataset_path, map_location="cpu", weights_only=False)
    
    # 8 Cap channels: FP1, FP2, C3, C4, P3, P4, O1, O2
    ch_indices = [0, 1, 4, 5, 6, 7, 8, 9]
    ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    
    X_tr = d["train"][0][:, ch_indices, :].float()
    y_tr = d["train"][1].long()
    X_val = d["val"][0][:, ch_indices, :].float()
    y_val = d["val"][1].long()
    X_te = d["test"][0][:, ch_indices, :].float()
    y_te = d["test"][1].long()

    print(f"8-Channel Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    model = EEGNet(in_channels=8, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for x_b, y_b in loader:
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_preds = val_out.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

            if val_acc > best_val_acc or best_state is None:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 30:
            print(f"Epoch {epoch:02d}: Loss={total_loss:.4f}, Val Acc={val_acc*100:.2f}% (Best={best_val_acc*100:.2f}%)")

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_te)
        test_preds = test_out.argmax(dim=-1)
        test_acc = (test_preds == y_te).float().mean().item()

    print(f"\nFinal Best 8-Channel Model Test Accuracy: {test_acc*100:.2f}%")

    out_dir = Path("storage/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eegnet_8ch_neurocap.pt"
    
    # Save checkpoint
    torch.save({
        "state_dict": best_state,
        "in_channels": 8,
        "num_classes": 2,
        "ch_names": ch_names,
        "sampling_rate": 250.0,
        "test_accuracy": test_acc,
        "dataset_name": "DS002680_gonogo_4_8ch",
        "description": "Authentic 8-Channel EEGNet trained specifically on 8-Channel Neuro Cap channels (Fp1, Fp2, C3, C4, P3, P4, O1, O2)"
    }, out_path)
    print(f"Saved checkpoint to: {out_path} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    train_8ch_model()
