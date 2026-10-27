import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import os

class ERPConv1DNet(nn.Module):
    """
    1D CNN baseline for ERP/EEG trials across time.
    Matches dl_models/cnn1d_BACKUP_before_groupnorm.py.

    Input shape: (N, C, T)
    """

    def __init__(
        self,
        in_channels: int = 31,
        num_classes: int = 2,
        base_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Conv1d expects (N, C, T)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(base_channels * 4, base_channels * 2)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(base_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # (N, 1, C, T) -> (N, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc_dropout(x)
        return self.fc_out(x)


def load_cnn1d_checkpoint(
    checkpoint_path: str,
    in_channels: int = 31,
    num_classes: int = 2,
    device: str = "cpu"
) -> Tuple[ERPConv1DNet, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    elif isinstance(raw_state, dict):
        state_dict = raw_state
    else:
        raise ValueError("Invalid checkpoint file.")

    cleaned_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model = ERPConv1DNet(in_channels=in_channels, num_classes=num_classes)
    load_result = model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()

    diagnostics = {
        "checkpoint_path": str(checkpoint_path),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "is_fully_compatible": len(load_result.missing_keys) == 0,
    }

    return model, diagnostics
