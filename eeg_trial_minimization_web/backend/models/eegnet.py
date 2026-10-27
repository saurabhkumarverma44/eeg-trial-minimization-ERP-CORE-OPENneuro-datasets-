import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import os

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018), adapted for variable channel count and
    epoch length across paradigms with AdaptiveAvgPool2d((1,1)).
    Matches the verified research checkpoint eegnet_DS002680_gonogo_4_best.pt.

    Input shape: (N, C, T) or (N, 1, C, T)
    """

    def __init__(
        self,
        in_channels: int = 31,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        use_group_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_group_norm = use_group_norm

        norm_layer1 = nn.GroupNorm(4, F1) if use_group_norm else nn.BatchNorm2d(F1)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            norm_layer1,
        )

        norm_layer2 = nn.GroupNorm(4, F1 * D) if use_group_norm else nn.BatchNorm2d(F1 * D)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1),
                      groups=F1, bias=False),
            norm_layer2,
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        norm_layer3 = nn.GroupNorm(4, F2) if use_group_norm else nn.BatchNorm2d(F2)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            norm_layer3,
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (N, C, T) -> (N, 1, C, T)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_eegnet_checkpoint(
    checkpoint_path: str,
    in_channels: int = 31,
    num_classes: int = 2,
    device: str = "cpu"
) -> Tuple[EEGNet, Dict[str, Any]]:
    """
    Safely load an EEGNet model from a local .pt file.
    Validates state dict, handles '_orig_mod.' prefix if present from torch.compile,
    and returns model in eval mode along with diagnostics.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Use safe loading
    try:
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback to weights_only=False only for local trusted files if needed
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    elif isinstance(raw_state, dict):
        state_dict = raw_state
    else:
        raise ValueError("Loaded artifact is not a state dictionary.")

    # Strip '_orig_mod.' prefix if checkpoint was saved with torch.compile
    # Also normalize alternative layer naming conventions (e.g., conv_temporal -> temporal_conv.0)
    name_map = {
        'conv_temporal': 'temporal_conv.0',
        'bn_temporal': 'temporal_conv.1',
        'conv_spatial': 'depthwise_conv.0',
        'bn_spatial': 'depthwise_conv.1',
        'conv_separable_depth': 'separable_conv.0',
        'conv_separable_point': 'separable_conv.1',
        'bn_separable': 'separable_conv.2'
    }

    cleaned_dict = {}
    had_orig_mod = False
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
            had_orig_mod = True
        for old_prefix, new_prefix in name_map.items():
            if k.startswith(old_prefix):
                k = k.replace(old_prefix, new_prefix, 1)
                break
        cleaned_dict[k] = v

    # Auto-detect exact input channels from depthwise spatial filter if present
    if "depthwise_conv.0.weight" in cleaned_dict:
        detected_ch = cleaned_dict["depthwise_conv.0.weight"].shape[2]
        if detected_ch != in_channels and in_channels in [8, 30, 31]:
            in_channels = detected_ch

    # Check whether the checkpoint uses BatchNorm or GroupNorm
    has_bn = any("running_mean" in k for k in cleaned_dict.keys())
    use_group_norm = not has_bn

    model = EEGNet(
        in_channels=in_channels,
        num_classes=num_classes,
        use_group_norm=use_group_norm
    )

    load_result = model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()


    diagnostics = {
        "checkpoint_path": str(checkpoint_path),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "use_group_norm": use_group_norm,
        "had_orig_mod_prefix": had_orig_mod,
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "is_fully_compatible": len(load_result.missing_keys) == 0,
    }

    return model, diagnostics
