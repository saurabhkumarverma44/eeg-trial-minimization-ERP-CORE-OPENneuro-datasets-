# dl_models/eegnet.py

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018), adapted for variable channel count and
    epoch length across paradigms.

    Standard EEGNet computes a fixed FC input size assuming constant
    channels/timepoints. Since this project's 22 paradigms vary in both
    (26-32 channels, 501-1025 samples), we use AdaptiveAvgPool2d before
    the classifier instead -- avoids a class of shape-mismatch bugs that
    would otherwise require per-paradigm special-casing.

    Input: (N, 1, C, T) -- single "image" channel, C EEG channels, T samples.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.GroupNorm(4, F1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1),
                      groups=F1, bias=False),
            nn.GroupNorm(4, F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.GroupNorm(4, F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        POOL_POSITIONS = 8
        self.global_pool = nn.AdaptiveAvgPool2d((1, POOL_POSITIONS))
        self.classifier = nn.Linear(F2 * POOL_POSITIONS, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)
