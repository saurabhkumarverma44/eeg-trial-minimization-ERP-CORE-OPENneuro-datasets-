import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class EEGGenerator(nn.Module):
    """
    Conditional 1D Conv generator for multi-channel EEG epochs.
    Takes latent noise (N, latent_dim) + one-hot/embedded class label -> produces (N, C, T).
    """
    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 2,
        out_channels: int = 31,
        time_steps: int = 1001,
        feature_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.time_steps = time_steps
        self.feature_dim = feature_dim

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        # Initial projection to (feature_dim * 4, 32)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, feature_dim * 4 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling via ConvTranspose1d layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1), # -> 64
            nn.BatchNorm1d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1), # -> 128
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1), # -> 256
            nn.BatchNorm1d(feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1), # -> 512
            nn.BatchNorm1d(feature_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(feature_dim // 4, out_channels, kernel_size=4, stride=2, padding=1), # -> 1024
        )
        self.adaptive_time = nn.AdaptiveAvgPool1d(time_steps)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        combined = torch.cat([z, emb], dim=1)
        x = self.fc(combined)
        x = x.view(x.size(0), self.feature_dim * 4, 32)
        x = self.deconv(x)
        x = self.adaptive_time(x) # (N, out_channels, time_steps)
        return x


class EEGCritic(nn.Module):
    """
    Conditional 1D Critic for WGAN-GP.
    Takes (N, C, T) + class label -> produces scalar Wasserstein score.
    Note: Standard BatchNorm is omitted in critic to avoid invalidating gradient penalty.
    """
    def __init__(
        self,
        in_channels: int = 31,
        num_classes: int = 2,
        time_steps: int = 1001,
        feature_dim: int = 64
    ):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, time_steps)

        # in_channels + 1 channel for conditional label map
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + 1, feature_dim, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(feature_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(feature_dim * 2, feature_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(feature_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(feature_dim * 4, feature_dim * 8, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(feature_dim * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feature_dim * 8, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Create 1D channel from label embedding
        lbl_map = self.label_emb(labels).unsqueeze(1) # (N, 1, T)
        combined = torch.cat([x, lbl_map], dim=1) # (N, C+1, T)
        feat = self.conv(combined)
        pooled = self.global_pool(feat).squeeze(-1)
        score = self.fc(pooled)
        return score.squeeze(-1)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Calculates the gradient penalty for WGAN-GP."""
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    fake = torch.ones(real_samples.size(0), device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def evaluate_synthetic_quality(real_trials: np.ndarray, fake_trials: np.ndarray) -> Dict[str, float]:
    """
    Computes objective quality metrics comparing synthetic EEG to real EEG:
    1. ERP Average Correlation: Pearson r of grand-average ERP across channels.
    2. Power Spectral Density (PSD) Cosine Similarity: across standard frequency bins.
    3. Variance Ratio: synthetic variance / real variance.
    """
    real_mean = np.mean(real_trials, axis=0) # (C, T)
    fake_mean = np.mean(fake_trials, axis=0) # (C, T)

    # Flattened ERP correlation
    r_flat = real_mean.flatten()
    f_flat = fake_mean.flatten()
    r_std = np.std(r_flat)
    f_std = np.std(f_flat)

    if r_std > 1e-8 and f_std > 1e-8:
        erp_corr = float(np.corrcoef(r_flat, f_flat)[0, 1])
    else:
        erp_corr = 0.0

    # Variance ratio
    real_var = float(np.var(real_trials))
    fake_var = float(np.var(fake_trials))
    var_ratio = float(fake_var / max(real_var, 1e-8))

    # FFT / Frequency distribution proxy
    real_fft = np.abs(np.fft.rfft(real_mean, axis=-1))
    fake_fft = np.abs(np.fft.rfft(fake_mean, axis=-1))
    norm_r = np.linalg.norm(real_fft)
    norm_f = np.linalg.norm(fake_fft)
    psd_sim = float(np.sum(real_fft * fake_fft) / max(norm_r * norm_f, 1e-8))

    return {
        "erp_correlation": max(-1.0, min(1.0, erp_corr)),
        "psd_cosine_similarity": max(0.0, min(1.0, psd_sim)),
        "variance_ratio": round(var_ratio, 4),
        "real_variance": round(real_var, 6),
        "synthetic_variance": round(fake_var, 6),
        "is_physiologically_verified": False,
        "label": "Synthetic Generated EEG (Labeled for research augmentation only)"
    }
