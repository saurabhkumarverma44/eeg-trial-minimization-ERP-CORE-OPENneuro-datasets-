# stage3_feature_extraction/connectivity_core.py
#
# Shared math for 3.4 (PLV) and 3.5 (wPLI). Both are computed *per single
# epoch*, across the time samples within that epoch (after band-pass
# filtering + Hilbert transform), rather than across trials at a fixed
# latency. This matches the Phase 3 spec's required output shape of
# (n_epochs, n_ch, n_ch) — one connectivity matrix per epoch — rather than
# the more conventional single matrix-per-condition PLV.
#
# PLV(i,j)  = | mean_t( exp(1j * (phase_i(t) - phase_j(t))) ) |
# wPLI(i,j) = | mean_t( imag(Z) ) | / mean_t( |imag(Z)| )
#             where Z(t) = analytic_i(t) * conj(analytic_j(t))
# wPLI is insensitive to zero-phase-lag interactions, making it more robust
# to volume conduction than PLV (hence the 3.5 "robustness check" framing).

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _bandpass(data: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """data: (n_channels, n_times). Zero-phase Butterworth band-pass."""
    nyq = sfreq / 2.0
    low = max(fmin / nyq, 1e-6)
    high = min(fmax / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _analytic_phase(data: np.ndarray) -> np.ndarray:
    """data: (n_channels, n_times) -> instantaneous phase, same shape."""
    return np.angle(hilbert(data, axis=-1))


def _analytic_signal(data: np.ndarray) -> np.ndarray:
    return hilbert(data, axis=-1)


def plv_matrix(epoch_data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns PLV matrix (n_channels, n_channels), symmetric, diagonal = 1.
    """
    filtered = _bandpass(epoch_data, sfreq, fmin, fmax)
    phase = _analytic_phase(filtered)  # (n_ch, n_times)
    n_ch = phase.shape[0]

    # exp(i*phase) for all channels -> (n_ch, n_times)
    z = np.exp(1j * phase)
    # PLV(i,j) = |mean_t( z_i * conj(z_j) )|
    cross = z @ z.conj().T / phase.shape[1]  # (n_ch, n_ch)
    plv = np.abs(cross)
    np.fill_diagonal(plv, 1.0)
    return plv


def wpli_matrix(epoch_data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    epoch_data: (n_channels, n_times) for ONE epoch.
    Returns wPLI matrix (n_channels, n_channels), symmetric, diagonal = 0
    (self-connectivity is undefined/zero-imaginary by construction for wPLI).
    """
    filtered = _bandpass(epoch_data, sfreq, fmin, fmax)
    analytic = _analytic_signal(filtered)  # (n_ch, n_times), complex
    n_ch, n_times = analytic.shape

    wpli = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_ch):
        cross_i = analytic[i][None, :] * np.conj(analytic)  # (n_ch, n_times)
        imag = cross_i.imag
        numerator = np.abs(imag.mean(axis=-1))
        denominator = np.abs(imag).mean(axis=-1)
        denominator_safe = np.where(denominator < 1e-15, np.nan, denominator)
        wpli[i, :] = numerator / denominator_safe
    np.fill_diagonal(wpli, 0.0)
    wpli = np.nan_to_num(wpli, nan=0.0)
    return wpli


def upper_triangle_flat(mat: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle (excludes diagonal) of a square matrix."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]

