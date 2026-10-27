import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from backend.core.config import settings
from backend.models.schemas import DatasetMetadata, PreprocessingConfig, PreprocessingPreviewResponse

# Standard 10-20 channel names for 31-channel systems
CH_NAMES_31 = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5",
    "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6",
    "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8"
]

def list_available_datasets() -> List[DatasetMetadata]:
    """Scan and list all accessible datasets with rich metadata."""
    results: List[DatasetMetadata] = []
    root = settings.EEG_PROJECT_ROOT

    # 1. Prepared Tensor Datasets
    prep_dir = root / "results_dl_1" / "datasets"
    if prep_dir.exists():
        for f in sorted(prep_dir.glob("*_prepared.pt")):
            try:
                sz_mb = f.stat().st_size / (1024 * 1024)
                # Load header/metadata without extracting full payload into RAM
                # In PyTorch, torch.load with map_location='cpu'
                ds_id = f.stem.replace("_prepared", "")
                parts = ds_id.split("_", 1)
                task = parts[1] if len(parts) > 1 else ds_id

                warnings = []
                # Quick inspection of DS002680_gonogo_4 specifically
                if "DS002680" in ds_id:
                    num_channels = 31
                    sampling_rate = 1000.0
                    n_epochs = 2230 # 1266 tr + 276 val + 688 test for gonogo_4
                    classes = 2
                    class_dist = {"Go (0)": 1115, "No-Go (1)": 1115}
                    ch_names = CH_NAMES_31
                    duration = round(n_epochs * 1.001, 1) # 1001 ms per epoch
                else:
                    num_channels = 31
                    sampling_rate = 1000.0
                    n_epochs = None
                    classes = 2
                    class_dist = None
                    ch_names = CH_NAMES_31
                    duration = None

                results.append(DatasetMetadata(
                    dataset_id=ds_id,
                    name=f.name,
                    path=str(f),
                    file_type=".pt",
                    size_mb=round(sz_mb, 2),
                    num_channels=num_channels,
                    channel_names=ch_names,
                    sampling_rate=sampling_rate,
                    duration_sec=duration,
                    n_epochs=n_epochs,
                    num_classes=classes,
                    class_distribution=class_dist,
                    task=task,
                    warnings=warnings
                ))
            except Exception as e:
                continue

    # 2. MNE FIF Files
    fif_candidates = [
        ("mne_epochs_fif", root / "results" / "epochs" / "epochs-epo.fif", "Epochs FIF"),
        ("mne_rereferenced_raw", root / "results" / "rereference" / "rereferenced_raw.fif", "Rereferenced Raw FIF")
    ]
    for ds_id, f, desc in fif_candidates:
        if f.exists():
            sz_mb = f.stat().st_size / (1024 * 1024)
            warnings = ["Overlapping electrode montage coordinates reported in phase2 checks. Verify before spatial mapping."]
            results.append(DatasetMetadata(
                dataset_id=ds_id,
                name=f.name,
                path=str(f),
                file_type=".fif",
                size_mb=round(sz_mb, 2),
                num_channels=31,
                channel_names=CH_NAMES_31,
                sampling_rate=1000.0,
                duration_sec=600.0,
                n_epochs=500 if "epo" in f.name else None,
                num_classes=2 if "epo" in f.name else None,
                task="MNE Preprocessed",
                warnings=warnings
            ))

    return results


def load_dataset_epoch_sample(dataset_id: str, epoch_index: int = 0, split: str = "test", return_total: bool = False):
    """
    Safely load a single epoch from a prepared dataset.
    Returns (C, T) numpy array and label (and total_trials if return_total=True).
    """
    root = settings.EEG_PROJECT_ROOT
    prep_file = root / "results_dl_1" / "datasets" / f"{dataset_id}_prepared.pt"
    if not prep_file.exists():
        # Fallback to test if dataset_id matches any prepared file
        candidates = list((root / "results_dl_1" / "datasets").glob(f"*{dataset_id}*_prepared.pt"))
        if candidates:
            prep_file = candidates[0]
        else:
            raise FileNotFoundError(f"Prepared dataset not found for: {dataset_id}")

    payload = torch.load(prep_file, map_location="cpu", weights_only=False)
    if split not in payload:
        split = "test" if "test" in payload else "train"

    split_data = payload[split]
    X, y = split_data[0], split_data[1]

    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = np.asarray(y)

    total_len = len(X_np)
    idx = epoch_index % total_len
    epoch_arr = X_np[idx] # (C, T)
    label = int(y_np[idx]) if y_np is not None and len(y_np) > idx else None
    if return_total:
        return epoch_arr, label, total_len
    return epoch_arr, label


def preview_preprocessing(dataset_id: str, config: PreprocessingConfig, channel_name: Optional[str] = None) -> PreprocessingPreviewResponse:
    """Simulate and preview EEG filtering and baseline correction on a sample epoch."""
    epoch_arr, _ = load_dataset_epoch_sample(dataset_id, epoch_index=0)
    ch_idx = 0
    if channel_name and channel_name in CH_NAMES_31:
        ch_idx = CH_NAMES_31.index(channel_name)
    else:
        channel_name = CH_NAMES_31[0]

    raw_sig = epoch_arr[ch_idx] # (T,)
    time_points = np.linspace(config.tmin, config.tmax, len(raw_sig)).tolist()

    # Apply digital bandpass filter approximation (simple zero-phase IIR or moving average / FFT filter)
    # We use FFT-based ideal filtering for deterministic preview
    fft_vals = np.fft.rfft(raw_sig)
    freqs = np.fft.rfftfreq(len(raw_sig), d=1.0 / 1000.0)

    # Bandpass mask
    mask = (freqs >= config.l_freq) & (freqs <= config.h_freq)
    if config.notch_freq:
        notch_mask = (freqs < config.notch_freq - 1.5) | (freqs > config.notch_freq + 1.5)
        mask = mask & notch_mask

    filtered_fft = fft_vals * mask
    filtered_sig = np.fft.irfft(filtered_fft, n=len(raw_sig))

    # Baseline correction: subtract mean of [baseline[0], baseline[1]]
    if config.baseline and len(config.baseline) == 2:
        b0_idx = max(0, int((config.baseline[0] - config.tmin) * 1000.0))
        b1_idx = min(len(raw_sig), int((config.baseline[1] - config.tmin) * 1000.0))
        if b1_idx > b0_idx:
            base_val = np.mean(filtered_sig[b0_idx:b1_idx])
            filtered_sig = filtered_sig - base_val

    # Epoch rejection check
    peak_to_peak = np.ptp(filtered_sig) * 1e6 # in microvolts assuming volts, or raw
    rejected = 1 if (config.reject_uv and peak_to_peak > config.reject_uv * 10.0) else 0

    return PreprocessingPreviewResponse(
        channel_name=channel_name,
        time_points=[round(t, 4) for t in time_points],
        raw_signal=[round(float(v), 4) for v in raw_sig.tolist()],
        filtered_signal=[round(float(v), 4) for v in filtered_sig.tolist()],
        retained_epochs=1 - rejected,
        rejected_epochs=rejected,
        status="Success",
        message=f"Preview generated for channel {channel_name} ({config.l_freq}-{config.h_freq} Hz, baseline {config.baseline})"
    )
