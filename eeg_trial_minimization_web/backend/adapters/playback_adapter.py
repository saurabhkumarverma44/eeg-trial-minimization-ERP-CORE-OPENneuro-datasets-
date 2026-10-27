import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from backend.adapters.base_adapter import BaseEEGAdapter
from backend.core.config import settings

CH_NAMES_31 = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5",
    "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6",
    "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8"
]

class SavedPlaybackAdapter(BaseEEGAdapter):
    """
    Streams actual recorded EEG trials from saved local files (.pt / .fif).
    Provides realistic playback timing while preserving genuine research signal values.
    """

    def __init__(self, dataset_id: str = "DS002680_gonogo_4", sampling_rate: float = 1000.0):
        self.dataset_id = dataset_id
        self.sampling_rate = sampling_rate
        self.channel_names = CH_NAMES_31
        self.is_connected = False
        self.data_buffer: Optional[np.ndarray] = None # (C, Total_Samples)
        self.labels: Optional[np.ndarray] = None
        self.current_sample_idx = 0
        self.total_samples = 0
        self.events_map: Dict[int, Dict[str, Any]] = {}

    def connect(self) -> bool:
        root = settings.EEG_PROJECT_ROOT
        prep_file = root / "results_dl_1" / "datasets" / f"{self.dataset_id}_prepared.pt"
        if not prep_file.exists():
            candidates = list((root / "results_dl_1" / "datasets").glob(f"*{self.dataset_id}*_prepared.pt"))
            if candidates:
                prep_file = candidates[0]
            else:
                prep_file = root / "results_dl_1" / "datasets" / "DS002680_gonogo_4_prepared.pt"

        if not prep_file.exists():
            return False

        # Load test split
        payload = torch.load(prep_file, map_location="cpu", weights_only=False)
        split_key = "test" if "test" in payload else "train"
        X, y = payload[split_key][0], payload[split_key][1]

        X_np = X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        # Concatenate first 20 epochs into a continuous timeline
        epochs_to_concat = min(len(X_np), 20)
        chunks = []
        self.events_map = {}
        sample_offset = 0

        for ep_idx in range(epochs_to_concat):
            ep_arr = X_np[ep_idx] # (C, T)
            chunks.append(ep_arr)
            # Place event marker at stimulus onset (e.g. 200ms into the epoch)
            stim_idx = sample_offset + 200
            lbl = int(y_np[ep_idx])
            self.events_map[stim_idx] = {
                "code": lbl + 1,
                "label": f"Go Trial (#{ep_idx})" if lbl == 0 else f"No-Go Trial (#{ep_idx})"
            }
            sample_offset += ep_arr.shape[1]

        self.data_buffer = np.concatenate(chunks, axis=1) # (C, Total_T)
        self.total_samples = self.data_buffer.shape[1]
        self.current_sample_idx = 0
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self.data_buffer = None

    def get_channels(self) -> List[str]:
        return self.channel_names

    def get_sampling_rate(self) -> float:
        return self.sampling_rate

    def get_next_chunk(self, chunk_size: int = 50) -> Optional[Dict[str, Any]]:
        if not self.is_connected or self.data_buffer is None:
            return None

        # Loop around buffer if reaching the end
        start = self.current_sample_idx
        end = start + chunk_size

        if end > self.total_samples:
            # Wrap around seamlessly
            part1 = self.data_buffer[:, start:]
            remaining = chunk_size - part1.shape[1]
            part2 = self.data_buffer[:, :remaining]
            chunk = np.concatenate([part1, part2], axis=1)
            self.current_sample_idx = remaining
        else:
            chunk = self.data_buffer[:, start:end]
            self.current_sample_idx = end

        # Check for event markers occurring in this chunk window
        events = []
        for offset in range(chunk_size):
            s_idx = (start + offset) % self.total_samples
            if s_idx in self.events_map:
                ev_info = self.events_map[s_idx]
                events.append({
                    "sample_offset": offset,
                    "code": ev_info["code"],
                    "label": ev_info["label"],
                    "timestamp": round(time.time(), 4)
                })

        dt = 1.0 / self.sampling_rate
        t_sec = np.arange(chunk_size) * dt + time.time()

        return {
            "samples": [[round(float(v), 2) for v in row] for row in chunk],
            "timestamps": [round(float(t), 4) for t in t_sec],
            "events": events,
            "is_simulated": False,
            "source_type": f"Saved Dataset Playback: {self.dataset_id} (Authentic Research Data)",
            "unit": "uV"
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "id": "playback",
            "name": f"Saved Data Playback ({self.dataset_id})",
            "connected": self.is_connected,
            "sampling_rate": self.sampling_rate,
            "channel_count": len(self.channel_names),
            "samples_transmitted": self.current_sample_idx,
            "total_available_samples": self.total_samples,
            "is_simulated": False,
            "source_dataset": self.dataset_id
        }
