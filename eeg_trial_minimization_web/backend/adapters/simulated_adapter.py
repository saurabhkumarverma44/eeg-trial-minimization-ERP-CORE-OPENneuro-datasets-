import time
import numpy as np
from typing import List, Dict, Any, Optional
from backend.adapters.base_adapter import BaseEEGAdapter

CH_NAMES_31 = [
    "Fp1", "Fp2", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5",
    "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6",
    "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4"
]

CAP_CHANNELS_8 = ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"]

class SimulatedEEGAdapter(BaseEEGAdapter):
    """
    Physiologically-inspired synthetic EEG signal generator.
    Generates multi-channel rhythms (Alpha 10Hz, Beta 20Hz, Theta 6Hz, Pink noise)
    plus simulated ERP oddball/Go-NoGo stimulus events at periodic intervals.
    Clearly and visibly labeled as synthetic data at all times.
    """

    def __init__(self, sampling_rate: float = 1000.0, num_channels: int = 31):
        self.sampling_rate = sampling_rate
        if num_channels == 8:
            self.channel_names = CAP_CHANNELS_8
        else:
            self.channel_names = CH_NAMES_31[:num_channels]
        self.is_connected = False
        self.sample_counter = 0
        self.start_time = 0.0
        self.last_event_time = 0.0
        self.event_counter = 0
        self.current_erp_samples_remaining = 0
        self.current_erp_code = 0

    def connect(self) -> bool:
        self.is_connected = True
        self.start_time = time.time()
        self.last_event_time = self.start_time
        self.sample_counter = 0
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_channels(self) -> List[str]:
        return self.channel_names

    def get_sampling_rate(self) -> float:
        return self.sampling_rate

    def get_next_chunk(self, chunk_size: int = 50) -> Optional[Dict[str, Any]]:
        if not self.is_connected:
            return None

        dt = 1.0 / self.sampling_rate
        t_indices = np.arange(self.sample_counter, self.sample_counter + chunk_size)
        t_sec = t_indices * dt

        num_ch = len(self.channel_names)
        # Background 1/f noise + band-specific oscillations
        # Standard noise scale ~ 5-10 microvolts
        noise = np.random.normal(0, 4.0, size=(num_ch, chunk_size))

        # Alpha rhythm (10 Hz) prominent in occipital electrodes (O1, Oz, O2, P3, P4, Pz)
        alpha = 8.0 * np.sin(2 * np.pi * 10.0 * t_sec)
        # Beta rhythm (22 Hz) prominent in frontal/central electrodes
        beta = 3.0 * np.sin(2 * np.pi * 22.0 * t_sec)
        # Theta rhythm (6 Hz)
        theta = 5.0 * np.sin(2 * np.pi * 6.0 * t_sec)

        samples = np.zeros((num_ch, chunk_size), dtype=np.float32)
        for i, ch in enumerate(self.channel_names):
            s = noise[i]
            if ch in ["O1", "Oz", "O2", "P3", "P4", "Pz"]:
                s += alpha * 1.5 + theta * 0.5
            elif ch in ["Fp1", "Fp2", "Fz", "F3", "F4", "Cz"]:
                s += beta * 1.2 + theta * 0.8
            else:
                s += alpha * 0.7 + beta * 0.7
            samples[i] = s

        # Periodic Event Markers (Go = code 1, No-Go = code 2) every ~2.5 seconds
        events = []
        now = time.time()
        if now - self.last_event_time >= 2.5:
            self.last_event_time = now
            self.event_counter += 1
            code = 1 if (self.event_counter % 2 == 1) else 2
            offset = np.random.randint(5, chunk_size - 10)
            events.append({
                "sample_offset": int(offset),
                "code": code,
                "label": "Go Stimulus" if code == 1 else "No-Go Stimulus",
                "timestamp": round(now, 4)
            })
            self.current_erp_samples_remaining = 400 # 400ms ERP deflection
            self.current_erp_code = code

        # Inject P300/ERP waveform if stimulus occurred
        if self.current_erp_samples_remaining > 0:
            erp_t = np.linspace(0.1, 0.5, chunk_size)
            # P300 positive peak around 300ms
            p300_shape = 25.0 * np.exp(-((erp_t - 0.3) ** 2) / 0.005)
            if self.current_erp_code == 2:
                # No-Go N2/P3 complex
                p300_shape += -15.0 * np.exp(-((erp_t - 0.2) ** 2) / 0.003)
            # Add primarily to parietal/central channels
            for i, ch in enumerate(self.channel_names):
                if ch in ["Cz", "Pz", "CP1", "CP2"]:
                    samples[i] += p300_shape
            self.current_erp_samples_remaining -= chunk_size

        self.sample_counter += chunk_size

        return {
            "samples": [[round(float(v), 2) for v in row] for row in samples],
            "timestamps": [round(float(t), 4) for t in t_sec],
            "events": events,
            "is_simulated": True,
            "source_type": "Synthetic Simulated EEG Engine (100% Synthetic)",
            "unit": "uV"
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "id": "simulated",
            "name": "Simulated EEG Generator",
            "connected": self.is_connected,
            "sampling_rate": self.sampling_rate,
            "channel_count": len(self.channel_names),
            "samples_transmitted": self.sample_counter,
            "event_count": self.event_counter,
            "is_simulated": True,
            "warning": "Data is synthetically generated for UI verification and offline testing. Never treat as physiological recordings."
        }
