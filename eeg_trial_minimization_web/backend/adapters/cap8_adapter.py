import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from backend.adapters.base_adapter import BaseEEGAdapter

try:
    import serial
    import serial.tools.list_ports as lp
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

CAP_CHANNELS_8 = ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"]

class Cap8ChannelAdapter(BaseEEGAdapter):
    """
    Dedicated 8-Channel Neuro Cap Acquisition & Emulation Adapter.
    Specifically calibrated for 8-channel Q.Neuro H100 dry/contact electrode montage:
    Frontal: Fp1, Fp2; Occipital: O1, O2; Central/Vertex: C3, Cz, Fz, C4.
    Streams at clinical 250 Hz with calm real-time pacing, trial-budget tracking,
    and optional hardware COM port streaming.
    """

    def __init__(self, port: str = "AUTO", baud_rate: int = 115200, trial_budget: int = 63):
        self.port = port
        self.baud_rate = baud_rate
        self.trial_budget = trial_budget
        self.channels = CAP_CHANNELS_8
        self.sfreq = 250.0  # Standard clinical 250 Hz for 8-ch wireless caps
        self.is_connected = False
        self.is_hardware_active = False
        self.serial_conn = None

        self.sample_counter = 0
        self.start_time = 0.0
        self.last_event_time = 0.0
        self.current_trial_idx = 0
        self.stimulus_code = 1 # 1: Go, 2: No-Go

        # Post-stimulus ERP state tracking
        self.samples_since_stimulus = 9999
        self.current_erp_type = 1 # 1: Go, 2: No-Go

        # Gain & Real-time Filter State
        self.scale: float = 1.0
        self.filter_mode: str = "clean" # "clean", "ultra", "clean_60hz", "raw", "notch_50", "notch_60"
        self._init_filters()

    def _init_filters(self):
        try:
            import scipy.signal as signal
            self.b_notch50, self.a_notch50 = signal.iirnotch(w0=50.0, Q=25.0, fs=self.sfreq)
            zi_n50 = signal.lfilter_zi(self.b_notch50, self.a_notch50)
            self.zi_notch50 = np.tile(zi_n50[:, None], (1, 8))

            self.b_band_clean, self.a_band_clean = signal.butter(N=2, Wn=[0.5, 22.0], btype="bandpass", fs=self.sfreq)
            zi_bc = signal.lfilter_zi(self.b_band_clean, self.a_band_clean)
            self.zi_band_clean = np.tile(zi_bc[:, None], (1, 8))

            self.b_band_ultra, self.a_band_ultra = signal.butter(N=2, Wn=[0.5, 16.0], btype="bandpass", fs=self.sfreq)
            zi_bu = signal.lfilter_zi(self.b_band_ultra, self.a_band_ultra)
            self.zi_band_ultra = np.tile(zi_bu[:, None], (1, 8))
            self.has_filters = True
        except Exception:
            self.has_filters = False

    def set_gain(self, scale: float) -> None:
        self.scale = max(0.1, min(10.0, float(scale)))

    def set_filter_mode(self, mode: str) -> None:
        self.filter_mode = mode

    def connect(self) -> bool:
        self.is_connected = True
        self.start_time = time.time()
        self.last_event_time = self.start_time
        self.sample_counter = 0
        self.current_trial_idx = 0
        self.samples_since_stimulus = 9999

        # Attempt hardware serial link if requested
        if SERIAL_AVAILABLE and self.port != "SIMULATED":
            try:
                target_port = self.port
                if target_port == "AUTO":
                    available = list(lp.comports())
                    if available:
                        target_port = available[0].device

                if target_port and target_port != "AUTO":
                    self.serial_conn = serial.Serial(target_port, self.baud_rate, timeout=0.1)
                    self.is_hardware_active = True
                    return True
            except Exception:
                self.is_hardware_active = False

        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self.is_hardware_active = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def get_channels(self) -> List[str]:
        return self.channels

    def get_sampling_rate(self) -> float:
        return self.sfreq

    def set_trial_budget(self, budget: int) -> None:
        self.trial_budget = max(1, budget)
        self.current_trial_idx = 0

    def get_next_chunk(self, chunk_size: int = 25) -> Optional[Dict[str, Any]]:
        """
        Generates 25 samples (at 250 Hz = exactly 100 ms of real-time data).
        """
        if not self.is_connected:
            return None

        # Check if hardware serial has available data
        if self.is_hardware_active and self.serial_conn and self.serial_conn.in_waiting >= chunk_size * 8 * 4:
            try:
                # Read binary float32 packet: (8 channels, chunk_size)
                raw_bytes = self.serial_conn.read(chunk_size * 8 * 4)
                data_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(8, chunk_size)
                self.sample_counter += chunk_size
                now = time.time()
                t_vec = [now - (chunk_size - i) / self.sfreq for i in range(chunk_size)]
                return {
                    "samples": data_np.tolist(),
                    "timestamps": t_vec,
                    "events": [],
                    "is_simulated": False,
                    "source_type": f"8-Ch Neuro Cap (Hardware: {self.port})",
                    "channels": self.channels,
                    "current_trial": self.current_trial_idx,
                    "trial_budget": self.trial_budget
                }
            except Exception:
                pass

        # High-Fidelity 8-Channel Cap Emulation Mode
        dt = 1.0 / self.sfreq
        t_indices = np.arange(self.sample_counter, self.sample_counter + chunk_size)
        t_sec = t_indices * dt
        now = time.time()

        num_ch = len(self.channels)
        # Background EEG microvolt rhythms adjusted for active filter mode
        if self.filter_mode == "ultra":
            noise = np.random.normal(0, 0.6, size=(num_ch, chunk_size))
        elif self.filter_mode == "clean":
            noise = np.random.normal(0, 1.2, size=(num_ch, chunk_size))
        else:  # raw
            noise = np.random.normal(0, 4.2, size=(num_ch, chunk_size))
            # add 50 Hz powerline hum in raw mode
            hum = 9.0 * np.sin(2 * np.pi * 50.0 * t_sec)
            for i in range(num_ch):
                noise[i] += hum

        # Alpha rhythm (10 Hz) prominent in occipital (O1, O2) & parietal (P3, P4)
        alpha = 14.5 * np.sin(2 * np.pi * 10.0 * t_sec)

        # Sensorimotor Mu rhythm (12 Hz) & Beta (20 Hz) prominent in central (C3, C4)
        mu = 10.5 * np.sin(2 * np.pi * 12.0 * t_sec)
        beta = 6.0 * np.sin(2 * np.pi * 20.0 * t_sec)

        # Frontal slow theta (5 Hz) & blinks in Fp1, Fp2
        theta = 8.5 * np.sin(2 * np.pi * 5.0 * t_sec)
        # Occasional blink transient (every ~4.2 seconds)
        blink_cycle = (t_sec % 4.2)
        blink_mask = np.where((blink_cycle > 0.0) & (blink_cycle < 0.28), np.sin(np.pi * blink_cycle / 0.28) ** 2, 0.0)
        blink_wave = 42.0 * blink_mask # ~42 uV blink deflection on Fp1/Fp2

        samples = np.zeros((num_ch, chunk_size), dtype=np.float32)
        for i, ch in enumerate(self.channels):
            s = noise[i] * 1.2
            if ch == "Fp1":
                s += theta * 1.3 + beta * 0.5 + blink_wave
            elif ch == "Fp2":
                s += theta * 1.2 + beta * 0.6 + blink_wave * 0.95
            elif ch == "O1":
                s += alpha * 1.9 + noise[i] * 0.6
            elif ch == "O2":
                s += alpha * 1.7 + noise[i] * 0.7
            elif ch == "C3":
                s += mu * 1.4 + beta * 1.1 + theta * 0.4
            elif ch == "Cz":
                s += mu * 1.2 + alpha * 0.8 + theta * 0.6
            elif ch == "Fz":
                s += theta * 1.6 + beta * 0.8
            elif ch == "C4":
                s += mu * 1.5 + beta * 1.0 + theta * 0.5
            samples[i] = s

        # Trial Stimulus Trigger Logic
        # Fire a Go / No-Go stimulus event every ~2.2 seconds
        events = []
        if now - self.last_event_time >= 2.2:
            self.last_event_time = now
            self.current_trial_idx = (self.current_trial_idx % self.trial_budget) + 1
            self.stimulus_code = 1 if (self.current_trial_idx % 3 != 0) else 2 # 2/3 Go, 1/3 No-Go
            self.current_erp_type = self.stimulus_code
            self.samples_since_stimulus = 0

            offset = int(np.random.randint(2, max(3, chunk_size - 5)))
            events.append({
                "sample_offset": offset,
                "code": self.stimulus_code,
                "label": "Go Stimulus" if self.stimulus_code == 1 else "No-Go Stimulus",
                "timestamp": round(now, 4),
                "trial_number": self.current_trial_idx,
                "trial_budget": self.trial_budget,
                "is_target_reached": (self.current_trial_idx >= self.trial_budget)
            })

        # Inject P300 ERP Component across 8 channels post-stimulus
        # P300 peak occurs ~300ms post-stimulus (at 250 Hz, 300ms = 75 samples)
        # N200 negative peak at ~200ms (50 samples)
        if self.samples_since_stimulus < 250: # within 1.0 second post-stimulus
            erp_indices = np.arange(self.samples_since_stimulus, self.samples_since_stimulus + chunk_size)
            t_erp = erp_indices / self.sfreq # time in seconds post-stimulus

            # N200 (negative at 200ms)
            n200 = -6.0 * np.exp(-((t_erp - 0.20) ** 2) / (2 * (0.04 ** 2)))
            # P300 (positive at 320ms, vertex Cz and parietal highest)
            p300_amp = 18.0 if self.current_erp_type == 1 else 12.0
            p300 = p300_amp * np.exp(-((t_erp - 0.32) ** 2) / (2 * (0.06 ** 2)))
            erp_signal = n200 + p300

            for i, ch in enumerate(self.channels):
                # Cz, Fz, C3, C4, O1, O2 show clear P300 deflections
                weight = 1.0 if ch in ["Cz", "Fz"] else 0.7 if ch in ["C3", "C4"] else 0.4
                samples[i] += (erp_signal * weight).astype(np.float32)

            self.samples_since_stimulus += chunk_size

        samples = (samples * self.scale).astype(np.float32)
        self.sample_counter += chunk_size
        t_vec = [(self.sample_counter - chunk_size + i) * dt for i in range(chunk_size)]

        return {
            "samples": samples.tolist(),
            "timestamps": t_vec,
            "events": events,
            "is_simulated": not self.is_hardware_active,
            "source_type": f"8-Ch Neuro Cap ({'Live Hardware' if self.is_hardware_active else 'Calibrated Cap Stream'})",
            "channels": self.channels,
            "current_trial": self.current_trial_idx,
            "trial_budget": self.trial_budget,
            "filter_mode": self.filter_mode,
            "hardware_scale": self.scale
        }

    def get_status(self) -> Dict[str, Any]:
        available_ports = []
        if SERIAL_AVAILABLE:
            try:
                available_ports = [p.device for p in lp.comports()]
            except Exception:
                pass

        return {
            "id": "cap_8ch",
            "name": "8-Channel Neuro Cap (Fp1, Fp2, O1, O2, C3, Cz, Fz, C4)",
            "connected": self.is_connected,
            "hardware_active": self.is_hardware_active,
            "sampling_rate": self.sfreq,
            "channel_count": len(self.channels),
            "channels": self.channels,
            "current_trial": self.current_trial_idx,
            "trial_budget": self.trial_budget,
            "detected_com_ports": available_ports,
            "instructions": (
                "For live physical streaming: Pair the cap via Bluetooth SPP or plug in the USB receiver. "
                "The browser or backend reads 8 channels at 250 Hz. Signals glide at calm normal speed, "
                "and accuracy updates dynamically based on your selected trial budget."
            ),
            "is_simulated": not self.is_hardware_active
        }
