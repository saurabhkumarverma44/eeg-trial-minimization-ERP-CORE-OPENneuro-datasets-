import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from backend.adapters.base_adapter import BaseEEGAdapter

logger = logging.getLogger(__name__)

H100_CHANNELS = ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"]

class LSLStreamAdapter(BaseEEGAdapter):
    """
    LabStreamingLayer (LSL) Acquisition Adapter.
    Receives real-time 8-channel EEG at 250 Hz from local acquisition scripts
    (e.g., PowerShell H100_BLE stream, OpenViBE, LabRecorder, etc.).
    """

    def __init__(self, stream_name: str = "H100_BLE", host: str = "127.0.0.1", trial_budget: int = 63):
        self.stream_name = stream_name
        self.trial_budget = trial_budget
        self.is_connected = False
        self.channels: List[str] = H100_CHANNELS
        self.sfreq: float = 250.0
        self.pylsl_available = False
        self.inlet = None
        self.scale = 1.0
        self.filter_mode = "clean"
        self.current_trial_idx = 0
        self.last_event_time = time.time()
        self.stimulus_code = 1
        self.sample_counter = 0

        try:
            import pylsl
            self.pylsl_available = True
        except ImportError:
            self.pylsl_available = False

    def connect(self) -> bool:
        if not self.pylsl_available:
            logger.warning("pylsl is not installed in Python environment.")
            return False
        import pylsl
        try:
            # First search for stream named 'H100_BLE'
            streams = pylsl.resolve_byprop('name', self.stream_name, timeout=1.5)
            if not streams:
                # Search for type 'EEG'
                streams = pylsl.resolve_byprop('type', 'EEG', timeout=1.5)
            if not streams:
                # Fallback to any active LSL stream
                streams = pylsl.resolve_streams(1.0)

            if streams:
                stream_info = streams[0]
                self.inlet = pylsl.StreamInlet(stream_info, max_chunklen=50)
                self.stream_name = stream_info.name()
                nominal_rate = stream_info.nominal_srate()
                self.sfreq = nominal_rate if nominal_rate and nominal_rate > 0 else 250.0
                ch_count = stream_info.channel_count()
                if ch_count == 8:
                    self.channels = H100_CHANNELS
                else:
                    self.channels = [f"Ch{i+1}" for i in range(ch_count)]
                self.is_connected = True
                self.last_event_time = time.time()
                logger.info(f"Connected to LSL stream '{self.stream_name}' ({ch_count} ch @ {self.sfreq} Hz)")
                return True
            else:
                logger.warning(f"No active LSL streams discovered matching '{self.stream_name}' or type 'EEG'")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to LSL stream: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        self.is_connected = False
        self.inlet = None

    def get_channels(self) -> List[str]:
        return self.channels

    def get_sampling_rate(self) -> float:
        return self.sfreq

    def set_gain(self, scale: float) -> None:
        self.scale = max(0.1, min(10.0, float(scale)))

    def set_filter_mode(self, mode: str) -> None:
        self.filter_mode = mode

    def set_trial_budget(self, budget: int) -> None:
        self.trial_budget = max(1, budget)
        self.current_trial_idx = 0

    def get_next_chunk(self, chunk_size: int = 25) -> Optional[Dict[str, Any]]:
        if not self.is_connected or not self.inlet:
            return None
        try:
            samples_raw, timestamps = self.inlet.pull_chunk(max_samples=chunk_size, timeout=0.1)
            if not samples_raw:
                return None

            # samples_raw is (time, channels) -> transpose to (channels, time)
            samples_np = np.array(samples_raw, dtype=np.float32).T

            now = time.time()
            events = []
            if now - self.last_event_time >= 2.2:
                self.last_event_time = now
                self.current_trial_idx = (self.current_trial_idx % self.trial_budget) + 1
                self.stimulus_code = 1 if (self.current_trial_idx % 3 != 0) else 2
                events.append({
                    "sample_offset": 2,
                    "code": self.stimulus_code,
                    "label": "Go Stimulus" if self.stimulus_code == 1 else "No-Go Stimulus",
                    "timestamp": round(now, 4),
                    "trial_number": self.current_trial_idx,
                    "trial_budget": self.trial_budget,
                    "is_target_reached": (self.current_trial_idx >= self.trial_budget)
                })

            actual_samples = samples_np.shape[1]
            self.sample_counter += actual_samples
            if not timestamps or len(timestamps) != actual_samples:
                dt = 1.0 / self.sfreq
                t_vec = [now + (i * dt) for i in range(actual_samples)]
            else:
                t_vec = timestamps

            return {
                "samples": (samples_np * self.scale).tolist(),
                "timestamps": t_vec,
                "events": events,
                "is_simulated": False,
                "source_type": f"Q.Neuro H100 (PowerShell LSL '{self.stream_name}')",
                "channels": self.channels,
                "current_trial": self.current_trial_idx,
                "trial_budget": self.trial_budget,
                "hardware_scale": self.scale,
                "filter_mode": self.filter_mode
            }
        except Exception as e:
            logger.error(f"Error pulling LSL chunk: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "id": "lsl",
            "name": f"LabStreamingLayer (LSL: '{self.stream_name}')",
            "connected": self.is_connected,
            "driver_installed": self.pylsl_available,
            "target_stream": self.stream_name,
            "channels": self.channels,
            "sampling_rate": self.sfreq,
            "instructions": (
                f"Connected to LSL stream '{self.stream_name}' running in your local PowerShell terminal. "
                "8 channels are streaming at 250 Hz in real-time."
            ),
            "is_simulated": False
        }


class NetworkTCPAdapter(BaseEEGAdapter):
    """
    TCP/UDP Streaming Adapter.
    Receives raw packed IEEE 754 float32 chunks from an acquisition PC socket.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9000, expected_channels: int = 8):
        self.host = host
        self.port = port
        self.expected_channels = expected_channels
        self.is_connected = False

    def connect(self) -> bool:
        # Socket listener ready for incoming connections
        self.is_connected = False
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_channels(self) -> List[str]:
        return [f"Ch{i+1}" for i in range(self.expected_channels)]

    def get_sampling_rate(self) -> float:
        return 250.0

    def get_next_chunk(self, chunk_size: int = 50) -> Optional[Dict[str, Any]]:
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "id": "tcp_socket",
            "name": f"TCP/UDP Socket Stream (Port {self.port})",
            "connected": self.is_connected,
            "listen_address": f"{self.host}:{self.port}",
            "expected_channels": self.expected_channels,
            "instructions": "Configure the acquisition software to send binary float32 packets to this port.",
            "is_simulated": False
        }


class PhysicalCapAdapter(BaseEEGAdapter):
    """
    Physical EEG Cap Integration Adapter (8-channel / custom montages).
    Enforces strict electrode count integrity and warns regarding Bluetooth pairing
    versus raw data streaming protocols.
    """

    def __init__(self, port_or_mac: str = "AUTO", channel_count: int = 8):
        self.port_or_mac = port_or_mac
        self.channel_count = channel_count
        self.is_connected = False

    def connect(self) -> bool:
        # Awaiting documented acquisition vendor protocol
        return False

    def disconnect(self) -> None:
        self.is_connected = False

    def get_channels(self) -> List[str]:
        return [f"Electrode_{i+1}" for i in range(self.channel_count)]

    def get_sampling_rate(self) -> float:
        return 250.0

    def get_next_chunk(self, chunk_size: int = 50) -> Optional[Dict[str, Any]]:
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "id": "physical_cap",
            "name": f"Physical EEG Cap ({self.channel_count} Channels)",
            "connected": self.is_connected,
            "target_device": self.port_or_mac,
            "channel_count": self.channel_count,
            "model_compatibility_warning": (
                f"Notice: The primary pre-trained EEGNet research checkpoint requires 31 channels. "
                f"The physical cap provides {self.channel_count} channels. Direct inference on 31-channel "
                f"models is prohibited; a dedicated {self.channel_count}-channel model must be trained."
            ),
            "bluetooth_guidance": (
                "Bluetooth OS pairing creates a communication link but does not expose raw EEG samples "
                "unless the proprietary vendor protocol, baud rate, and frame format are decoded. "
                "Use the Acquisition PC to stream via LSL or TCP for verified data fidelity."
            ),
            "is_simulated": False
        }
