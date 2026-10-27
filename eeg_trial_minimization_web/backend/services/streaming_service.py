import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from backend.adapters.base_adapter import BaseEEGAdapter
from backend.adapters.simulated_adapter import SimulatedEEGAdapter
from backend.adapters.playback_adapter import SavedPlaybackAdapter
from backend.adapters.hardware_adapters import LSLStreamAdapter, NetworkTCPAdapter, PhysicalCapAdapter
from backend.adapters.cap8_adapter import Cap8ChannelAdapter
from backend.adapters.h100_ble_adapter import H100BleAdapter
from backend.models.schemas import StreamStatus

logger = logging.getLogger(__name__)

class StreamingService:
    def __init__(self):
        self.trial_budget: int = 63
        self.active_adapter: BaseEEGAdapter = Cap8ChannelAdapter(trial_budget=self.trial_budget)
        self.adapter_id: str = "cap_8ch"
        self.is_streaming: bool = False
        self.connected_websockets: Set[WebSocket] = set()
        self.stream_task: Optional[asyncio.Task] = None
        self.samples_transmitted: int = 0
        self.dropped_chunks: int = 0
        self.latency_ms: float = 0.0
        self.filter_mode: str = "clean"

    async def connect_client(self, websocket: WebSocket):
        await websocket.accept()
        self.connected_websockets.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.connected_websockets)}")

    def disconnect_client(self, websocket: WebSocket):
        self.connected_websockets.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.connected_websockets)}")

    def set_trial_budget(self, budget: int):
        self.trial_budget = budget
        if hasattr(self.active_adapter, "set_trial_budget"):
            self.active_adapter.set_trial_budget(budget)

    def set_gain(self, gain: float):
        if hasattr(self.active_adapter, "set_gain"):
            self.active_adapter.set_gain(gain)

    def set_filter_mode(self, mode: str):
        self.filter_mode = mode
        if hasattr(self.active_adapter, "set_filter_mode"):
            self.active_adapter.set_filter_mode(mode)

    def set_adapter(self, adapter_type: str, dataset_id: str = "DS002680_gonogo_4", channel_count: int = 8, trial_budget: int = 63, port: str = "AUTO", baud_rate: int = 115200, scale: float = 1.0):
        was_streaming = self.is_streaming
        if was_streaming:
            self.stop_stream()

        self.adapter_id = adapter_type
        self.trial_budget = trial_budget

        if adapter_type in ["h100_ble", "ble", "h100"]:
            self.active_adapter = H100BleAdapter(address=port if port != "AUTO" else None, scale=scale, trial_budget=trial_budget)
        elif adapter_type in ["cap_8ch", "physical"]:
            self.active_adapter = Cap8ChannelAdapter(port=port, baud_rate=baud_rate, trial_budget=trial_budget)
        elif adapter_type == "simulated":
            self.active_adapter = SimulatedEEGAdapter(sampling_rate=1000.0, num_channels=channel_count if channel_count > 0 else 31)
        elif adapter_type == "playback":
            self.active_adapter = SavedPlaybackAdapter(dataset_id=dataset_id, sampling_rate=1000.0)
        elif adapter_type in ["lsl", "h100_lsl"]:
            self.active_adapter = LSLStreamAdapter(stream_name=port if port not in ["AUTO", ""] else "H100_BLE", trial_budget=trial_budget)
        elif adapter_type == "tcp":
            self.active_adapter = NetworkTCPAdapter(expected_channels=channel_count)
        else:
            self.active_adapter = H100BleAdapter(address=port if port != "AUTO" else None, scale=scale, trial_budget=trial_budget)


        if was_streaming:
            self.start_stream()

    def start_stream(self) -> bool:
        if self.is_streaming:
            return True

        success = self.active_adapter.connect()
        if not success:
            logger.warning(f"Failed to connect adapter '{self.adapter_id}'. Defaulting to simulated.")
            self.active_adapter = SimulatedEEGAdapter()
            self.active_adapter.connect()

        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._stream_loop())
        logger.info(f"Started streaming EEG using adapter: {self.adapter_id}")
        return True

    def stop_stream(self) -> bool:
        self.is_streaming = False
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
        self.active_adapter.disconnect()
        logger.info("Stopped streaming EEG.")
        return True

    async def _stream_loop(self):
        """Streams chunked data strictly synchronized to real-time clock."""
        sfreq = self.active_adapter.get_sampling_rate()
        if sfreq <= 250.0:
            chunk_size = 25  # 100ms chunk at 250Hz
            interval_sec = 0.100
        else:
            chunk_size = 50  # 50ms chunk at 1000Hz
            interval_sec = 0.050

        while self.is_streaming:
            try:
                t0 = asyncio.get_event_loop().time()
                chunk_data = self.active_adapter.get_next_chunk(chunk_size=chunk_size)


                if chunk_data and self.connected_websockets:
                    msg = json.dumps(chunk_data)
                    dead_clients = []
                    for ws in list(self.connected_websockets):
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead_clients.append(ws)

                    for dead in dead_clients:
                        self.connected_websockets.discard(dead)

                    self.samples_transmitted += chunk_size
                    self.latency_ms = round((asyncio.get_event_loop().time() - t0) * 1000.0, 2)
                elif not chunk_data:
                    self.dropped_chunks += 1

                elapsed = asyncio.get_event_loop().time() - t0
                sleep_dur = max(0.005, interval_sec - elapsed)
                await asyncio.sleep(sleep_dur)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                await asyncio.sleep(0.05)

    def get_status(self) -> StreamStatus:
        channels = self.active_adapter.get_channels()
        adapter_stat = self.active_adapter.get_status()
        return StreamStatus(
            is_streaming=self.is_streaming,
            adapter_type=self.adapter_id,
            sampling_rate=self.active_adapter.get_sampling_rate(),
            channel_count=len(channels),
            channel_names=channels,
            samples_transmitted=self.samples_transmitted,
            dropped_chunks=self.dropped_chunks,
            latency_ms=self.latency_ms,
            source_label=adapter_stat.get("name", self.adapter_id)
        )

# Global singleton
streaming_service = StreamingService()
