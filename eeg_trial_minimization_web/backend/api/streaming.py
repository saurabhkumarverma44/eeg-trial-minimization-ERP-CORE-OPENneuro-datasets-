from typing import List, Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from backend.services.streaming_service import streaming_service
from backend.models.schemas import StreamStatus, HardwareAdapterInfo

router = APIRouter(tags=["Streaming & Hardware"])

class ConfigureStreamRequest(BaseModel):
    adapter_type: str = "cap_8ch"
    dataset_id: str = "DS002680_gonogo_4"
    channel_count: int = 8
    trial_budget: int = 63
    port: Optional[str] = "AUTO"
    baud_rate: Optional[int] = 115200
    scale: Optional[float] = 1.0

class ConnectBleRequest(BaseModel):
    address: Optional[str] = "AUTO"
    scale: float = 1.0
    trial_budget: int = 63

class ConnectLslRequest(BaseModel):
    stream_name: str = "H100_BLE"
    trial_budget: int = 63

class GainRequest(BaseModel):
    gain: float = 1.0

class FilterRequest(BaseModel):
    mode: str = "clean"  # "clean", "clean_60hz", "notch_50", "notch_60", "raw"

class ConnectCapRequest(BaseModel):
    port: str = "AUTO"
    baud_rate: int = 115200

class TrialBudgetRequest(BaseModel):
    trial_budget: int

@router.get("/api/streaming/status", response_model=StreamStatus)
async def get_stream_status():
    return streaming_service.get_status()

@router.post("/api/streaming/start")
async def start_streaming():
    success = streaming_service.start_stream()
    return {"status": "success", "is_streaming": success}

@router.post("/api/streaming/stop")
async def stop_streaming():
    streaming_service.stop_stream()
    return {"status": "success", "is_streaming": False}

@router.post("/api/streaming/configure")
async def configure_streaming(req: ConfigureStreamRequest):
    streaming_service.set_adapter(
        adapter_type=req.adapter_type,
        dataset_id=req.dataset_id,
        channel_count=req.channel_count,
        trial_budget=req.trial_budget,
        port=req.port or "AUTO",
        baud_rate=req.baud_rate or 115200,
        scale=req.scale or 1.0
    )
    return {"status": "success", "configured_adapter": req.adapter_type, "trial_budget": req.trial_budget}

@router.post("/api/hardware/ble/scan")
async def scan_ble_devices():
    """Scan for nearby Bluetooth Low Energy devices, highlighting Q.Neuro H100."""
    devices = []
    try:
        from bleak import BleakScanner
        discovered = await BleakScanner.discover(timeout=4.0)
        for d in discovered:
            name = d.name or "Unknown BLE Device"
            is_h100 = name.upper().startswith("H100")
            devices.append({
                "name": name,
                "address": d.address,
                "rssi": getattr(d, "rssi", None),
                "is_h100": is_h100
            })
        devices.sort(key=lambda x: (not x["is_h100"], x["name"]))
    except Exception as e:
        return {"devices": [], "count": 0, "error": str(e)}
    return {"devices": devices, "count": len(devices)}

@router.post("/api/hardware/ble/connect")
async def connect_h100_ble(req: ConnectBleRequest):
    """Connect to Q.Neuro H100 (BLE) cap at 250 Hz."""
    try:
        streaming_service.set_adapter(
            adapter_type="h100_ble",
            port=req.address or "AUTO",
            trial_budget=req.trial_budget,
            scale=req.scale
        )
        streaming_service.start_stream()
        return {
            "success": True,
            "adapter": "h100_ble",
            "sampling_rate": 250.0,
            "channels": ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"],
            "message": f"Connected to Q.Neuro H100 (BLE) at 250 Hz (Device: {req.address or 'AUTO'})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect H100 BLE: {str(e)}")

@router.get("/api/hardware/lsl/streams")
async def list_lsl_streams():
    """Discover active LabStreamingLayer (LSL) streams (e.g. from local PowerShell H100_BLE script)."""
    try:
        import pylsl
        streams = pylsl.resolve_streams(0.8)
        found = []
        for s in streams:
            found.append({
                "name": s.name(),
                "type": s.type(),
                "channel_count": s.channel_count(),
                "sampling_rate": s.nominal_srate(),
                "source_id": s.source_id(),
                "is_h100": s.name() == "H100_BLE" or "H100" in s.name()
            })
        return {"streams": found, "count": len(found)}
    except Exception as e:
        return {"streams": [], "count": 0, "error": str(e)}

@router.post("/api/hardware/lsl/connect")
async def connect_lsl_stream(req: ConnectLslRequest):
    """Connect backend streaming service to the active local LSL stream (e.g. running in PowerShell)."""
    try:
        streaming_service.set_adapter(
            adapter_type="lsl",
            port=req.stream_name,
            trial_budget=req.trial_budget
        )
        success = streaming_service.start_stream()
        if not success or not getattr(streaming_service.active_adapter, "is_connected", False):
            raise HTTPException(
                status_code=400,
                detail=f"Could not find active LSL stream '{req.stream_name}'. Ensure your PowerShell script is running with 'Streaming: LSL {req.stream_name}'."
            )
        return {
            "success": True,
            "adapter": "lsl",
            "stream_name": req.stream_name,
            "sampling_rate": 250.0,
            "channels": ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"],
            "message": f"Successfully connected to PowerShell LSL Stream ('{req.stream_name}')! 8 channels streaming @ 250 Hz."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect LSL stream: {str(e)}")

@router.post("/api/hardware/gain")
async def update_hardware_gain(req: GainRequest):
    """Dynamically adjust signal gain / scaling factor."""
    streaming_service.set_gain(req.gain)
    return {"status": "success", "gain": req.gain}

@router.post("/api/hardware/filter")
async def update_hardware_filter(req: FilterRequest):
    """Dynamically set real-time digital filtering mode (clean, clean_60hz, notch_50, notch_60, raw)."""
    streaming_service.set_filter_mode(req.mode)
    return {"status": "success", "filter_mode": req.mode}

@router.get("/api/hardware/filter")
async def get_hardware_filter():
    """Retrieve active real-time filter mode."""
    return {"status": "success", "filter_mode": getattr(streaming_service, "filter_mode", "clean")}

@router.post("/api/hardware/connect_cap")
async def connect_physical_cap(req: ConnectCapRequest):
    """
    Connect to physical 8-channel neuro cap via backend COM port on Windows.
    Works in any browser without needing browser Web Serial API.
    """
    available_ports = []
    try:
        import serial.tools.list_ports as lp
        available_ports = [{"device": p.device, "description": p.description} for p in lp.comports()]
    except Exception:
        pass

    target = req.port
    if target == "AUTO":
        if available_ports:
            target = available_ports[0]["device"]
        else:
            target = None

    if not target:
        return {
            "success": False,
            "is_hardware": False,
            "message": "No physical COM port detected on Windows. Please plug in your USB/Bluetooth receiver or specify a COM port.",
            "available_ports": available_ports
        }

    try:
        import serial
        s = serial.Serial(target, req.baud_rate, timeout=0.1)
        s.close()
        streaming_service.set_adapter(
            adapter_type="cap_8ch",
            channel_count=8,
            port=target,
            baud_rate=req.baud_rate
        )
        streaming_service.start_stream()
        return {
            "success": True,
            "is_hardware": True,
            "connected_port": target,
            "message": f"Successfully connected to 8-channel neuro cap on {target} at {req.baud_rate} baud! Live stream active."
        }
    except Exception as e:
        return {
            "success": False,
            "is_hardware": False,
            "error": str(e),
            "message": f"Could not connect to {target}: {str(e)}.",
            "available_ports": available_ports
        }

@router.post("/api/streaming/trial_budget")
async def update_trial_budget(req: TrialBudgetRequest):
    streaming_service.set_trial_budget(req.trial_budget)
    return {"status": "success", "trial_budget": req.trial_budget}

@router.get("/api/hardware/com_ports")
async def list_com_ports():
    ports = []
    try:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            ports.append({"device": p.device, "description": p.description, "hwid": p.hwid})
    except Exception:
        pass
    return {"ports": ports, "count": len(ports)}

@router.get("/api/hardware/adapters", response_model=List[HardwareAdapterInfo])
async def list_hardware_adapters():
    return [
        HardwareAdapterInfo(
            id="h100_ble",
            name="Q.Neuro H100 (Bluetooth BLE, 250 Hz, 8-Ch)",
            status="active" if streaming_service.adapter_id == "h100_ble" else "available",
            description="Native Bluetooth Low Energy acquisition for Q.Neuro H100 dry-sensor cap (Fp1, Fp2, O1, O2, C3, Cz, Fz, C4) at clinical 250 Hz with real-time gain calibration.",
            supported_channels=[8],
            is_available=True,
            requires_external_service=False,
            config_params={"channels": 8, "sampling_rate": 250.0, "scale": 1.0}
        ),
        HardwareAdapterInfo(
            id="cap_8ch",
            name="8-Channel Neuro Cap (Fp1, Fp2, C3, C4, P3, P4, O1, O2)",
            status="active" if streaming_service.adapter_id in ["cap_8ch", "physical"] else "available",
            description="Dedicated acquisition & calibration adapter for 8-channel research caps. Real-time 250 Hz calm stream, trial-budget tracking, ERP stimulus triggers, and COM port hardware link.",
            supported_channels=[8],
            is_available=True,
            requires_external_service=False,
            config_params={"channels": 8, "sampling_rate": 250.0, "baud_rate": 115200}
        ),
        HardwareAdapterInfo(
            id="simulated",
            name="Synthetic EEG Stream Generator (31-Channel)",
            status="active" if streaming_service.adapter_id == "simulated" else "available",
            description="Synthetic 31-channel multi-rhythm generator (Alpha, Beta, Theta, Pink Noise, Simulated ERP markers). Runs locally without external hardware.",
            supported_channels=[8, 16, 31, 64],
            is_available=True,
            requires_external_service=False,
            config_params={"sampling_rate": 1000.0, "channels": 31}
        ),

        HardwareAdapterInfo(
            id="playback",
            name="Saved Data Replay Adapter",
            status="active" if streaming_service.adapter_id == "playback" else "available",
            description="Streams authentic trials from local research datasets (DS002680, ERP_CORE, DS006018, or FIF files) with realistic playback timing and real markers.",
            supported_channels=[31],
            is_available=True,
            requires_external_service=False,
            config_params={"dataset_id": "DS002680_gonogo_4"}
        ),
        HardwareAdapterInfo(
            id="lsl",
            name="LabStreamingLayer (LSL) Receiver",
            status="standby",
            description="Discovers and subscribes to real-time EEG streams over local LAN/Ethernet from acquisition software (OpenViBE, LabRecorder, Neuroscan).",
            supported_channels=[8, 16, 32, 64],
            is_available=True,
            requires_external_service=True,
            config_params={"stream_name": "EEG", "type": "EEG"}
        ),
        HardwareAdapterInfo(
            id="tcp",
            name="TCP/UDP Socket Stream Receiver",
            status="standby",
            description="Direct network socket listener for custom hardware bridges or acquisition programs broadcasting IEEE-754 float32 packets.",
            supported_channels=[8, 16, 31],
            is_available=True,
            requires_external_service=True,
            config_params={"port": 9000, "protocol": "TCP"}
        ),
        HardwareAdapterInfo(
            id="physical",
            name="Physical EEG Cap (8-Channel)",
            status="standby",
            description="Prepared integration interface for physical EEG cap. Requires acquisition PC bridge or vendor protocol specifications. Strictly validates 8-channel compatibility.",
            supported_channels=[8],
            is_available=False,
            requires_external_service=True,
            config_params={"channels": 8, "baud_rate": 115200}
        )
    ]

@router.websocket("/ws/eeg")
async def websocket_eeg_stream(websocket: WebSocket):
    await streaming_service.connect_client(websocket)
    try:
        while True:
            # Keep socket alive and receive any client messages (e.g. ping/command)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        streaming_service.disconnect_client(websocket)
    except Exception:
        streaming_service.disconnect_client(websocket)
