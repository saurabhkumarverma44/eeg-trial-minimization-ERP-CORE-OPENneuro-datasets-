# Physical EEG Cap & Hardware Integration Guide

This document specifies the technical pathway for integrating external acquisition devices, dedicated acquisition PCs, and physical EEG caps into the platform.

---

## 1. Intended End-to-End Data Path

```
+------------------+       Bluetooth / USB       +------------------------+
|  Physical EEG    | ------------------------->  | Acquisition PC         |
|  Cap (e.g. 8-Ch) |                             | (Vendor Software)      |
+------------------+                             +-----------+------------+
                                                             |
                                                             | LSL / TCP Socket
                                                             v
+------------------------+    WebSocket          +------------------------+
| Frontend Dashboard     | <-------------------  | Laptop Backend Core    |
| (Live Waveforms, ERP)  |                       | (Adapters / Inference) |
+------------------------+                       +------------------------+
```

1. **Physical Cap**: Electrodes record continuous scalp potential.
2. **Acquisition PC**: The vendor's proprietary software handles raw device communication, impedance checking, and packet framing.
3. **Network Bridge**: The acquisition software broadcasts data over the local network via **LabStreamingLayer (LSL)** or a raw **TCP socket**.
4. **Laptop Backend**: Subscribes to the stream, extracts stimulus markers, validates electrode count, and routes chunks to WebSockets.
5. **Dashboard**: Renders 60 FPS scrolling waveforms and triggers real-time ERP predictions.

---

## 2. Critical Electrode Compatibility (Section 11)

### 31-Channel Model vs. 8-Channel Physical Cap
- The verified research checkpoint (`eegnet_DS002680_gonogo_4_best.pt`) was trained strictly on **31 EEG channels** (`(N, 31, 1001)`).
- The physical cap appears to feature **8 channels**.

### Absolute Scientific Prohibitions
1. **Never feed 8-channel data into a 31-channel checkpoint**: The spatial convolutional filter expects 31 specific electrode projections. Feeding mismatched channels causes erroneous predictions.
2. **Never duplicate channels**: Padding 8 channels to 31 creates synthetic spatial correlations that invalidate classification.
3. **Never invent missing electrodes**: The platform strictly enforces dimension matching.

### Required Resolution for Physical Cap Inference
- To classify live trials from an 8-channel cap, train and upload a **dedicated 8-channel EEGNet or 1D CNN model** via the **Model Management** dashboard.
- The platform will validate the 8-channel model and allow activation for live physical streaming.

---

## 3. Bluetooth Pairing vs. Raw EEG Streaming

> [!WARNING]
> Bluetooth pairing in Windows or Linux simply establishes an RFCOMM / BLE link. It does **not** decode EEG samples.

Without the vendor's serial framing protocol, CRC check, and microvolt scale factor:
- Arbitrary serial reads will produce corrupted bytes.
- Direct browser Web Bluetooth is unsupported for medical-grade raw EEG due to permission constraints and buffer dropping.
- **Recommended Setup**: Keep the physical cap connected to the acquisition PC and stream via LSL or TCP.

---

## 4. Setting Up LabStreamingLayer (LSL)

1. Connect both the Acquisition PC and the laptop running this platform to the same local network (Wi-Fi or Ethernet).
2. On the Acquisition PC, enable LSL outlet in your acquisition program (e.g., OpenViBE, LabRecorder, Neuroscan, BrainVision).
3. In the platform, navigate to **Hardware Connections** and set active adapter to `LabStreamingLayer (LSL)`.
4. Install `pylsl` in the platform's `.venv` if required:
   ```powershell
   .\.venv\Scripts\pip.exe install pylsl
   ```

---

## 5. Setting Up TCP Socket Stream

If the vendor software can write raw IEEE 754 float32 packets to a network socket:
1. In vendor software, set target IP to your laptop IP and target port to `9000`.
2. Format: Continuous binary stream of `(num_channels * 4)` bytes per time step.
3. Select `TCP/UDP Socket Stream` in the **Hardware Connections** tab.
