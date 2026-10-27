# Troubleshooting & Diagnostic Guide

Common diagnostic procedures and solutions for the EEG Platform.

---

## 1. Port Conflicts

### Symptom: `Address already in use` (Port 8000 or 5173)
If port 8000 (Backend) or port 5173 (Frontend) is already in use by another application:
- Find the process occupying port 8000:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
  ```
- Terminate the process if needed:
  ```powershell
  Stop-Process -Id <PID> -Force
  ```
- Or change `PORT` in `.env`.

---

## 2. Model Checkpoint Loading Issues

### Symptom: Missing keys prefixed with `_orig_mod.`
- **Cause**: The model was saved after using `torch.compile()`.
- **Solution**: The platform's built-in `load_eegnet_checkpoint()` automatically strips `_orig_mod.` prefixes. If uploading custom checkpoints, verify layer names in the **Model Management** validation report.

### Symptom: `Electrode Channel Incompatibility Error`
- **Cause**: Input tensor has a different channel dimension than the model's expected `in_channels` (e.g. 8 channels sent to a 31-channel model).
- **Solution**: Train an 8-channel model or select a 31-channel dataset. Channel dimension mismatch is blocked by design to prevent scientific inaccuracies.

---

## 3. Storage & External Drive Disconnections

### Symptom: `Project root does not exist: D:\...`
- **Cause**: The external Seagate drive was unmounted or assigned a different drive letter by Windows.
- **Solution**:
  1. Check connected drives in PowerShell:
     ```powershell
     Get-PSDrive -PSProvider FileSystem
     ```
  2. If the drive letter changed (e.g., to `E:`), update `EEG_PROJECT_ROOT` in `.env`:
     ```env
     EEG_PROJECT_ROOT=E:\erp_trial_min_backup_20260626\erp_trial_min
     ```
  3. Restart the backend.

---

## 4. WebSocket Streaming Inactive

### Symptom: Canvas waveform shows flat lines or no incoming chunks
- **Check**: Look at the top navigation bar. Click **Start Stream** if it shows stopped.
- Verify that the active adapter is set to `Synthetic Simulated EEG` or `Saved Data Playback` in the **Live EEG Stream** dashboard.
- Check browser console (F12) for WebSocket connection errors to `ws://127.0.0.1:8000/ws/eeg`.
