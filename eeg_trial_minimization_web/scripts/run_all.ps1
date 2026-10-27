# scripts/run_all.ps1
# Complete local launcher for the EEG Trial Minimisation Web Platform

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ProjectDir "frontend"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "   EEG TRIAL-MINIMISATION RESEARCH PLATFORM - LOCAL LAUNCHER              " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

Set-Location $ProjectDir

# 1. Environment check
if (-not (Test-Path ".env")) {
    Write-Host "[1/3] Copying environment configuration..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

# 2. Start Backend
Write-Host "[2/3] Launching FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
$backendProc = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 3. Start Frontend
Write-Host "[3/3] Launching Vite React Frontend on http://localhost:5173 ..." -ForegroundColor Green
Set-Location $FrontendDir
$frontendProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" -PassThru -NoNewWindow

Write-Host "`n==========================================================================" -ForegroundColor Green
Write-Host "  PLATFORM ONLINE AND OPERATIONAL                                         " -ForegroundColor Green
Write-Host "  - Frontend Dashboard:      http://localhost:5173                        " -ForegroundColor White
Write-Host "  - Backend REST API:        http://127.0.0.1:8000                        " -ForegroundColor White
Write-Host "  - Swagger API Docs:        http://127.0.0.1:8000/docs                   " -ForegroundColor White
Write-Host "  - Live Waveform WebSocket: ws://127.0.0.1:8000/ws/eeg                   " -ForegroundColor White
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate both servers.`n" -ForegroundColor Yellow

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "`nStopping servers..." -ForegroundColor Red
    if ($backendProc -and -not $backendProc.HasExited) { Stop-Process -Id $backendProc.Id -Force }
    if ($frontendProc -and -not $frontendProc.HasExited) { Stop-Process -Id $frontendProc.Id -Force }
    Write-Host "Clean shutdown complete." -ForegroundColor Green
}
