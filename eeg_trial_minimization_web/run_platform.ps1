# run_platform.ps1
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting EEG Trial Minimization Research Platform" -ForegroundColor Green
Write-Host "Q.Neuro H100 8-Channel Cap + 250 Hz Real-Time Streaming" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Launch Backend
Write-Host "[1/2] Launching Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$scriptDir'; .\.venv\Scripts\Activate.ps1; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

# Launch Frontend
Write-Host "[2/2] Launching Frontend on http://localhost:5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$scriptDir\frontend'; npm run dev"

Start-Sleep -Seconds 3

Write-Host "Opening browser at http://localhost:5173/live..." -ForegroundColor Green
Start-Process "http://localhost:5173/live"
