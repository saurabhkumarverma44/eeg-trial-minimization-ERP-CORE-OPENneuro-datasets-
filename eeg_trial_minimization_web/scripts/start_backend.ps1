# scripts/start_backend.ps1
# Starts the FastAPI EEG Platform Backend on Windows

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting EEG Platform Backend (FastAPI + PyTorch)      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $ProjectDir

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[Error] Virtual environment not found. Please run 'py -3.12 -m venv .venv' first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "[Notice] .env not found. Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
