# scripts/start_frontend.ps1
# Starts the Vite React Frontend Development Server on Windows

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ProjectDir "frontend"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting EEG Platform Frontend (Vite + React + TS)     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $FrontendDir

if (-not (Test-Path "node_modules")) {
    Write-Host "[Notice] node_modules not found. Running npm install..." -ForegroundColor Yellow
    npm install
}

Write-Host "Launching Vite development server..." -ForegroundColor Green
npm run dev
