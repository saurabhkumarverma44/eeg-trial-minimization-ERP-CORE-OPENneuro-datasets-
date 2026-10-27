@echo off
title EEG Trial Minimization Platform
echo ========================================================
echo Starting EEG Trial Minimization Research Platform
echo Q.Neuro H100 8-Channel Cap + 250 Hz Real-Time Streaming
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Launching FastAPI Backend Server (Port 8000)...
start "EEG Backend (FastAPI)" cmd /k ".\.venv\Scripts\Activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo [2/2] Launching React Vite Frontend (Port 5173)...
start "EEG Frontend (Vite)" cmd /k "cd frontend && npm run dev"

timeout /t 3 /nobreak >nul

echo.
echo ========================================================
echo Both servers are launching!
echo Open your browser at: http://localhost:5173/live
echo ========================================================
start http://localhost:5173/live
pause
