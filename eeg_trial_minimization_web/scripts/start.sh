#!/usr/bin/env bash
# scripts/start.sh
# Linux / NVIDIA Jetson AGX Orin startup script

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR="$(dirname "$DIR")"

echo "=========================================================================="
echo "  Starting EEG Platform on Linux / Jetson                                 "
echo "=========================================================================="

cd "$PROJECT_DIR"

# Check environment
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start Backend in background
echo "Starting FastAPI Backend..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd "$PROJECT_DIR/frontend"
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo "=========================================================================="
echo "  Platform running: http://localhost:5173"
echo "  Press Ctrl+C to terminate."
echo "=========================================================================="

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
