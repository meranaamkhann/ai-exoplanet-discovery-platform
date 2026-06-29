#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         ExoNova — Exoplanet Discovery Platform       ║"
echo "║         ISRO Hackathon 2026 · Problem Statement 7    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Backend
echo "[1/2] Starting FastAPI backend..."
cd "$SCRIPT_DIR/backend"
setsid nohup python3 -m uvicorn main:app \
    --host 0.0.0.0 --port 8000 \
    > server.log 2>&1 < /dev/null &
BACKEND_PID=$!
echo "      PID $BACKEND_PID — waiting for startup..."

for i in {1..30}; do
    sleep 1
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        MODEL_STATUS=$(curl -s http://localhost:8000/api/health | python3 -c "import json,sys; d=json.load(sys.stdin); print('loaded: '+d['model_run_id'] if d['model_loaded'] else 'NOT LOADED — run: python3 ml/train.py')")
        echo "      Backend ready — model $MODEL_STATUS"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "      WARNING: Backend may not have started. Check backend/server.log"
    fi
done

# Frontend
echo ""
echo "[2/2] Starting React dev server..."
echo "      Proxying /api/* → http://localhost:8000"
echo ""
echo "  Dashboard:  http://localhost:5173"
echo "  API docs:   http://localhost:8000/docs"
echo ""
cd "$SCRIPT_DIR/frontend"
npm run dev
