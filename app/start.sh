#!/usr/bin/env bash
# Start the AutoMol web app (backend + frontend).
# Run from: MolAgent-Marketplace/MolAgentLight/app/
#
# The frontend is served at: http://localhost:5173
# The backend API is at:     http://localhost:8000/api

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting backend on http://127.0.0.1:8000 ..."
uv run --with fastapi --with uvicorn --with python-multipart --with pydantic-settings --with "fastmcp[tasks]" --with pandas \
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
cd frontend
[ ! -d node_modules ] && npm install
npm run dev -- --port 5173 &
FRONTEND_PID=$!
cd ..

echo ""
echo "═══════════════════════════════════════════════"
echo "  AutoMol Web App running"
echo "  Open: http://localhost:5173"
echo "  API:  http://localhost:8000/api/health"
echo "═══════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
