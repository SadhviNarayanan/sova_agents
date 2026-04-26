#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

PIDS="$(lsof -ti "tcp:${PORT}" || true)"
if [[ -n "$PIDS" ]]; then
  echo "Stopping existing service on port ${PORT}: $PIDS"
  kill $PIDS || true
  sleep 1
  PIDS="$(lsof -ti "tcp:${PORT}" || true)"
  if [[ -n "$PIDS" ]]; then
    echo "Force stopping existing service on port ${PORT}: $PIDS"
    kill -9 $PIDS || true
  fi
fi

echo "Starting Sova backend at http://localhost:${PORT}"
exec "$PYTHON" -m uvicorn carerelay_backend.main:app --host "$HOST" --port "$PORT"
