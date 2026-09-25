#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

exec uvicorn backend.app.main:app \
  --host "${PATHOVISION_API_HOST:-0.0.0.0}" \
  --port "${PATHOVISION_API_PORT:-8000}" \
  --reload
