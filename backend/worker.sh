#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

exec celery \
  -A backend.app.celery_app.celery_app \
  worker \
  -Q "${PATHOVISION_CELERY_QUEUE:-analysis}" \
  --loglevel="${CELERY_LOG_LEVEL:-INFO}"
