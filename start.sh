#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p runtime checkpoints data

echo "Starting PathoVision AI..."
echo "Frontend:        http://localhost:5173"
echo "API:             http://localhost:8000"
echo "Swagger:         http://localhost:8000/docs"
echo "RabbitMQ UI:     http://localhost:15672"
echo "Flower:          http://localhost:5555"
echo "MinIO Console:   http://localhost:9001"
e
docker compose up --build
