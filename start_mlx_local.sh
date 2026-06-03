#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-checkpoints/fish-audio-s2-pro-8bit-mlx-normalized}"
HOST_PORT="${HOST_PORT:-8881}"
API_PORT="${API_PORT:-8880}"
PYTHON="${PYTHON:-.venv-mlx/bin/python}"

"${PYTHON}" local_mlx/host_server.py \
  --host 127.0.0.1 \
  --port "${HOST_PORT}" \
  --model-path "${MODEL_PATH}" &

HOST_PID=$!
cleanup() {
  kill "${HOST_PID}" 2>/dev/null || true
}
trap cleanup EXIT

until curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null; do
  sleep 1
done

API_PORT="${API_PORT}" FISH_MLX_UPSTREAM="http://host.docker.internal:${HOST_PORT}" \
  docker compose -f compose.mlx.yml up --build
