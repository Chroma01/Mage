#!/usr/bin/env bash
# Launch the Mage-VL-Cowork SGLang inference server via Docker.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGE="${IMAGE:-xiaoyizhang/mage-vl-cowork:0.1.0}"
PORT="${PORT:-30001}"
GPU_DEVICE="${GPU_DEVICE:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-mage-vl-cowork}"

echo "Starting Mage-VL-Cowork SGLang server on 127.0.0.1:${PORT} with model=${MODEL_DIR}"
docker pull "${IMAGE}"

exec docker run --rm --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_DEVICE}" --shm-size=8g \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e PORT="${PORT}" \
  -e SERVED_MODEL_NAME="Mage-VL-Cowork" \
  -e SGLANG_MAX_PIXELS=4000000 \
  --mount "type=bind,src=${MODEL_DIR},dst=/models/Mage-VL-Cowork,readonly" \
  "${IMAGE}" "$@"
