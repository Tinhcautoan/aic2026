#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Launch vLLM server with Qwen2.5-VL-32B-Instruct-AWQ (4-bit)
#
# VRAM budget on A100 80GB:
#   weights  ~16 GB  (AWQ 4-bit)
#   KV cache ~16 GB  (max 8192 context, 10 images)
#   overhead ~2 GB
#   total    ~34 GB  (fits comfortably)
#
# Usage:
#   bash scripts/serve_vlm.sh
#   # or with custom port:
#   VLM_PORT=9000 bash scripts/serve_vlm.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct-AWQ}"
HOST="${VLM_HOST:-0.0.0.0}"
PORT="${VLM_PORT:-8080}"
GPU_UTIL="${VLM_GPU_UTIL:-0.80}"
MAX_CTX="${VLM_MAX_CTX:-8192}"
MAX_IMAGES="${VLM_MAX_IMAGES:-10}"

echo "Starting vLLM server:"
echo "  model       = $MODEL"
echo "  listen      = $HOST:$PORT"
echo "  gpu_util    = $GPU_UTIL"
echo "  max_ctx     = $MAX_CTX"
echo "  max_images  = $MAX_IMAGES per prompt"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --quantization awq \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_CTX" \
    --limit-mm-per-prompt "image=$MAX_IMAGES" \
    --dtype auto
