#!/bin/bash
# Start vLLM server with Qwen3.6-27B for rule card compilation
# RTX PRO 6000 Blackwell (96GB VRAM): model ~54GB, ~38GB free for KV cache
# Reuses model cache from REGEN_3 project
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_CACHE="/home/hako/02_work/02_tyro/REGEN_3/models"
VLLM_VENV="/home/hako/02_work/02_tyro/REGEN_3/.venv-vllm"

# Limit GPU power to 300W to prevent overheating
sudo nvidia-smi -pl 300

echo "Starting vLLM with Qwen3.6-27B for rule card compilation..."
echo "Model cache: $MODEL_CACHE"
echo "vLLM venv: $VLLM_VENV"
echo "Endpoint: http://localhost:8000/v1"

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --download-dir "$MODEL_CACHE" \
  --max-model-len 32768 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
