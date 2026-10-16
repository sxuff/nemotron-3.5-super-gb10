#!/usr/bin/env bash
# Nemotron 3.5 Super VL 120B-A12B NVFP4 + LiLiCorr (k=7) on one DGX Spark / GB10.
# NVIDIA's DGX Spark quick-start, with two changes required to start on one GB10:
#   --max-model-len 262144                      (1M default exhausts host memory at startup)
#   --limit-mm-per-prompt video=0                (video encoder profiling exhausts host memory; images uncapped)
set -euo pipefail
MODELS=${MODELS:-$HOME/models}
TARGET=${TARGET:-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
DRAFT=${DRAFT:-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2}
PORT=${PORT:-8000}
NAME=${NAME:-nemotron35-super}

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" --gpus all --ipc=host --shm-size 16g \
  -p 127.0.0.1:${PORT}:8000 \
  -v "$MODELS":/models:ro \
  -v "${CACHE:-$HOME/.cache/nemotron35-vllm}":/root/.cache \
  "$IMAGE" \
  /models/$TARGET \
  --served-model-name nemotron-3.5-super-vl \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 \
  --max-num-batched-tokens 8192 \
  --max-model-len 262144 \
  --limit-mm-per-prompt "{\"video\":0,\"image\":${MAX_IMAGES:-999}}" \
  --speculative_config.model /models/$DRAFT \
  --speculative_config.method dflash \
  --speculative_config.num_speculative_tokens 7 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  ${EXTRA_ARGS:-}
