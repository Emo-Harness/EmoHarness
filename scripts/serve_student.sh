#!/usr/bin/env bash
# Serve the frozen Target (Qwen3-VL-8B-Instruct) with vLLM on port 8000.
# Sized for one A100-80GB in a 2-CPU / 8 GB-RAM cgroup: no CPU swap space, no
# host-side multimodal processor cache, a modest batch, and native video in.
# Override the model with EMOHARNESS_STUDENT_MODEL, the port with VLLM_PORT.
set -euo pipefail
VENV="${EMOHARNESS_VLLM_VENV:-${HOME}/.venvs/emoharness-vllm011}"
MODEL="${EMOHARNESS_STUDENT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
# Triton JIT-compiles a small CUDA helper against Python.h; the system
# python3.12 has no dev headers and we have no root, so point gcc at the
# libpython3.12-dev headers extracted under ~/.local (see docs/run-indist.md).
HDR="${EMOHARNESS_PY_HEADERS:-${HOME}/.local/python3.12-headers/usr/include}"
export CPATH="${HDR}/python3.12:${HDR}${CPATH:+:${CPATH}}"
# Bounded glibc arenas: the frontend decodes videos on many threads and would
# otherwise fragment its heap into gigabytes it never returns.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The GPU may carry memory leaked by an earlier container crash (a context
# whose owner we can no longer see or kill), so size vLLM to what is actually
# free at start rather than to a fixed fraction of the card.
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
AUTO_UTIL="$(python3 -c "print(max(0.30, min(0.90, round((${FREE_MIB} - 8000) / ${TOTAL_MIB}, 2))))")"
echo "gpu free ${FREE_MIB}/${TOTAL_MIB} MiB -> gpu-memory-utilization ${AUTO_UTIL}" >&2
exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-12}" \
  --swap-space 0 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":32,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":32}}' \
  --enable-prefix-caching \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-${AUTO_UTIL}}" \
  --disable-log-requests \
  --port "${PORT}"
