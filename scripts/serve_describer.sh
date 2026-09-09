#!/usr/bin/env bash
# Serve the audio describer (Qwen2.5-Omni-7B thinker) with vLLM on port 8001,
# the endpoint `configs/preprocess.yaml` and AUDIO_DESCRIBER_ENDPOINT expect.
# Runs only during preprocessing; stop it before serving the Target.
set -euo pipefail
VENV="${EMOHARNESS_VLLM_VENV:-${HOME}/.venvs/emoharness-vllm011}"
MODEL="${EMOHARNESS_DESCRIBER_MODEL:-Qwen/Qwen2.5-Omni-7B}"
PORT="${DESCRIBER_PORT:-8001}"
# Triton JIT-compiles a small CUDA helper against Python.h; the system
# python3.12 has no dev headers and we have no root, so point gcc at the
# libpython3.12-dev headers extracted under ~/.local (see docs/run-indist.md).
HDR="${EMOHARNESS_PY_HEADERS:-${HOME}/.local/python3.12-headers/usr/include}"
export CPATH="${HDR}/python3.12:${HDR}${CPATH:+:${CPATH}}"
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
  --max-model-len "${DESCRIBER_MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${DESCRIBER_MAX_NUM_SEQS:-8}" \
  --swap-space 0 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"audio":1}' \
  --gpu-memory-utilization "${DESCRIBER_GPU_UTIL:-${AUTO_UTIL}}" \
  --disable-log-requests \
  --port "${PORT}"
