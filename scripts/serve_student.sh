#!/usr/bin/env bash
# Serve one student with vLLM on port 8000, as every evaluation serves it.
#
#   EMOHARNESS_STUDENT_MODEL=Qwen/Qwen3-VL-4B-Instruct \
#   EMOHARNESS_VLLM_VENV=~/.venvs/emoharness-vllm011 scripts/serve_student.sh
#
# EMOHARNESS_STUDENT_MODEL picks the model (default Qwen3-VL-8B-Instruct) and
# EMOHARNESS_VLLM_VENV the vLLM environment (0.11.2 for Qwen3-VL, 0.19.1 for
# Qwen3.5 and Gemma 4). The id advertised on /v1/models is the HF id the target
# configs ask for (--served-model-name). No CPU swap space, native video in, and
# gpu-memory-utilization derived from the VRAM actually free at start. Override
# the port with VLLM_PORT, the context with VLLM_MAX_MODEL_LEN (default 16384),
# and the batch with VLLM_MAX_NUM_SEQS.
set -euo pipefail
VENV="${EMOHARNESS_VLLM_VENV:-${HOME}/.venvs/emoharness-vllm011}"
MODEL="${EMOHARNESS_STUDENT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
# Triton JIT-compiles a small CUDA helper against Python.h; the system
# python3.12 has no dev headers and we have no root, so point gcc at the
# libpython3.12-dev headers extracted under ~/.local.
HDR="${EMOHARNESS_PY_HEADERS:-${HOME}/.local/python3.12-headers/usr/include}"
export CPATH="${HDR}/python3.12:${HDR}${CPATH:+:${CPATH}}"
# The multimodal processor cache is off by default: it buys nothing when every
# request carries a different clip, and host RAM is the scarce resource. Set
# VLLM_MM_CACHE_GB for a row that sends the same clip several times in a row --
# best-of-n resamples one clip n times, and decoding it n times is what pushed
# anon memory past the watchdog on 2026-09-12 while single-call rows were fine.
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
# Two flags differ across the serving environments this project uses.
# vLLM 0.19 removed --swap-space (there is no CPU swap any more, which is what
# --swap-space 0 asked for) and replaced --disable-log-requests with the
# inverted --enable-log-requests, whose default is already off. Passing an
# unknown flag is a hard argparse error, so ask the binary what it accepts
# rather than branching on a version number; on 0.11.2 both are still present
# and the command line is unchanged.
# The venv is used by path, not activated, so its bin/ is not on PATH. Newer
# vLLM JIT-compiles kernels through FlashInfer, which shells out to `ninja`;
# without this the build fails with FileNotFoundError and the engine dies
# mid-request (Qwen3.5's gated-delta attention hits this on every prefill).
export PATH="${VENV}/bin:${PATH}"
# ninja defaults to one compile job per core. On 10 cores the FlashInfer JIT
# build for Qwen3.5's gated-delta kernels started ~10 nvcc/g++ jobs at once and
# added 8 GiB of anon in 15 seconds, which restarted the whole 32 GiB container
# (memory.peak reset, oom_kill still 0, so it is the supervisor, not the cgroup
# OOM killer). Two jobs keep the build inside the budget; it only pays the cost
# once, because the compiled kernels are cached under ~/.cache.
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
HELP="$("${VENV}/bin/vllm" serve --help=all 2>&1 || true)"
case "${HELP}" in *--max-model-len*) ;; *) HELP="$("${VENV}/bin/vllm" serve --help 2>&1 || true)" ;; esac
EXTRA=()
case "${HELP}" in *--swap-space*) EXTRA+=(--swap-space 0) ;; esac
case "${HELP}" in *--disable-log-requests*) EXTRA+=(--disable-log-requests) ;; esac
echo "serving ${MODEL} from ${VENV} with extra flags: ${EXTRA[*]:-none}" >&2
exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}" \
  --mm-processor-cache-gb "${VLLM_MM_CACHE_GB:-0}" \
  --limit-mm-per-prompt '{"image":32,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":32}}' \
  --enable-prefix-caching \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-${AUTO_UTIL}}" \
  "${EXTRA[@]}" \
  --port "${PORT}"
