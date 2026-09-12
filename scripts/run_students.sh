#!/usr/bin/env bash
# The ladder decomposition (ADR-0009) and the student size sweep. No paid API.
#
#   scripts/run_students.sh                  # every student, in order
#   STUDENTS=8b scripts/run_students.sh      # only this one
#   ROWS=L1,L2 STUDENTS=8b scripts/run_students.sh
#
# Why these rows. ADR-0009 defines modality gain = L1 - L0, format gain =
# L2 - L1, scaffold gain = L3 - L2, and best-of-4 over L2 as the equal-call
# control for L3's four calls. The 2026-09-11 run measured only L0 and L3, so
# the +11.6 it reports is modality + format + scaffold combined and has no
# matched-compute baseline. This script fills those holes on the 8B student and
# then repeats the whole single-student ladder on Qwen3-VL-2B and 4B, which says
# whether the scaffold gain, and the headroom the Builder works in, depend on
# student capacity.
#
# Each student writes its own results tree (EMOHARNESS_RESULTS_DIR) while every
# cache stays shared: Target-call keys already include the model id, and the
# preprocessing cache is model-independent. The 8B keeps the main tree so its
# new rows sit beside the L0/L3/L4/L6 already there.
#
# Every step writes results/indist/<key>_<row>.done on success and appends to
# logs/students_<key>_<row>.log, so an interrupted pass resumes where it stopped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
MARK="${ROOT}/results/indist"
mkdir -p "${MARK}" logs

MACHINE="${EMOHARNESS_MACHINE:-1}"
NV="${ROOT}/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cublas/lib:${NV}/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export EMOHARNESS_WORKERS="${EMOHARNESS_WORKERS:-16}"
# At most this many video-carrying requests in flight, because the vLLM frontend
# decodes each clip on the CPU and holds it in host RAM.
#
# History, all on the first box (2 CPU / 8 GB): three tripped the watchdog on
# L2, two held but left the GPU at 2% KV utilisation with 85% of each call's
# 36 s spent queueing on this semaphore, i.e. the cap, not the accelerator, set
# the pace. The box moved on 2026-09-12 to 10 CPU / 32 GB / H200-144GB, so the
# cap rises to eight: enough to keep the engine batching, still below the core
# count so decode threads do not starve the client.
export EMOHARNESS_VIDEO_WORKERS="${EMOHARNESS_VIDEO_WORKERS:-8}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
# best-of-n sends one clip n times, so let the server keep the processed video
# instead of decoding it n times; single-call rows never hit this cache and pay
# only its ceiling. Without it best_of_4 tripped the watchdog every few minutes
# once the cache frontier reached the larger ER_Wild clips (2026-09-12 02:48).
# Four GB rather than one now that host RAM is 32 GB.
export VLLM_MM_CACHE_GB="${VLLM_MM_CACHE_GB:-4}"
# Two arenas bounded fragmentation when 8 GB was the whole budget; with 32 GB
# the serialisation it causes across ten cores costs more than it saves.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
# 84% of the cgroup, as 6800 was of the old 8 GB: enough headroom for the kill
# to land before the container itself is OOM-killed.
MEM_LIMIT_MB="${EMOHARNESS_MEM_LIMIT_MB:-27000}"
ITEMS="${EMOHARNESS_ITEMS:-subset}"

log() { printf '%s [m%s] %s\n' "$(date '+%F %T')" "${MACHINE}" "$*"; }

# --- the students, in run order ----------------------------------------------
# key | model id | target config | results dir ('-' keeps the main tree) | rows
STUDENTS_TABLE=(
  "8b|Qwen/Qwen3-VL-8B-Instruct|qwen3vl_8b_instruct.yaml|-|L1,L2,best_of_4"
  "2b|Qwen/Qwen3-VL-2B-Instruct|qwen3vl_2b_instruct.yaml|results/students/qwen3vl-2b|L0,L1,L2,L3,best_of_4"
  "4b|Qwen/Qwen3-VL-4B-Instruct|qwen3vl_4b_instruct.yaml|results/students/qwen3vl-4b|L0,L1,L2,L3,best_of_4"
)

# --- server ------------------------------------------------------------------
healthy() { curl -sf "http://127.0.0.1:${1}/health" >/dev/null 2>&1; }
serving_model() { # the model id the server on 8000 currently answers with
  curl -sf "http://127.0.0.1:8000/v1/models" 2>/dev/null |
    "${PY}" -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' 2>/dev/null || true
}
stop_server() {
  local pidfile="${MARK}/serve_student.m${MACHINE}.pid"
  if [[ -f "${pidfile}" ]]; then
    pkill -TERM -P "$(cat "${pidfile}")" 2>/dev/null || true
    kill -TERM "$(cat "${pidfile}")" 2>/dev/null || true
    rm -f "${pidfile}"
  fi
  pkill -TERM -f "emoharness-vllm011/bin/[p]ython" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "emoharness-vllm011/bin/[p]ython" >/dev/null || break; sleep 2
  done
  pkill -KILL -f "emoharness-vllm011/bin/[p]ython" 2>/dev/null || true
  # The engine core renames itself and outlives its parent when only the
  # interpreter path is matched; an orphan keeps tens of GB of GPU.
  pkill -KILL -f "VLLM::EngineCor[e]" 2>/dev/null || true
  local used='?'
  for _ in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    (( used < 35000 )) && break; sleep 2
  done
  log "stopped the student (gpu used ${used} MiB)"
}
start_server() { # model id
  local model="$1" current pid deadline
  current="$(serving_model)"
  if healthy 8000 && [[ "${current}" == "${model}" ]]; then
    log "${model} already serving"; return 0
  fi
  # A server that is still loading has no open port and no model list yet, so
  # neither check above sees it. Starting a second one then puts two engines on
  # the GPU at once, which is how 2026-09-12 01:05 nearly ran the box out of
  # memory. Any live vLLM process means: stop it first, whatever state it is in.
  if [[ -n "${current}" ]] || pgrep -f "emoharness-vllm011/bin/[p]ython" >/dev/null; then
    log "port 8000 serves ${current:-a server that is still loading}; switching to ${model}"
    stop_server
  fi
  EMOHARNESS_STUDENT_MODEL="${model}" nohup bash scripts/serve_student.sh \
    > "logs/serve_student.m${MACHINE}.log" 2>&1 &
  pid=$!; echo "${pid}" > "${MARK}/serve_student.m${MACHINE}.pid"
  log "started ${model} (pid ${pid}); waiting for port 8000"
  # A first run downloads the weights, so allow for that on top of load time.
  deadline=$(( $(date +%s) + 3600 ))
  until healthy 8000; do
    if (( $(date +%s) > deadline )); then log "${model} never became ready"; return 1; fi
    sleep 10
  done
  log "${model} ready"
}

# --- memory watchdog ---------------------------------------------------------
anon_mb() { awk '$1=="anon"{printf "%d",$2/1e6}' /sys/fs/cgroup/memory.stat; }
memory_watchdog() {
  while true; do
    local mb top; mb="$(anon_mb)"
    top="$(ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<=3 {printf "%s=%dM ", $2, $1/1024}')"
    printf '%s m%s anon_mb=%s %s\n' "$(date '+%T')" "${MACHINE}" "${mb}" "${top}" \
      >> "logs/memory.m${MACHINE}.log"
    if (( mb > MEM_LIMIT_MB )); then
      log "WATCHDOG: anon ${mb} MB > ${MEM_LIMIT_MB}; killing the step and the server"
      if [[ -f "${MARK}/step.m${MACHINE}.pid" ]]; then
        kill -TERM "$(cat "${MARK}/step.m${MACHINE}.pid")" 2>/dev/null || true
      fi
      stop_server
      sleep 20
    fi
    sleep 15
  done
}
memory_watchdog &
WATCHDOG_PID=$!
trap 'kill "${WATCHDOG_PID}" 2>/dev/null || true' EXIT

# --- one row -----------------------------------------------------------------
run_row() { # key, target config, results dir, row
  local key="$1" target="$2" results="$3" row="$4"
  local marker="${MARK}/${key}_${row}.done"
  local logfile="logs/students_${key}_${row}.log"
  if [[ -f "${marker}" ]]; then log "skip ${key} ${row} (done)"; return 0; fi
  log "=== ${key} ${row} ==="
  printf '\n===== %s [m%s] %s %s =====\n' "$(date '+%F %T')" "${MACHINE}" "${key}" "${row}" >> "${logfile}"
  local -a env=()
  [[ "${results}" != '-' ]] && env=(EMOHARNESS_RESULTS_DIR="${results}")
  env "${env[@]}" "${PY}" examples/run_ladder.py \
    --rows "${row}" --items "${ITEMS}" --target "configs/target/${target}" --force \
    >> "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${MARK}/step.m${MACHINE}.pid"
  if wait "${pid}"; then
    rm -f "${MARK}/step.m${MACHINE}.pid"
    touch "${marker}"; log "=== ${key} ${row} ok ==="
  else
    rm -f "${MARK}/step.m${MACHINE}.pid"
    log "=== ${key} ${row} FAILED (see ${logfile}) ==="; return 1
  fi
}

# --- drive -------------------------------------------------------------------
SELECTED="${STUDENTS:-8b,2b,4b}"
for entry in "${STUDENTS_TABLE[@]}"; do
  IFS='|' read -r key model target results rows <<< "${entry}"
  [[ ",${SELECTED}," == *",${key},"* ]] || continue
  rows="${ROWS:-${rows}}"
  # Nothing to do for this student? Then do not pay for a model switch.
  pending=0
  IFS=',' read -r -a row_list <<< "${rows}"
  for row in "${row_list[@]}"; do
    [[ -f "${MARK}/${key}_${row}.done" ]] || pending=1
  done
  if (( pending == 0 )); then log "skip student ${key} (every row done)"; continue; fi
  log "##### student ${key}: ${model} -> ${results} #####"
  start_server "${model}"
  for row in "${row_list[@]}"; do
    run_row "${key}" "${target}" "${results}" "${row}"
  done
done
stop_server
log "ALL DONE"
