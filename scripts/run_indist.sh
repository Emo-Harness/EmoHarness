#!/usr/bin/env bash
# The in-distribution effect check, end to end, resumable, on one or two boxes.
#
#   scripts/run_indist.sh                       # this machine's default steps
#   EMOHARNESS_MACHINE=2 scripts/run_indist.sh  # the second box's steps
#   STEPS=row_l3,row_l6 scripts/run_indist.sh   # exactly these steps
#
# Protocol: build/val carved out of MME-Emotion's seen tasks by
# examples/split_benchmark_indist.py (already applied: data/manifests/), the
# Stage 1 scaffold as L3, Stage 2 Builder replicates with GPT 5.6 Luna (the
# same model verifying each round's proposals), and L0 / L3 / L4 / L6 on the
# 1,000-item subset of the remaining test items, scored locally (no official
# judge). Preprocessing is deterministic tools only (ffmpeg, whisper, openSMILE);
# no audio-language model is served. Every step writes
# results/indist/<step>.done on success and logs to logs/indist_<step>.log.
#
# Two machines share this checkout, the caches, and the markers over NFS; each
# serves its own vLLM on localhost and writes its own ledger file.
#
# Memory: the cgroup is 8 GB and the whole container is restarted on OOM (and
# the GPU keeps a ghost allocation afterwards), so a watchdog restarts the
# server and the current step when anon memory passes MEM_LIMIT_MB. Every
# step resumes from the call cache, so a restart costs minutes, not hours.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
MARK="${ROOT}/results/indist"
mkdir -p "${MARK}" logs

# --- environment shared by every step ---------------------------------------
MACHINE="${EMOHARNESS_MACHINE:-1}"
# Secrets live outside the checkout; ~/.emoharness_env (chmod 600, on the shared
# home) is sourced so both boxes see the same OPENAI_API_KEY. Never log it.
KEY_FILE="${EMOHARNESS_ENV_FILE:-${HOME}/.emoharness_env}"
load_key_file() { if [[ -f "${KEY_FILE}" ]]; then set -a; source "${KEY_FILE}"; set +a; fi; }
load_key_file
NV="${ROOT}/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cublas/lib:${NV}/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export EMOHARNESS_WORKERS="${EMOHARNESS_WORKERS:-16}"
# At most this many video-carrying requests in flight (host RAM, see openai_compat.py).
# Eight, not three: the box moved on 2026-09-12 to 10 CPU / 32 GB / H200-144GB
# and the old cap left the GPU at 2% KV utilisation (see scripts/run_students.sh).
export EMOHARNESS_VIDEO_WORKERS="${EMOHARNESS_VIDEO_WORKERS:-8}"
export VLLM_MM_CACHE_GB="${VLLM_MM_CACHE_GB:-4}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
if [[ "${MACHINE}" != "1" ]]; then
  export EMOHARNESS_LEDGER_NAME="${EMOHARNESS_LEDGER_NAME:-ledger.m${MACHINE}.jsonl}"
fi
ROUNDS="${EMOHARNESS_ROUNDS:-6}"
SUBSET_PER_TASK="${EMOHARNESS_SUBSET_PER_TASK:-125}"
LUNA_BUDGET_USD="${EMOHARNESS_LUNA_BUDGET_USD:-40}"
# 84% of the 32 GB cgroup, as 6800 was of the old 8 GB.
MEM_LIMIT_MB="${EMOHARNESS_MEM_LIMIT_MB:-27000}"

log() { printf '%s [m%s] %s\n' "$(date '+%F %T')" "${MACHINE}" "$*"; }

# --- servers -----------------------------------------------------------------
healthy() { curl -sf "http://127.0.0.1:${1}/health" >/dev/null 2>&1; }
wait_ready() { # port, seconds
  local port="$1" deadline=$(( $(date +%s) + ${2:-900} ))
  until healthy "${port}"; do
    if (( $(date +%s) > deadline )); then log "server on ${port} not ready"; return 1; fi
    sleep 10
  done
}
start_server() { # student (the only server this protocol runs)
  local kind="$1" port pid
  case "${kind}" in student) port=8000 ;; *) log "unknown server ${kind}"; return 1 ;; esac
  if healthy "${port}"; then
    log "${kind} already serving on ${port}"; return 0
  fi
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    log "${kind} answers but its engine is dead; restarting it"
    stop_server "${kind}"
  fi
  nohup bash "scripts/serve_${kind}.sh" > "logs/serve_${kind}.m${MACHINE}.log" 2>&1 &
  pid=$!; echo "${pid}" > "${MARK}/serve_${kind}.m${MACHINE}.pid"
  log "started ${kind} (pid ${pid}); waiting for port ${port}"
  wait_ready "${port}" 1200
}
stop_server() { # student
  local kind="$1" pidfile="${MARK}/serve_${1}.m${MACHINE}.pid"
  if [[ -f "${pidfile}" ]]; then
    pkill -TERM -P "$(cat "${pidfile}")" 2>/dev/null || true
    kill -TERM "$(cat "${pidfile}")" 2>/dev/null || true
    rm -f "${pidfile}"
  fi
  # The API server, its spawned EngineCore, and any straggler all run on the
  # vLLM venv's interpreter; match that path so nothing keeps the GPU.
  pkill -TERM -f "emoharness-vllm011/bin/[p]ython" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "emoharness-vllm011/bin/[p]ython" >/dev/null || break; sleep 2
  done
  pkill -KILL -f "emoharness-vllm011/bin/[p]ython" 2>/dev/null || true
  # The engine core renames itself (setproctitle) and outlives its parent when
  # only the interpreter path is matched; an orphaned one keeps ~40 GB of GPU.
  pkill -KILL -f "VLLM::EngineCor[e]" 2>/dev/null || true
  local used="?"
  for _ in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    (( used < 35000 )) && break; sleep 2
  done
  log "stopped ${kind} (gpu used ${used} MiB)"
}
restart_student() { stop_server student; start_server student; }

# --- memory watchdog ---------------------------------------------------------
anon_mb() { awk '$1=="anon"{printf "%d",$2/1e6}' /sys/fs/cgroup/memory.stat; }
memory_watchdog() {
  while true; do
    local mb top; mb="$(anon_mb)"
    top="$(ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<=3 {printf "%s=%dM ", $2, $1/1024}')"
    printf '%s m%s anon_mb=%s %s\n' "$(date '+%T')" "${MACHINE}" "${mb}" "${top}" >> "logs/memory.m${MACHINE}.log"
    if (( mb > MEM_LIMIT_MB )); then
      log "WATCHDOG: anon ${mb} MB > ${MEM_LIMIT_MB}; killing the step and the server"
      if [[ -f "${MARK}/step.m${MACHINE}.pid" ]]; then
        kill -TERM "$(cat "${MARK}/step.m${MACHINE}.pid")" 2>/dev/null || true
      fi
      stop_server student
      sleep 20
    fi
    sleep 15
  done
}
memory_watchdog &
WATCHDOG_PID=$!
trap 'kill "${WATCHDOG_PID}" 2>/dev/null || true' EXIT

# --- API key -----------------------------------------------------------------
require_key() { # wait (rather than fail and burn attempts) until the key exists
  local waited=0
  while [[ -z "${OPENAI_API_KEY:-}" ]]; do
    load_key_file
    [[ -n "${OPENAI_API_KEY:-}" ]] && break
    if (( waited % 600 == 0 )); then log "waiting for OPENAI_API_KEY (put OPENAI_API_KEY=... in ${KEY_FILE}, chmod 600)"; fi
    sleep 30; waited=$((waited + 30))
  done
  log "OPENAI_API_KEY present (${#OPENAI_API_KEY} chars)"
}

# --- budget guard ------------------------------------------------------------
luna_spend() {
  "${PY}" - <<'EOF'
from pathlib import Path
from emoharness.common.ledger import read_entries
paid = ('builder', 'verifier', 'judge_proxy', 'teacher_direct')
total = 0.0
for ledger in sorted(Path('results').glob('ledger*.jsonl')):
    total += sum(e.cost_usd for e in read_entries(ledger) if e.role in paid)
print(f"{total:.2f}")
EOF
}
check_budget() {
  local spent; spent="$(luna_spend)"
  log "Luna spend so far (all ledgers): \$${spent} (cap \$${LUNA_BUDGET_USD})"
  if "${PY}" -c "import sys; sys.exit(0 if float('${spent}') < float('${LUNA_BUDGET_USD}') else 1)"; then
    return 0
  fi
  log "BUDGET EXCEEDED: \$${spent} >= \$${LUNA_BUDGET_USD}; stopping"; return 1
}

# --- steps -------------------------------------------------------------------
run_py() { # run one example under the watchdog's eye
  "${PY}" "$@" &
  local pid=$!
  echo "${pid}" > "${MARK}/step.m${MACHINE}.pid"
  wait "${pid}"
  local rc=$?
  rm -f "${MARK}/step.m${MACHINE}.pid"
  return "${rc}"
}
step_fit_prosody() { run_py examples/fit_prosody_reference.py; }
# Preprocessing needs no server: whisper runs in-process on the GPU (ctranslate2)
# and openSMILE on the CPU. The student is started afterwards.
step_preprocess_external() { run_py examples/preprocess.py --items external; }
step_leakage() { run_py examples/leakage_report.py --force; }
step_preregister() { run_py examples/preregister.py --size-per-task "${SUBSET_PER_TASK}" --force; }
step_preprocess_subset() { run_py examples/preprocess.py --items subset; }
step_switch_to_student() { start_server student; }
step_row_l0() { start_server student; run_py examples/run_ladder.py --rows L0 --items subset --force; }
step_row_l3() { start_server student; run_py examples/run_ladder.py --rows L3 --items subset --force; }
step_build_stage2() {
  require_key; start_server student; check_budget
  run_py examples/build_harness.py --stage 2 --replicate 0 --rounds "${ROUNDS}" --force
}
step_build_stage2_rep1() {
  require_key; start_server student; check_budget
  run_py examples/build_harness.py --stage 2 --replicate 1 --rounds "${ROUNDS}" --force
}
step_build_stage2_rep2() {
  require_key; start_server student; check_budget
  run_py examples/build_harness.py --stage 2 --replicate 2 --rounds "${ROUNDS}" --force
}
step_row_l4() { start_server student; check_budget; run_py examples/run_ladder.py --rows L4 --items subset --force; }
# Stage 3 unlocks the verification functions (ADR-0005): the Builder may rewrite
# harness/verification/*.py, checked against the static allowlist before any
# candidate is written. Those functions run locally on the stage JSON, so a
# candidate that only edits them reuses every cached Target call. Its base is the
# first completed Stage 2 replicate (ADR-0006 §6), and the Builder now ends the
# stage itself (ADR-0006 §10).
step_build_stage3() {
  require_key; start_server student; check_budget
  run_py examples/build_harness.py --stage 3 --replicate 0 --force
}
step_row_l5() { start_server student; check_budget; run_py examples/run_ladder.py --rows L5 --items subset --force; }
step_row_l6() { require_key; check_budget; run_py examples/run_ladder.py --rows L6 --items subset --force; }
step_analyze() { run_py examples/analyze.py --resamples 2000 --force; }
step_stop_student() { stop_server student; }

# One machine can run everything in this order; a second box sharing the
# checkout takes the steps under "*" (they only need the caches and the
# markers), and each side skips whatever the other has already marked done.
case "${MACHINE}" in
  1) DEFAULT_STEPS=(fit_prosody preprocess_external leakage preregister preprocess_subset switch_to_student
                    row_l0 row_l3 build_stage2 row_l4 build_stage2_rep1 build_stage2_rep2 row_l6 analyze stop_student) ;;
  *) DEFAULT_STEPS=(row_l6 build_stage2_rep2 build_stage2_rep1 stop_student) ;;
esac
IFS=',' read -r -a SELECTED <<< "${STEPS:-$(IFS=,; echo "${DEFAULT_STEPS[*]}")}"

for step in "${SELECTED[@]}"; do
  # stop_student is an action, not a milestone: it must run every time.
  if [[ "${step}" != "stop_student" && -f "${MARK}/${step}.done" ]]; then log "skip ${step} (done)"; continue; fi
  log "=== ${step} ==="
  printf '\n===== %s [m%s] %s =====\n' "$(date '+%F %T')" "${MACHINE}" "${step}" >> "logs/indist_${step}.log"
  if "step_${step}" >> "logs/indist_${step}.log" 2>&1; then
    [[ "${step}" == "stop_student" ]] || touch "${MARK}/${step}.done"; log "=== ${step} ok ==="
  else
    log "=== ${step} FAILED (see logs/indist_${step}.log) ==="; exit 1
  fi
done
log "ALL DONE"
