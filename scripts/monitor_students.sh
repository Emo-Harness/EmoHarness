#!/usr/bin/env bash
# One status line every INTERVAL seconds while scripts/run_students_forever.sh
# runs on this machine, plus an immediate line for every step transition,
# failure, or watchdog restart. Exits when the wrapper finishes or gives up.
#
#   MONITOR_INTERVAL=1800 bash scripts/monitor_students.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MACHINE="${MACHINE:-${EMOHARNESS_MACHINE:-1}}"
INTERVAL="${MONITOR_INTERVAL:-1800}"
RUNLOG="logs/run_students.m${MACHINE}.log"
# The log is appended across runs; only lines written after this monitor
# started belong to the run it watches.
seen=$(wc -l < "${RUNLOG}" 2>/dev/null || echo 0)
start=${seen}

# Counting every cached call over NFS is slow, so count one of the 256 shards
# and scale: a rough monotonic progress signal, cheap enough to poll.
cached_calls() {
  local n
  n=$(ls data/derived/calls/00 2>/dev/null | wc -l)
  echo $(( n * 256 ))
}

status_line() {
  local step rows anon gpu oom done_rows served elapsed
  step="$(grep -oE '=== [0-9a-z]+ [A-Za-z0-9_]+ ===' "${RUNLOG}" 2>/dev/null | tail -1 | tr -d '=' | xargs)"
  done_rows="$(ls results/indist/*_L*.done results/indist/*_best_of_4.done 2>/dev/null | wc -l)"
  anon="$(awk '$1=="anon"{printf "%.1fG",$2/1e9}' /sys/fs/cgroup/memory.stat)"
  gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%dG/%d%%",$1/1024,$2}')"
  oom="$(awk '$1=="oom_kill"{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)"
  served="$(curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null |
    .venv/bin/python -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)"
  elapsed="$(grep -n "=== ${step} ===" "${RUNLOG}" 2>/dev/null | tail -1 | cut -d: -f2- | awk '{print $2}')"
  echo "[$(date '+%H:%M') m${MACHINE}] step=${step:-?} since=${elapsed:-?} rows_done=${done_rows}/13 serving=${served##*/} calls~$(cached_calls) anon=${anon} gpu=${gpu} oom_kill=${oom:-?}"
}

while true; do
  total=$(wc -l < "${RUNLOG}" 2>/dev/null || echo 0)
  if (( total > seen )); then
    tail -n +$((seen + 1)) "${RUNLOG}" |
      grep -E '=== .* (ok|FAILED) ===|ALL DONE|WATCHDOG|##### student|attempt|giving up|never became ready' || true
    seen=${total}
  fi
  if tail -n +$((start + 1)) "${RUNLOG}" 2>/dev/null | grep -qE '\] finished$|giving up'; then
    status_line; echo "RUN ENDED: $(tail -1 "${RUNLOG}")"; exit 0
  fi
  if ! pgrep -f "run_students_forever" >/dev/null; then
    status_line; echo "RUN WRAPPER GONE without finishing"; exit 2
  fi
  now=$(date +%s)
  if (( now - ${last:-0} >= INTERVAL )); then status_line; last=${now}; fi
  sleep 30
done
