#!/usr/bin/env bash
# One status line every INTERVAL seconds while scripts/run_indist_forever.sh
# runs on this machine; step transitions, failures, and watchdog restarts are
# reported as soon as they appear. Exits when the wrapper finishes or gives up.
#   MACHINE=2 bash scripts/monitor_indist.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MACHINE="${MACHINE:-${EMOHARNESS_MACHINE:-1}}"
INTERVAL="${MONITOR_INTERVAL:-900}"
RUNLOG="logs/run_indist.m${MACHINE}.log"
# The log is appended across runs; only lines written after this monitor
# started belong to the run it watches.
seen=$(wc -l < "${RUNLOG}" 2>/dev/null || echo 0)
start=${seen}
spend() {
  .venv/bin/python - 2>/dev/null <<'PY' || echo "?"
from pathlib import Path
from emoharness.common.ledger import read_entries
paid = ('builder', 'judge_proxy', 'teacher_direct')
total = sum(x.cost_usd for p in Path('results').glob('ledger*.jsonl') for x in read_entries(p) if x.role in paid)
print(f"{total:.2f}")
PY
}
status_line() {
  local step rounds anon gpu errs oom
  step="$(grep -o '=== [a-z0-9_]* ===' "${RUNLOG}" 2>/dev/null | tail -1 | tr -d '=' | xargs)"
  rounds="$(ls -d results/construction/stage2/*/round_* 2>/dev/null | wc -l)"
  anon="$(awk '$1=="anon"{printf "%.1fG",$2/1e9}' /sys/fs/cgroup/memory.stat)"
  gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%dG/%d%%",$1/1024,$2}')"
  errs="$(grep -c 'Traceback\|ERROR' "logs/indist_${step}.log" 2>/dev/null || echo 0)"
  oom="$(awk '$1=="oom_kill"{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)"
  echo "[$(date '+%H:%M') m${MACHINE}] step=${step:-?} rounds=${rounds} anon=${anon} gpu=${gpu} oom_kill=${oom:-?} luna=\$$(spend) errs=${errs} rows=$(ls results/raw_predictions/ 2>/dev/null | tr '\n' ' ')"
}
while true; do
  total=$(wc -l < "${RUNLOG}" 2>/dev/null || echo 0)
  if (( total > seen )); then
    tail -n +$((seen + 1)) "${RUNLOG}" | grep -E '=== .* (ok|FAILED) ===|ALL DONE|BUDGET|WATCHDOG|attempt|finished|giving up' || true
    seen=${total}
  fi
  if tail -n +$((start + 1)) "${RUNLOG}" 2>/dev/null | grep -qE '\] finished$|giving up|budget exceeded; not retrying'; then status_line; echo "RUN ENDED: $(tail -1 "${RUNLOG}")"; exit 0; fi
  if ! pgrep -f "run_indist_forever" >/dev/null; then status_line; echo "RUN WRAPPER GONE without finishing"; exit 2; fi
  now=$(date +%s)
  if (( now - ${last:-0} >= INTERVAL )); then status_line; last=${now}; fi
  sleep 30
done
