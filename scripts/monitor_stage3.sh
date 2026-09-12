#!/usr/bin/env bash
# One status line every INTERVAL seconds while a construction stage runs, plus an
# immediate line for anything notable in its log. Exits when the stage finishes,
# fails, or its process disappears.
#
#   STAGE=3 MONITOR_INTERVAL=900 bash scripts/monitor_stage3.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
STAGE="${STAGE:-3}"
REPLICATE="${REPLICATE:-0}"
INTERVAL="${MONITOR_INTERVAL:-900}"
STEPLOG="logs/indist_build_stage${STAGE}.log"
RUNLOG="logs/stage${STAGE}.m1.log"
OUT="results/construction/stage${STAGE}/${REPLICATE}"
# Both logs are appended across attempts; only what arrives after this monitor
# started belongs to the run it watches.
seen=$(wc -l < "${STEPLOG}" 2>/dev/null || echo 0)

spend() {
  .venv/bin/python - 2>/dev/null <<'PY' || echo '?'
from pathlib import Path
from emoharness.common.ledger import read_entries
paid = ('builder', 'verifier', 'judge_proxy', 'teacher_direct')
total = sum(
    x.cost_usd
    for p in Path('results').glob('ledger*.jsonl')
    for x in read_entries(p)
    if x.role in paid
)
print(f'{total:.2f}')
PY
}

status_line() {
  local rounds accepted vetoed evaluated anon gpu cont
  rounds=$(ls -d "${OUT}"/round_* 2>/dev/null | wc -l)
  # accepted.json carries a name only when a candidate was promoted.
  accepted=$(grep -l '"accepted": "' "${OUT}"/round_*/accepted.json 2>/dev/null | wc -l)
  vetoed=$(grep -o '"issues"' "${OUT}"/round_*/candidates.json 2>/dev/null | wc -l)
  evaluated=$(grep -o '"patch_name"' "${OUT}"/round_*/candidates.json 2>/dev/null | wc -l)
  # The Builder's own stop answer, from the newest proposal it wrote.
  cont=$(.venv/bin/python -c "
import json, sys
from pathlib import Path
files = sorted(Path('${OUT}').glob('round_*/proposal.json'))
if not files:
    print('-')
else:
    d = json.loads(files[-1].read_text())
    print('continue' if d.get('continue_search', True) else 'STOP')
" 2>/dev/null || echo '-')
  anon=$(awk '$1=="anon"{printf "%.1fG",$2/1e9}' /sys/fs/cgroup/memory.stat)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' '{printf "%dG/%d%%",$1/1024,$2}')
  echo "[$(date '+%H:%M')] stage${STAGE} rep${REPLICATE} rounds=${rounds} evaluated=${evaluated} accepted=${accepted} vetoed=${vetoed} builder=${cont} luna=\$$(spend) anon=${anon} gpu=${gpu}"
}

while true; do
  total=$(wc -l < "${STEPLOG}" 2>/dev/null || echo 0)
  if (( total > seen )); then
    tail -n +$((seen + 1)) "${STEPLOG}" |
      grep -E 'stopped|accepted candidate|vetoed candidate|rejected candidate|does not load|Traceback|ERROR|replicate' || true
    seen=${total}
  fi
  if [[ -f "${OUT}/loop_result.json" ]]; then
    status_line
    .venv/bin/python -c "
import json
d = json.load(open('${OUT}/loop_result.json'))
acc = [r['accepted'] for r in d['rounds'] if r['accepted']]
print(f\"STAGE ${STAGE} ENDED: {d['stopped_reason']}, {len(d['rounds'])} rounds, accepted {acc or 'nothing'}\")
"
    exit 0
  fi
  # Watch the orchestrator, not the step: run_indist.sh spends the first few
  # minutes bringing the server up, so build_harness.py does not exist yet and
  # checking for it alone ends this monitor before the stage has begun.
  if ! pgrep -f "run_indis[t].sh" >/dev/null &&
     ! pgrep -f "build_harnes[s].py" >/dev/null; then
    status_line
    echo "STAGE ${STAGE} PROCESS GONE without writing loop_result.json; tail of ${RUNLOG}:"
    tail -3 "${RUNLOG}" 2>/dev/null
    exit 2
  fi
  now=$(date +%s)
  if (( now - ${last:-0} >= INTERVAL )); then status_line; last=${now}; fi
  sleep 30
done
