#!/usr/bin/env bash
# Keep scripts/run_indist.sh going until it prints ALL DONE or has failed
# MAX_ATTEMPTS times in a row. Every attempt resumes from the .done markers
# and the shared call cache, so a watchdog restart or a dead engine costs the
# minutes the server takes to come back, not the work already done.
#
#   scripts/run_indist_forever.sh                     # machine 1
#   EMOHARNESS_MACHINE=2 scripts/run_indist_forever.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MACHINE="${EMOHARNESS_MACHINE:-1}"
MAX_ATTEMPTS="${EMOHARNESS_MAX_ATTEMPTS:-12}"
RUNLOG="logs/run_indist.m${MACHINE}.log"
attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  printf '%s [m%s] attempt %s/%s\n' "$(date '+%F %T')" "${MACHINE}" "${attempt}" "${MAX_ATTEMPTS}" >> "${RUNLOG}"
  if bash scripts/run_indist.sh >> "${RUNLOG}" 2>&1; then
    printf '%s [m%s] finished\n' "$(date '+%F %T')" "${MACHINE}" >> "${RUNLOG}"
    exit 0
  fi
  if grep -q 'BUDGET EXCEEDED' "${RUNLOG}"; then
    printf '%s [m%s] budget exceeded; not retrying\n' "$(date '+%F %T')" "${MACHINE}" >> "${RUNLOG}"
    exit 2
  fi
  sleep 30
done
printf '%s [m%s] giving up after %s attempts\n' "$(date '+%F %T')" "${MACHINE}" "${MAX_ATTEMPTS}" >> "${RUNLOG}"
exit 1
