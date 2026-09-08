#!/usr/bin/env bash
# The whole pipeline in --fake mode, end to end, into a scratch root.
# No GPU, no data, no API keys, no network. Override the destination with
# EMOHARNESS_FAKE_ROOT; it is recreated from scratch on every run.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

FAKE_ROOT="${EMOHARNESS_FAKE_ROOT:-/tmp/emoharness-fake}"
rm -rf "${FAKE_ROOT}"
mkdir -p "${FAKE_ROOT}"
export EMOHARNESS_ROOT="${FAKE_ROOT}"

step() {
  echo
  echo "=== $* ==="
  run_example "$@" --fake
}

step prepare_benchmark.py
step prepare_external.py
step preprocess.py --items all
step leakage_report.py
step preregister.py
step run_ladder.py --rows L0,L1,L2,L3,best_of_4
step build_harness.py --stage 2
step run_ladder.py --rows L4
step build_harness.py --stage 3 --freeze
step run_ladder.py --rows L5,L6
step judge_official.py --row L0 --items full
step judge_official.py --row L3 --items full
step judge_official.py --row L5 --items full
step calibrate_proxy_judge.py --row L3 --items full
step analyze.py --resamples 200

echo
echo "=== done: ${FAKE_ROOT} ==="
cat "${FAKE_ROOT}/results/metrics/ladder.md"
