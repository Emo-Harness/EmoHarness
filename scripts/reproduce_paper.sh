#!/usr/bin/env bash
# Every stage of the paper's pipeline, in order: data, mme, mhe, intentqa, report.
# Each stage skips the steps that already finished, so an interrupted run resumes.
#
#   scripts/reproduce_paper.sh                       # everything, on this machine
#   scripts/reproduce_paper.sh mhe --students qwen3vl_4b
#   scripts/reproduce_paper.sh all --dry-run         # print the steps only
#   scripts/reproduce_paper.sh all --fake --root /tmp/emoharness-paper-fake
#
# See experiments/paper/pipeline.py for what each stage needs.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
STAGE="${1:-all}"
shift || true
cd "${EMOHARNESS_REPO_ROOT}"
${PYTHON} -m experiments.paper.pipeline "${STAGE}" "$@"
