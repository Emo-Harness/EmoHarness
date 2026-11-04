#!/usr/bin/env bash
# The paper's conditions on the evaluation items: EMOHARNESS_ROWS (default L0,L1,L3 =
# Video, Audio, Shared), EMOHARNESS_ITEMS (default clean), EMOHARNESS_TARGET a
# configs/target/*.yaml, EMOHARNESS_BUNDLE a harness bundle (L5 = MHE).
# VLLM_ENDPOINT points at the served student.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ARGS=(--rows "${EMOHARNESS_ROWS:-L0,L1,L3}" --items "${EMOHARNESS_ITEMS:-clean}")
if [[ -n "${EMOHARNESS_TARGET:-}" ]]; then ARGS+=(--target "${EMOHARNESS_TARGET}"); fi
if [[ -n "${EMOHARNESS_BUNDLE:-}" ]]; then ARGS+=(--bundle "${EMOHARNESS_BUNDLE}"); fi
run_example mme/run_ladder.py "${ARGS[@]}" "$@"
