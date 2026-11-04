#!/usr/bin/env bash
# One MHE construction as in the paper: Stage 3, four generations, from the Shared
# scaffold, on the isolated construction split. EMOHARNESS_BUILDER_CONFIG picks the
# Builder (default v3_full.yaml = GPT-5.6 Luna; v3_full_gpt6_1_sol.yaml = GPT-6.1 Sol),
# EMOHARNESS_TARGET the student. Needs OPENAI_API_KEY and the served student.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
export EMOHARNESS_BUILDER_CONFIG="${EMOHARNESS_BUILDER_CONFIG:-v3_full.yaml}"
export EMOHARNESS_EXTERNAL_MANIFEST="${EMOHARNESS_EXTERNAL_MANIFEST:-external_v3.jsonl}"
ARGS=(--stage 3 --replicate "${EMOHARNESS_REPLICATE:-0}" --rounds "${EMOHARNESS_ROUNDS:-4}" --bundle harness)
if [[ -n "${EMOHARNESS_TARGET:-}" ]]; then ARGS+=(--target "${EMOHARNESS_TARGET}"); fi
run_example mhe/build_harness.py "${ARGS[@]}" "$@"
