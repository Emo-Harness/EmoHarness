#!/usr/bin/env bash
# Phase 6: one construction stage. EMOHARNESS_STAGE and EMOHARNESS_REPLICATE
# select the stage and replicate; EMOHARNESS_ROUNDS caps the Builder rounds.
# Pass --freeze to record the freeze and print the git tag command.
# Keys: OPENAI_API_KEY for the Builder and for the proxy judge that scores the
# Reasoning gate (both GPT 5.6 Luna in configs/builder/). A config with
# `provider: gemini` reads GEMINI_API_KEY instead.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--stage "${EMOHARNESS_STAGE:-2}" --replicate "${EMOHARNESS_REPLICATE:-0}")
if [[ -n "${EMOHARNESS_ROUNDS:-}" ]]; then ARGS+=(--rounds "${EMOHARNESS_ROUNDS}"); fi
if [[ -n "${EMOHARNESS_TARGET:-}" ]]; then ARGS+=(--target "${EMOHARNESS_TARGET}"); fi
run_example build_harness.py "${ARGS[@]}" "$@"
