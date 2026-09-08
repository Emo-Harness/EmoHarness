#!/usr/bin/env bash
# Phases 4 and 7: run ladder rows. EMOHARNESS_ROWS and EMOHARNESS_ITEMS override
# the defaults; EMOHARNESS_TARGET points at a configs/target/*.yaml and
# EMOHARNESS_BUNDLE at a harness bundle directory.
# Keys: VLLM_ENDPOINT for the served Target; OPENAI_API_KEY for the L6 teacher
# row (OPENAI_ENDPOINT overrides its base URL). A teacher config with
# `provider: gemini` reads GEMINI_API_KEY instead.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--rows "${EMOHARNESS_ROWS:-L0,L3,L5}")
if [[ -n "${EMOHARNESS_ITEMS:-}" ]]; then ARGS+=(--items "${EMOHARNESS_ITEMS}"); fi
if [[ -n "${EMOHARNESS_TARGET:-}" ]]; then ARGS+=(--target "${EMOHARNESS_TARGET}"); fi
if [[ -n "${EMOHARNESS_BUNDLE:-}" ]]; then ARGS+=(--bundle "${EMOHARNESS_BUNDLE}"); fi
run_example run_ladder.py "${ARGS[@]}" "$@"
