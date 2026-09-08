#!/usr/bin/env bash
# Phase 5: proxy-versus-reference judge agreement and the Reasoning gate margin.
# Keys: GEMINI_API_KEY for the proxy judge (Gemini 3 Flash by default, and the
# only step that still needs a Gemini key) and OPENAI_API_KEY for the reference.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--row "${EMOHARNESS_ROW:-L3}" --items "${EMOHARNESS_ITEMS:-external:val}")
run_example calibrate_proxy_judge.py "${ARGS[@]}" "$@"
