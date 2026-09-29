#!/usr/bin/env bash
# Phase 5: proxy-versus-reference judge agreement and the Reasoning gate margin.
# Keys: OPENAI_API_KEY for the proxy judge (configs/builder/proxy_judge.yaml)
# and for the reference judge; GEMINI_API_KEY only if the proxy judge config
# selects `provider: gemini`.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--row "${EMOHARNESS_ROW:-L3}" --items "${EMOHARNESS_ITEMS:-external:val}")
run_example calibrate_proxy_judge.py "${ARGS[@]}" "$@"
