#!/usr/bin/env bash
# Phase 8: every table, from the raw predictions on disk.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--resamples "${EMOHARNESS_RESAMPLES:-10000}")
if [[ -n "${EMOHARNESS_ROWS:-}" ]]; then ARGS+=(--rows "${EMOHARNESS_ROWS}"); fi
if [[ -n "${EMOHARNESS_ITEMS:-}" ]]; then ARGS+=(--items "${EMOHARNESS_ITEMS}"); fi
run_example analyze.py "${ARGS[@]}" "$@"
