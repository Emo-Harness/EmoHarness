#!/usr/bin/env bash
# Phase 2 gate: the four leakage checks. EMOHARNESS_LEAKAGE_ARGS defaults to --drop.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# shellcheck disable=SC2086
run_example leakage_report.py ${EMOHARNESS_LEAKAGE_ARGS:---drop} "$@"
