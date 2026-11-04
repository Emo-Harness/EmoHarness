#!/usr/bin/env bash
# Phase 2 check: the leakage report over both manifests. Report only, as in the
# paper; EMOHARNESS_LEAKAGE_ARGS=--drop removes flagged construction clips instead.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
# shellcheck disable=SC2086
run_example data/leakage_report.py ${EMOHARNESS_LEAKAGE_ARGS:-} "$@"
