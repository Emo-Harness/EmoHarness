#!/usr/bin/env bash
# Phase 1: benchmark manifest, checksums, and annotation report.
# EMOHARNESS_BENCHMARK_ARGS adds defaults; MME_EMOTION_ROOT names the source tree.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# shellcheck disable=SC2086
run_example prepare_benchmark.py ${EMOHARNESS_BENCHMARK_ARGS:-} "$@"
