#!/usr/bin/env bash
# Phase 1: the MME-Emotion manifest, with the media hashes the clean item set needs.
# MME_EMOTION_ROOT names the downloaded tree; EMOHARNESS_BENCHMARK_ARGS adds flags.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
run_example data/prepare_benchmark.py --hash-media ${EMOHARNESS_BENCHMARK_ARGS:-} "$@"
