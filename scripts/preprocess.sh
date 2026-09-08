#!/usr/bin/env bash
# Phase 3: fill the shared clip cache. EMOHARNESS_ITEMS selects what to cache.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# shellcheck disable=SC2086
run_example preprocess.py --items "${EMOHARNESS_ITEMS:-all}" ${EMOHARNESS_PREPROCESS_ARGS:-} "$@"
