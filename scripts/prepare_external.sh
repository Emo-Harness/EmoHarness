#!/usr/bin/env bash
# Phase 2: construction manifest, split assignment, per-source label histograms.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# shellcheck disable=SC2086
run_example prepare_external.py ${EMOHARNESS_EXTERNAL_ARGS:-} "$@"
