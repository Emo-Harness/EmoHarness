#!/usr/bin/env bash
# Phase 5: draw the subset and render docs/preregistration.md, then tag prereg-v1.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# shellcheck disable=SC2086
run_example preregister.py ${EMOHARNESS_PREREGISTER_ARGS:-} "$@"
