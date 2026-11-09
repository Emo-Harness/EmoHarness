#!/usr/bin/env bash
# Phase 3: the eGeMAPS reference statistics; run before preprocess.sh, or prosody stays empty.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
run_example data/fit_prosody_reference.py "$@"
