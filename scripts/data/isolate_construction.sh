#!/usr/bin/env bash
# Phase 2: data/manifests/external_v3.jsonl, the isolated construction split every
# MHE run reads (399 / 394 / 129 items for the paper).
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
run_example data/isolate_construction.py "$@"
