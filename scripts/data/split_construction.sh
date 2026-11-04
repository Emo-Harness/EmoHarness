#!/usr/bin/env bash
# Phase 2: carve build / val / ext_test out of the seen task types (ER-Lab, ER-Wild, SA).
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
run_example data/split_construction.py "$@"
