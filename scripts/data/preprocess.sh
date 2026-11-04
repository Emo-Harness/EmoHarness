#!/usr/bin/env bash
# Phase 3: the perception cache (2 fps frames, Whisper, openSMILE) for every clip.
# EMOHARNESS_ITEMS picks the item set (default all); a GPU and ffmpeg are needed.
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
run_example data/preprocess.py --items "${EMOHARNESS_ITEMS:-all}" "$@"
