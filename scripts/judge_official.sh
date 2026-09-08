#!/usr/bin/env bash
# Phase 7: official judging. EMOHARNESS_ROW and EMOHARNESS_ITEMS select the run;
# MME_EMOTION_REPO, MME_EMOTION_ROOT, and MME_EMOTION_EVIDENCE_DIR locate the
# official checkout, the videos, and the judge's own evidence directory.
# Keys: OPENAI_API_KEY, which the official scripts read for the judge model.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ARGS=(--row "${EMOHARNESS_ROW:-L5}" --items "${EMOHARNESS_ITEMS:-full}")
run_example judge_official.py "${ARGS[@]}" "$@"
