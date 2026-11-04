"""Smoke tests for emoharness.common.constants.

The official prompt template's render/parse round trip is covered by
tests/unit/data/test_prompt_template.py; this module only checks the two
invariants constants.py itself is responsible for.
"""

from __future__ import annotations

from emoharness.common.constants import NUM_STEPS, OFFICIAL_SYNONYMS, STEP_TITLES


def test_step_titles_length_matches_num_steps() -> None:
    assert len(STEP_TITLES) == NUM_STEPS


def test_official_synonym_keys_are_lowercase() -> None:
    assert all(key == key.lower() for key in OFFICIAL_SYNONYMS)
