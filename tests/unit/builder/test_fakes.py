"""The Builder double: one valid prompt edit per round, never the same twice."""

from __future__ import annotations

import pytest

from emoharness.builder.fakes import EDITABLE_PROMPT, fake_builder
from emoharness.common.errors import EmoHarnessError


def test_the_builder_proposes_one_editable_prompt_per_round() -> None:
    builder = fake_builder(stage=2, rounds=3)

    proposals = [builder.propose('packet', ()) for _ in range(3)]

    paths = {file.path for p in proposals for c in p.candidates for file in c.files}
    assert paths == {EDITABLE_PROMPT}


def test_every_proposed_prompt_keeps_the_placeholder_its_stage_needs() -> None:
    proposal = fake_builder(stage=3, rounds=1).propose('packet', ())

    assert '{noun}' in proposal.candidates[0].files[0].content


def test_proposals_differ_between_rounds() -> None:
    builder = fake_builder(stage=2, rounds=2)

    first = builder.propose('packet', ()).candidates[0]
    second = builder.propose('packet', ()).candidates[0]

    assert first.files[0].content != second.files[0].content


def test_the_builder_records_the_packet_it_was_given() -> None:
    builder = fake_builder(stage=2, rounds=1)

    builder.propose('the packet', ('frame.jpg',))

    assert builder.calls[0].packet_text == 'the packet'
    assert builder.calls[0].image_paths == ('frame.jpg',)


def test_the_builder_runs_out_after_its_rounds() -> None:
    builder = fake_builder(stage=2, rounds=1)
    builder.propose('packet', ())

    with pytest.raises(EmoHarnessError, match='ran out of proposals'):
        builder.propose('packet', ())
