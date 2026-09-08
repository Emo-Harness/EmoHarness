from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emoharness.common.errors import PromptParseError
from emoharness.data.prompt_template import (
    Quantifier,
    cardinality_for,
    parse_prompt,
    render_prompt,
    render_prompt_for_task,
)
from emoharness.data.tasks import Task

LABELS = ('anger', 'happiness', 'Doubt/Confusion', 'very strong negative')


@pytest.mark.parametrize('task', list(Task))
def test_render_then_parse_round_trips_every_task(task: Task) -> None:
    parsed = parse_prompt(render_prompt_for_task(task, LABELS))
    assert parsed.label_set == LABELS


def test_parse_rejects_foreign_prompt() -> None:
    with pytest.raises(PromptParseError):
        parse_prompt('What emotion is shown? Options: a, b')


@given(
    st.lists(
        st.text(alphabet='abcdefghij /-', min_size=1, max_size=12)
        .map(str.strip)
        .filter(bool),
        min_size=2,
        max_size=8,
        unique=True,
    )
)
def test_labels_survive_round_trip(labels: list[str]) -> None:
    parsed = parse_prompt(render_prompt(Quantifier.MULTIPLE, 'emotions', tuple(labels)))
    assert parsed.label_set == tuple(labels)
    assert parsed.quantifier is Quantifier.MULTIPLE


def test_cardinality_follows_quantifier_only() -> None:
    assert cardinality_for(Quantifier.MOST_PROMINENT, 3).maximum == 1
    assert cardinality_for(Quantifier.ONE_OR_SEVERAL, 3).minimum == 1
    assert cardinality_for(Quantifier.MULTIPLE, 3).minimum == 2
