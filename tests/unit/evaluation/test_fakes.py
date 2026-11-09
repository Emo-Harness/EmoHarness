"""The judge double: ratings shaped like the official judge's, and a table of them."""

from __future__ import annotations

import pytest

from emoharness.common.errors import ValidationError
from emoharness.evaluation.fakes import (
    fake_score_block,
    fake_score_table,
    fake_step_scorer,
    fake_step_scores,
)
from emoharness.evaluation.judge import judge_input_for
from emoharness.evaluation.official import parse_score_string


class TestFakeScoreBlock:
    def test_the_block_parses_with_the_official_parser(self) -> None:
        block = fake_score_block('item', correct=2, total=3, num_steps=5)

        scores = parse_score_string(block)

        assert scores is not None
        assert len(scores) == 5
        assert scores[-1] == pytest.approx(2 / 3)

    def test_a_one_step_response_carries_only_the_official_ratio(self) -> None:
        assert (
            fake_score_block('item', 1, 2, num_steps=1) == '<score>Step 1: 1/2</score>'
        )


class TestFakeStepScores:
    def test_the_last_step_carries_the_official_ratio(self) -> None:
        scores = fake_step_scores('item', correct=2, total=3, num_steps=5)

        assert scores.recognition == pytest.approx(2 / 3)
        assert scores.num_steps == 5

    def test_reasoning_steps_are_zero_or_one(self) -> None:
        scores = fake_step_scores('item', 1, 1)

        assert set(scores.scores[:-1]) <= {0.0, 1.0}

    def test_the_same_id_always_scores_the_same(self) -> None:
        assert fake_step_scores('a', 1, 1) == fake_step_scores('a', 1, 1)


class TestFakeScoreTable:
    def test_the_table_is_keyed_by_example_id(self) -> None:
        table = fake_score_table({'a': (1, 1, 5), 'b': (0, 2, 3)})

        assert set(table) == {'a', 'b'}
        assert table['a'].example_id == 'a'
        assert table['b'].num_steps == 3

    def test_a_salt_moves_the_reasoning_steps_but_keeps_the_id(self) -> None:
        plain = fake_score_table({'a': (1, 2, 5)})
        salted = fake_score_table({'a': (1, 2, 5)}, '#proxy')

        assert salted['a'].example_id == 'a'
        assert salted['a'].recognition == plain['a'].recognition
        assert salted['a'].scores != plain['a'].scores


class TestFakeStepScorer:
    def test_the_scorer_answers_from_the_table(self) -> None:
        table = fake_score_table({'a': (1, 1, 5)})
        scorer = fake_step_scorer(table)

        scored = scorer.score(judge_input_for('a', (), ('anger',), 'Step 1: anything'))

        assert scored == table['a']

    def test_an_unknown_id_is_refused(self) -> None:
        scorer = fake_step_scorer(fake_score_table({'a': (1, 1, 5)}))

        with pytest.raises(ValidationError, match='no fake score'):
            scorer.score(judge_input_for('b', (), ('anger',), 'Step 1: anything'))
