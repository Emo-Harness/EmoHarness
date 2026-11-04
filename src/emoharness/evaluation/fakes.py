"""The judge double: ratings a run can read back without paying a judge.

:class:`emoharness.evaluation.judge.FakeStepScorer` answers from a table; this
module writes that table. Ratings are built as the official ``<score>`` block
and parsed back with the official parser, so a stand-in score travels the same
path a real one does, multi-label C/K included.
"""

from __future__ import annotations

from collections.abc import Mapping

from emoharness.common.constants import NUM_STEPS
from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_text
from emoharness.evaluation.judge import FakeStepScorer
from emoharness.evaluation.official import parse_score_string, score_string
from emoharness.evaluation.schema import StepScores

#: One item's ingredients: labels right, labels asked for, steps in the response.
ScoreSpec = tuple[int, int, int]


def fake_score_block(
    example_id: str, correct: int, total: int, num_steps: int = NUM_STEPS
) -> str:
    """The ``<score>...</score>`` string the official judge writes for one item.

    Reasoning steps are 0/1 or 1/1, decided by the item id; the last step carries
    the official C/K, so multi-label items round-trip through the real parser.
    """
    digest = sha256_text(example_id)
    head = tuple(
        (int(digest[index], 16) % 2, 1) for index in range(max(0, num_steps - 1))
    )
    return score_string((*head, (correct, total)))


def fake_step_scores(
    example_id: str, correct: int, total: int, num_steps: int = NUM_STEPS
) -> StepScores:
    """Judge-shaped ratings for one item, parsed back with the official parser."""
    scores = parse_score_string(fake_score_block(example_id, correct, total, num_steps))
    if scores is None:  # pragma: no cover -- the block above always parses
        raise ValidationError(f'could not build fake scores for {example_id}')
    return StepScores(example_id=example_id, scores=scores)


def fake_score_table(
    items: Mapping[str, ScoreSpec], salt: str = ''
) -> Mapping[str, StepScores]:
    """Ratings for a whole item set, keyed by example id.

    ``salt`` changes the reasoning ratings without touching the last step, so two
    scorers over the same items disagree the way two judges would.
    """
    return {
        example_id: fake_step_scores(
            example_id + salt, correct, total, num_steps
        ).model_copy(update={'example_id': example_id})
        for example_id, (correct, total, num_steps) in items.items()
    }


def fake_step_scorer(scores: Mapping[str, StepScores]) -> FakeStepScorer:
    """A ``StepScorer`` answering from a pre-computed table, keyed by example id."""
    return FakeStepScorer(scores)
