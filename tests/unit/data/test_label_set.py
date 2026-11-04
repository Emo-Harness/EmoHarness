"""The Label set: normalization, canonical spelling, answer parsing, local scores."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import PromptParseError, ValidationError
from emoharness.data.label_set import (
    LabelSet,
    normalize_label,
    split_labels,
)
from emoharness.data.prompt_template import (
    Quantifier,
    parse_prompt,
    render_prompt,
)

SET = ('anger', 'Happiness', 'sadness', 'neutral')
SINGLE = LabelSet.from_labels(SET)
MULTI = LabelSet.from_labels(SET, Quantifier.MULTIPLE, 'emotions')

# The official ML-ER and FG-ER prompts' own worked examples.
ML_ER_PREDICTED = (
    'disappointment',
    'fear',
    'contempt',
    'disgust',
    'sadness',
    'anger',
    'anxiety',
    'happiness',
)
FG_ER_PREDICTED = (
    'disquietment',
    'disappointment',
    'fatigue',
    'sadness',
    'confusion',
    'aversion',
    'yearning',
)


class TestNormalization:
    def test_official_synonyms_only_for_scoring(self) -> None:
        assert normalize_label('  Happy ') == 'happiness'
        assert normalize_label('Angry') == 'anger'
        assert normalize_label('joy') == 'joy'

    def test_split_handles_both_separators(self) -> None:
        assert split_labels('anger,disgust') == ('anger', 'disgust')
        assert split_labels('Anticipation;Confidence') == (
            'Anticipation',
            'Confidence',
        )
        assert split_labels('  ,  ') == ()


class TestConstruction:
    def test_from_prompt_reads_the_grammar(self) -> None:
        prompt = render_prompt(Quantifier.ONE_OR_SEVERAL, 'emotions', SET)
        label_set = LabelSet.from_prompt(prompt)
        assert label_set.labels == SET
        assert label_set.quantifier is Quantifier.ONE_OR_SEVERAL
        assert label_set.noun == 'emotions'

    def test_from_parsed_agrees_with_from_prompt(self) -> None:
        prompt = render_prompt(Quantifier.MULTIPLE, 'emotions', SET)
        assert LabelSet.from_parsed(parse_prompt(prompt)) == LabelSet.from_prompt(
            prompt
        )

    def test_from_labels_defaults_to_the_single_label_wording(self) -> None:
        assert SINGLE.quantifier is Quantifier.MOST_PROMINENT
        assert SINGLE.noun == 'emotion'

    def test_a_prompt_that_is_not_official_is_rejected(self) -> None:
        with pytest.raises(PromptParseError):
            LabelSet.from_prompt('name the vibe')

    def test_fewer_than_two_candidates_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match='at least 2'):
            LabelSet.from_labels(('anger',))


class TestCardinalityAndMode:
    def test_is_multi_label_follows_the_quantifier_alone(self) -> None:
        assert SINGLE.is_multi_label is False
        assert MULTI.is_multi_label is True
        assert (
            LabelSet.from_labels(SET, Quantifier.ONE_OR_SEVERAL).is_multi_label is True
        )

    @pytest.mark.parametrize(
        ('quantifier', 'minimum', 'maximum'),
        [
            (Quantifier.MOST_PROMINENT, 1, 1),
            (Quantifier.ONE_OR_SEVERAL, 1, 3),
            (Quantifier.MULTIPLE, 2, 3),
        ],
    )
    def test_cardinality_delegates_to_the_grammar(
        self, quantifier: Quantifier, minimum: int, maximum: int
    ) -> None:
        cardinality = LabelSet.from_labels(SET, quantifier).cardinality(3)
        assert (cardinality.minimum, cardinality.maximum) == (minimum, maximum)


class TestCanonical:
    def test_canonical_copies_the_prompt_spelling(self) -> None:
        labels = LabelSet.from_labels(('Doubt/Confusion', 'neutral'))
        assert labels.canonical('doubt/confusion') == 'Doubt/Confusion'
        assert labels.canonical('NEUTRAL') == 'neutral'
        assert SINGLE.canonical('happy') == 'Happiness'
        assert labels.canonical('nothing like a candidate') is None

    def test_canonical_many_drops_unknowns_and_dedupes_in_order(self) -> None:
        found = SINGLE.canonical_many(('sad', 'boredom', 'sadness', 'Angry'))
        assert found == ('sadness', 'anger')

    def test_render_joins_the_way_the_prompt_does(self) -> None:
        assert SINGLE.render(('anger', 'sadness')) == 'anger, sadness'


class TestParseAnswer:
    def test_the_answer_block_wins(self) -> None:
        assert SINGLE.parse_answer('Step 1 ... <answer>happy</answer>') == (
            'Happiness',
        )
        assert SINGLE.parse_answer('<answer>sad, angry</answer>') == (
            'sadness',
            'anger',
        )

    def test_the_last_block_wins_when_there_are_several(self) -> None:
        text = '<answer>anger</answer> then <answer>neutral</answer>'
        assert SINGLE.parse_answer(text) == ('neutral',)

    def test_without_a_block_the_whole_text_is_scanned(self) -> None:
        assert SINGLE.parse_answer('I think the person is neutral.') == ('neutral',)

    def test_an_unusable_block_falls_back_to_the_whole_text(self) -> None:
        assert SINGLE.parse_answer('clearly anger. <answer>???</answer>') == ('anger',)

    def test_nothing_recognizable_yields_nothing(self) -> None:
        assert SINGLE.parse_answer('<answer>unknown</answer>') == ()
        assert SINGLE.parse_answer('') == ()

    @given(
        chosen=st.lists(st.sampled_from(SET), min_size=1, max_size=4, unique=True).map(
            tuple
        )
    )
    def test_a_rendered_answer_round_trips(self, chosen: tuple[str, ...]) -> None:
        text = f'Step 5: Decision.\n<answer>{MULTI.render(chosen)}</answer>'
        assert MULTI.parse_answer(text) == chosen


class TestRecognition:
    def test_single_label_matches_any_gold_label(self) -> None:
        assert SINGLE.recognition(('sadness',), ('sadness',)) == 1.0
        assert SINGLE.recognition(('happy',), ('Happiness',)) == 1.0
        assert SINGLE.recognition(('anger',), ('sadness', 'anger')) == 1.0
        assert SINGLE.recognition((), ('sadness',)) == 0.0
        assert SINGLE.recognition(('joy',), ('sadness',)) == 0.0

    def test_single_label_only_considers_the_first_predicted(self) -> None:
        # 'anger' is not gold, but it is listed first, so the item scores 0 even
        # though 'sadness' (gold) appears later in the predicted tuple.
        assert SINGLE.recognition(('anger', 'sadness'), ('sadness',)) == 0.0

    def test_gold_is_required(self) -> None:
        with pytest.raises(ValidationError):
            SINGLE.recognition(('anger',), ())

    def test_official_ck_eight_predicted_versus_three_gold(self) -> None:
        assert len(ML_ER_PREDICTED) == 8
        gold = ('fear', 'sadness', 'anxiety')
        assert MULTI.recognition(ML_ER_PREDICTED, gold) == 1.0

    def test_official_ck_seven_predicted_versus_one_gold(self) -> None:
        assert len(FG_ER_PREDICTED) == 7
        assert MULTI.recognition(FG_ER_PREDICTED, ('sadness',)) == 1.0

    def test_official_ck_neutral_versus_two_gold_is_zero(self) -> None:
        assert MULTI.recognition(('neutral',), ('anger', 'fear')) == 0.0

    @pytest.mark.parametrize(
        ('predicted', 'gold', 'expected'),
        [
            (('happy',), ('happy', 'anger', 'disgust'), 1 / 3),
            (
                ('happy', 'disgust', 'sad'),
                ('sad', 'anger', 'surprise', 'disgust'),
                2 / 4,
            ),
        ],
    )
    def test_official_ck_more_examples(
        self, predicted: tuple[str, ...], gold: tuple[str, ...], expected: float
    ) -> None:
        assert MULTI.recognition(predicted, gold) == pytest.approx(expected)

    def test_overlap_counts_distinct_normalized_labels(self) -> None:
        assert MULTI.overlap(('sad', 'sadness'), ('sadness', 'anger')) == 1
        assert MULTI.overlap((), ('anger',)) == 0


_POOL = ('a', 'b', 'c', 'd', 'e')
_predicted = st.lists(st.sampled_from(_POOL), min_size=0, max_size=5).map(tuple)
_gold = st.lists(st.sampled_from(_POOL), min_size=1, max_size=5, unique=True).map(tuple)


@given(predicted=_predicted, gold=_gold)
def test_recognition_always_lands_in_the_unit_interval(
    predicted: tuple[str, ...], gold: tuple[str, ...]
) -> None:
    labels = LabelSet.from_labels(_POOL)
    multi = LabelSet.from_labels(_POOL, Quantifier.MULTIPLE, 'emotions')
    assert 0.0 <= multi.recognition(predicted, gold) <= 1.0
    assert labels.recognition(predicted, gold) in (0.0, 1.0)


class TestPRF:
    def test_hand_computed(self) -> None:
        perfect = MULTI.prf(('anger', 'disgust'), ('anger', 'disgust'))
        assert perfect.precision == perfect.recall == perfect.f1 == 1.0

        partial = MULTI.prf(('anger',), ('anger', 'disgust'))
        assert partial.precision == 1.0
        assert partial.recall == 0.5
        assert partial.f1 == pytest.approx(2 / 3)

        miss = MULTI.prf(('happy', 'sad'), ('anger',))
        assert (miss.precision, miss.recall, miss.f1) == (0.0, 0.0, 0.0)

        empty = MULTI.prf((), ('anger',))
        assert (empty.precision, empty.recall) == (0.0, 0.0)

    def test_normalizes_before_comparing(self) -> None:
        assert MULTI.prf(('Angry',), ('anger',)).f1 == 1.0
