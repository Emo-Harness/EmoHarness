"""What the harness is allowed to answer: cardinality, the evidence gate, spelling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emoharness.common.constants import MULTILABEL_CAP, NUM_STEPS
from emoharness.data.manifest import Example
from emoharness.data.prompt_template import Quantifier, cardinality_for
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.schema import HarnessOutput, PerceivedInputs
from emoharness.target.fake import FakeTarget

from .conftest import reading, reading_script, run_once, scripted_responder

_LABEL_POOL: Final[tuple[str, ...]] = (
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happiness',
    'neutral',
    'sadness',
    'surprise',
)
_KNOWN_IDS: Final[tuple[str, ...]] = ('v1', 'v2', 'v3', 'a1', 'a2', 'o1', 't1')


def _run_with(
    bundle: HarnessBundle,
    example: Example,
    inputs: PerceivedInputs,
    rows: Sequence[Any],
) -> HarnessOutput:
    """One item whose two reading stages both answer with ``rows``."""
    target = FakeTarget(responder=scripted_responder(reading_script(rows)))
    return run_once(bundle, target, example, inputs)


def test_multiple_emits_at_least_two_and_drops_the_unsupported(
    bundle: HarnessBundle,
    multilabel_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    rows = [
        reading('anger', support=('v1', 'a1'), confidence=0.9),
        reading('disgust', support=('v2',), confidence=0.7),
        reading('fear', support=('zz',), confidence=0.6),
    ]

    output = _run_with(
        bundle, multilabel_example, make_inputs(multilabel_example), rows
    )

    assert output.labels == ('anger', 'disgust')
    assert output.steps[-1].endswith('the predicted emotions are anger, disgust.')
    assert output.response_text.endswith('<answer>anger, disgust</answer>')


def test_multiple_fills_from_the_ranking_when_too_few_pass_the_gate(
    bundle: HarnessBundle,
    multilabel_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    rows = [
        reading('anger', support=('v1',), confidence=0.9),
        reading('disgust', support=('zz',), confidence=0.8),
        reading('fear', support=('zz',), confidence=0.1),
    ]

    output = _run_with(
        bundle, multilabel_example, make_inputs(multilabel_example), rows
    )

    assert output.labels == ('anger', 'disgust')


def test_multiple_stops_at_the_cap(
    bundle: HarnessBundle,
    multilabel_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    rows = [
        reading(label, support=('v1', 'v2'), confidence=0.9 - index / 10)
        for index, label in enumerate(multilabel_example.label_set)
    ]

    output = _run_with(
        bundle, multilabel_example, make_inputs(multilabel_example), rows
    )

    assert len(output.labels) == MULTILABEL_CAP
    assert output.labels == multilabel_example.label_set[:MULTILABEL_CAP]


def test_one_or_several_emits_one_when_only_one_is_supported(
    bundle: HarnessBundle,
    make_example: Callable[..., Example],
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    example = make_example(Quantifier.ONE_OR_SEVERAL)
    rows = [
        reading('sadness', support=('v1', 'a1'), confidence=0.9),
        reading('anger', support=('zz',), confidence=0.8),
    ]

    output = _run_with(bundle, example, make_inputs(example), rows)

    assert output.labels == ('sadness',)
    assert output.steps[-1] == (
        'Based on the analysis, the predicted emotion is sadness.'
    )


def test_one_or_several_stops_at_the_cap(
    bundle: HarnessBundle,
    make_example: Callable[..., Example],
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    example = make_example(Quantifier.ONE_OR_SEVERAL)
    rows = [
        reading(label, support=('v1',), confidence=0.9 - index / 10)
        for index, label in enumerate(example.label_set)
    ]

    output = _run_with(bundle, example, make_inputs(example), rows)

    assert len(output.labels) == MULTILABEL_CAP
    assert output.labels == example.label_set[:MULTILABEL_CAP]


def test_the_most_prominent_emits_exactly_one(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    rows = [
        reading('sadness', support=('v1', 'a1'), confidence=0.9),
        reading('anger', support=('v2', 'a2'), confidence=0.8),
    ]

    output = _run_with(bundle, benchmark_example, inputs, rows)

    assert output.labels == ('sadness',)


def test_a_label_outside_the_candidate_list_is_dropped(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    rows = [
        reading('boredom', support=('v1', 'a1'), confidence=0.99),
        reading('sadness', support=('a1',), confidence=0.1),
    ]

    output = _run_with(bundle, benchmark_example, inputs, rows)

    assert output.labels == ('sadness',)
    assert all(item.label != 'boredom' for item in output.hypotheses.hypotheses)


def test_when_no_label_survives_the_uniform_fallback_answers(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    rows = [
        reading('boredom', support=('v1',), confidence=0.9),
        'not even a row',
    ]

    output = _run_with(bundle, benchmark_example, inputs, rows)

    assert tuple(item.label for item in output.hypotheses.hypotheses) == (
        benchmark_example.label_set
    )
    assert output.labels == (benchmark_example.label_set[0],)


def test_a_reading_stage_that_answers_with_the_wrong_shape_falls_back(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    script = {
        'generate_hypotheses': {'coarse': 'x', 'hypotheses': 'sadness'},
        'verify': {'modalities_agree': True, 'hypotheses': 'sadness'},
    }
    target = FakeTarget(responder=scripted_responder(script))

    output = run_once(bundle, target, benchmark_example, inputs)

    assert tuple(item.label for item in output.hypotheses.hypotheses) == (
        benchmark_example.label_set
    )
    assert output.labels == (benchmark_example.label_set[0],)


def test_an_unreadable_confidence_becomes_the_default(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    rows = [
        {'label': 'sadness', 'support': ['v1'], 'confidence': 'not a number'},
        reading('anger', support=('v2',), confidence=0.4),
    ]

    output = _run_with(bundle, benchmark_example, inputs, rows)

    assert output.hypotheses.hypotheses[0].confidence == 0.5
    assert output.labels == ('sadness',)


def test_a_label_is_emitted_with_the_candidate_lists_own_spelling(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    rows = [reading('HAPPY', support=('v1', 'a1'), confidence=0.9)]

    output = _run_with(bundle, benchmark_example, inputs, rows)

    assert output.labels == ('happiness',)
    assert output.response_text.endswith('<answer>happiness</answer>')


@st.composite
def _cases(draw: st.DrawFn) -> tuple[Quantifier, tuple[str, ...], list[dict[str, Any]]]:
    labels = tuple(
        draw(
            st.lists(st.sampled_from(_LABEL_POOL), min_size=2, max_size=8, unique=True)
        )
    )
    ids = st.sampled_from((*_KNOWN_IDS, 'zz'))
    rows = draw(
        st.lists(
            st.builds(
                reading,
                label=st.sampled_from((*labels, 'not a candidate', 'HAPPY')),
                support=st.lists(ids, max_size=3),
                contradiction=st.lists(ids, max_size=3),
                confidence=st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=1,
            max_size=6,
        )
    )
    return draw(st.sampled_from(list(Quantifier))), labels, rows


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=_cases())
def test_run_always_answers_inside_the_rules(
    bundle: HarnessBundle,
    make_example: Callable[..., Example],
    make_inputs: Callable[..., PerceivedInputs],
    case: tuple[Quantifier, tuple[str, ...], list[dict[str, Any]]],
) -> None:
    quantifier, labels, rows = case
    example = make_example(quantifier, labels)

    output = _run_with(bundle, example, make_inputs(example), rows)

    cardinality = cardinality_for(quantifier, MULTILABEL_CAP)
    assert output.labels
    assert len(output.labels) == len(set(output.labels))
    assert set(output.labels) <= set(labels)
    assert len(output.labels) <= MULTILABEL_CAP
    assert cardinality.minimum <= len(output.labels)
    assert len(output.labels) <= (cardinality.maximum or MULTILABEL_CAP)
    assert len(output.steps) == NUM_STEPS
    assert output.response_text.endswith('</answer>')
