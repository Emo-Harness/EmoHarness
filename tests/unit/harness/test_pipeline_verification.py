"""Cross-modal agreement, and what happens when Builder-written code misbehaves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

from emoharness.data.manifest import Example
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.schema import HarnessOutput, PerceivedInputs
from emoharness.target.fake import FakeTarget

from .conftest import reading, reading_script, run_once, scripted_responder

_HEADER: Final = 'from __future__ import annotations\n\nfrom typing import Any\n\n\n'

FAILING_AGREEMENT: Final = _HEADER + (
    'def modalities_agree(record: dict[str, Any], hypotheses: dict[str, Any]) -> bool:\n'
    '    return 1 / 0\n'
)
FAILING_RANKER: Final = _HEADER + (
    'def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[Any]:\n'
    "    return hypotheses['a key that is not there']\n"
)
USELESS_RANKER: Final = _HEADER + (
    'def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[Any]:\n'
    '    return [1, 2, 3]\n'
)
INVENTING_RANKER: Final = _HEADER + (
    'def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[Any]:\n'
    '    rows = []\n'
    "    for hypothesis in list(hypotheses['hypotheses']):\n"
    '        rows = rows + [hypothesis]\n'
    "    invented = {'label': 'not a candidate', 'support': ['v1'], 'confidence': 0.99}\n"
    '    return [invented] + rows\n'
)

#: Seen cues favour sadness, heard cues favour anger.
_SPLIT_ROWS: Final[tuple[dict[str, Any], ...]] = (
    reading('sadness', support=('v1', 'v2'), confidence=0.8),
    reading('anger', support=('a1', 'a2'), confidence=0.7),
)
#: Both sides favour sadness.
_AGREEING_ROWS: Final[tuple[dict[str, Any], ...]] = (
    reading('sadness', support=('v1', 'a1'), confidence=0.8),
    reading('anger', support=(), confidence=0.3),
)


def _run(
    bundle: HarnessBundle,
    example: Example,
    inputs: PerceivedInputs,
    rows: Sequence[dict[str, Any]],
    agree: bool = True,
) -> HarnessOutput:
    target = FakeTarget(responder=scripted_responder(reading_script(rows, agree=agree)))
    return run_once(bundle, target, example, inputs)


def test_agreement_is_true_when_both_sides_favour_one_label(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    output = _run(bundle, benchmark_example, inputs, _AGREEING_ROWS, agree=True)

    assert output.verification.modalities_agree is True
    assert output.verification.notes == ''
    assert output.labels == ('sadness',)


def test_agreement_is_false_when_the_sides_point_apart(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=False)

    assert output.verification.modalities_agree is False
    assert output.verification.notes == ''


def test_a_target_that_disagrees_with_the_check_is_recorded(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=True)

    assert output.verification.modalities_agree is False
    assert 'target reported modalities_agree=True' in output.verification.notes


def test_a_failing_agreement_check_falls_back_to_the_built_in_rule(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'broken_agreement', {'verification/cross_modal.py': FAILING_AGREEMENT}
    )

    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=False)

    assert 'agreement check fell back to the built-in rule' in output.verification.notes
    assert output.verification.modalities_agree is False
    assert output.labels == ('sadness',)


def test_the_built_in_rule_agrees_when_one_side_is_silent(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'silent_side', {'verification/cross_modal.py': FAILING_AGREEMENT}
    )
    rows = (reading('sadness', support=('v1', 'zz'), confidence=0.8),)

    output = _run(bundle, benchmark_example, inputs, rows, agree=True)

    assert 'fell back' in output.verification.notes
    assert output.verification.modalities_agree is True


def test_a_failing_ranker_falls_back_to_confidence_order(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'broken_ranker', {'verification/alternative_hypotheses.py': FAILING_RANKER}
    )

    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=False)

    assert 'ranker fell back' in output.verification.notes
    assert tuple(item.label for item in output.verification.ranked) == (
        'sadness',
        'anger',
    )
    assert output.labels == ('sadness',)


def test_a_ranker_that_returns_nothing_usable_falls_back(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'useless_ranker', {'verification/alternative_hypotheses.py': USELESS_RANKER}
    )

    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=False)

    assert 'ranker returned no usable rows' in output.verification.notes
    assert output.labels == ('sadness',)


def test_a_ranker_cannot_smuggle_in_a_label_or_an_id(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'inventive_ranker', {'verification/alternative_hypotheses.py': INVENTING_RANKER}
    )
    rows = (
        reading('sadness', support=('v1', 'zz'), confidence=0.8),
        reading('anger', support=('a1',), contradiction=('zz',), confidence=0.7),
    )

    output = _run(bundle, benchmark_example, inputs, rows, agree=False)

    ranked = output.verification.ranked
    assert tuple(item.label for item in ranked) == ('sadness', 'anger')
    assert ranked[0].support == ('v1',)
    assert ranked[1].contradiction == ()
    assert output.verification.notes == ''
    assert output.labels == ('sadness',)


REVERSE_RANKER = _HEADER + (
    'def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[Any]:\n'
    "    return list(hypotheses['hypotheses'])[::-1]\n"
)


def test_code_ranker_controls_labels_even_when_confidence_disagrees(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'reverse', {'verification/alternative_hypotheses.py': REVERSE_RANKER}
    )
    output = _run(bundle, benchmark_example, inputs, _SPLIT_ROWS, agree=False)
    assert output.labels == ('anger',)
    assert (
        output.verification.ranked[0].confidence
        < output.verification.ranked[1].confidence
    )


def test_question_and_duplicate_ids_cannot_manufacture_support(
    bundle: HarnessBundle,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    rows = (
        reading(
            'anger', support=('t1', 'v1', 'v1'), contradiction=('a1',), confidence=0.99
        ),
        reading('sadness', support=('a2',), confidence=0.2),
    )
    output = _run(bundle, benchmark_example, inputs, rows)
    assert output.verification.ranked[0].support == ('v1',)
    assert output.labels == ('sadness',)


def test_conflict_routing_skips_only_agreeing_student_verification(
    bundle: HarnessBundle,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    routed = bundle.model_copy(
        update={'routing': bundle.routing.model_copy(update={'verify_on': 'conflict'})}
    )
    target = FakeTarget(responder=scripted_responder(reading_script(_AGREEING_ROWS)))
    output = run_once(routed, target, benchmark_example, inputs)
    assert output.target_calls == 3
    assert output.fallback_used is False
    assert 'verify' not in [request.tag for request in target.requests]
    assert '"modalities_agree": true' in target.requests[-1].prompt

    target = FakeTarget(
        responder=scripted_responder(reading_script(_SPLIT_ROWS, agree=False))
    )
    output = run_once(routed, target, benchmark_example, inputs)
    assert output.target_calls == 4
    assert '"modalities_agree": false' in target.requests[2].prompt


def test_malformed_agreement_cannot_skip_student_review(
    edited_bundle: Callable[..., HarnessBundle],
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    source = _HEADER + (
        'def modalities_agree(record: dict[str, Any], hypotheses: dict[str, Any]) -> bool:\n'
        "    return 'false'\n"
    )
    bundle = edited_bundle('bad_bool', {'verification/cross_modal.py': source})
    bundle = bundle.model_copy(
        update={'routing': bundle.routing.model_copy(update={'verify_on': 'conflict'})}
    )
    target = FakeTarget(responder=scripted_responder(reading_script(_AGREEING_ROWS)))
    output = run_once(bundle, target, benchmark_example, inputs)
    assert output.target_calls == 4
    assert 'must return a bool' in output.verification.notes


def test_failed_hypothesis_generation_cannot_skip_student_review(
    bundle: HarnessBundle,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = bundle.model_copy(
        update={'routing': bundle.routing.model_copy(update={'verify_on': 'conflict'})}
    )
    target = FakeTarget(
        responder=scripted_responder({'generate_hypotheses': 'invalid JSON'})
    )
    output = run_once(bundle, target, benchmark_example, inputs)
    assert 'verify' in [request.tag for request in target.requests]
    assert output.fallback_used is True
