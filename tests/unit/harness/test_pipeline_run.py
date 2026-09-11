"""One item through the harness: the calls it costs and the answer it emits."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from emoharness.common.constants import (
    CALL_BUDGET_EXPECTED,
    CALL_BUDGET_HARD_CAP,
    NUM_STEPS,
    STEP_TITLES,
)
from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.pipeline import PipelineConfig
from emoharness.harness.schema import PerceivedInputs
from emoharness.target.fake import FakeTarget

from .conftest import (
    HYPOTHESES_JSON,
    SYNTHESIZE_JSON,
    cited_ids,
    run_once,
    scripted_responder,
)


def test_the_happy_path_costs_four_calls_and_five_grounded_steps(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    output = run_once(bundle, target, benchmark_example, inputs)

    assert target.call_count == CALL_BUDGET_EXPECTED
    assert output.target_calls == CALL_BUDGET_EXPECTED
    assert output.parse_failures == 0
    assert output.fallback_used is False
    assert output.example_id == benchmark_example.example_id
    assert len(output.steps) == NUM_STEPS
    for title, step in zip(STEP_TITLES[: NUM_STEPS - 1], output.steps, strict=False):
        assert step.startswith(f'{title}: ')
    assert (
        output.steps[-1] == 'Based on the analysis, the predicted emotion is sadness.'
    )
    assert output.labels == ('sadness',)
    assert set(output.labels) <= set(benchmark_example.label_set)
    assert output.response_text.endswith('<answer>sadness</answer>')
    assert cited_ids(output)
    assert cited_ids(output) <= output.evidence.item_ids()
    assert SYNTHESIZE_JSON['justification'] in output.steps[3]


def test_every_stage_sees_its_own_prompt(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    output = run_once(bundle, target, benchmark_example, inputs)

    assert [request.tag for request in target.requests] == list(bundle.routing.stages)
    assert tuple(trace.stage for trace in output.traces) == tuple(bundle.routing.stages)
    visual, hypotheses, verify, synthesize = target.requests
    assert visual.video_path == inputs.video_path
    assert visual.json_schema is not None
    assert hypotheses.video_path is None
    assert 'v1' in hypotheses.prompt
    assert inputs.audio.transcript in verify.prompt
    assert STEP_TITLES[0] in synthesize.prompt


def test_a_parse_failure_costs_one_retry(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    target = FakeTarget(
        responder=scripted_responder(fail_stages=frozenset({'generate_hypotheses'}))
    )

    output = run_once(bundle, target, benchmark_example, inputs)

    assert target.call_count == CALL_BUDGET_EXPECTED + 1
    assert output.target_calls == CALL_BUDGET_EXPECTED + 1
    assert output.parse_failures == 1
    assert output.fallback_used is False
    assert output.labels == ('sadness',)


def test_a_fenced_answer_still_parses(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    fenced = f'Here you go:\n```json\n{json.dumps(HYPOTHESES_JSON)}\n```\nThanks.'
    target = FakeTarget(responder=scripted_responder({'generate_hypotheses': fenced}))

    output = run_once(bundle, target, benchmark_example, inputs)

    assert output.parse_failures == 0
    assert output.labels == ('sadness',)


def test_a_target_that_never_answers_json_stays_inside_the_hard_cap(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    target = FakeTarget(default_text='never JSON')

    output = run_once(bundle, target, benchmark_example, inputs)

    assert target.call_count == CALL_BUDGET_HARD_CAP
    assert output.target_calls == CALL_BUDGET_HARD_CAP
    assert output.parse_failures == CALL_BUDGET_HARD_CAP
    assert output.fallback_used is True
    assert len(output.steps) == NUM_STEPS
    assert output.labels
    assert set(output.labels) <= set(benchmark_example.label_set)


def test_budget_exhaustion_skips_the_remaining_stages(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    target = FakeTarget(responder=scripted_responder())
    config = PipelineConfig(call_hard_cap=2, retries_per_stage=0)

    output = run_once(bundle, target, benchmark_example, inputs, config)

    assert target.call_count == 2
    assert {trace.stage: trace.calls for trace in output.traces} == {
        'extract_visual': 1,
        'generate_hypotheses': 1,
        'verify': 0,
        'synthesize': 0,
    }
    assert output.fallback_used is True
    assert output.labels == ('sadness',)
    assert all('No reliable cue observed.' in step for step in output.steps[:-1])


def test_the_response_text_is_what_the_judge_reads(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    output = run_once(bundle, target, benchmark_example, inputs)

    lines = output.response_text.splitlines()
    assert len(lines) == NUM_STEPS + 1
    for number, (line, step) in enumerate(zip(lines, output.steps, strict=False), 1):
        assert line == f'Step {number}: {step}'
    assert json.loads(json.dumps(output.model_dump(mode='json')))


def test_inputs_must_belong_to_the_example(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    multilabel_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    with pytest.raises(ValidationError, match='do not belong'):
        run_once(bundle, target, benchmark_example, make_inputs(multilabel_example))


def test_a_stage_removed_from_routing_is_never_called(
    edited_bundle: Callable[..., HarnessBundle],
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    bundle = edited_bundle(
        'routed',
        {
            'routing.yaml': 'stages: [extract_visual, generate_hypotheses, synthesize]\n'
            'verification:\n  agreement_check: cross_modal\n'
            '  ranker: alternative_hypotheses\n'
        },
    )

    output = run_once(bundle, target, benchmark_example, inputs)

    assert target.call_count == 3
    assert {trace.stage: trace.calls for trace in output.traces}['verify'] == 0
    assert len(output.steps) == NUM_STEPS


def test_an_unfilled_placeholder_renders_as_nothing(
    edited_bundle: Callable[..., HarnessBundle],
    repo_root: Path,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    source = (repo_root / 'harness' / 'prompts' / 'extract_visual.md').read_text(
        encoding='utf-8'
    )
    bundle = edited_bundle(
        'placeheld',
        {
            'prompts/extract_visual.md': f'{source}\nRecord so far: [{{evidence_json}}]\n'
        },
    )

    output = run_once(bundle, target, benchmark_example, inputs)

    assert target.requests[0].prompt.rstrip().endswith('Record so far: []')
    assert output.labels == ('sadness',)
