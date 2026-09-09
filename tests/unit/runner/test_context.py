"""The composition root: what a ``--fake`` context wires each seam up to."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emoharness.builder.client import FakeBuilder
from emoharness.common.constants import NUM_STEPS
from emoharness.data.manifest import Example
from emoharness.harness.bundle_models import STAGE_NAMES
from emoharness.harness.loader import load_bundle
from emoharness.harness.pipeline import HarnessPipeline
from emoharness.runner.context import RunContext
from emoharness.runner.items import load_benchmark
from emoharness.runner.perception import perceived_inputs
from emoharness.target.base import TargetRequest
from emoharness.target.fake import FAKE_PRICING, FAKE_TEACHER_ID


@pytest.fixture
def example(ctx: RunContext) -> Example:
    return load_benchmark(ctx.paths)[0]


def test_a_fake_context_yields_a_target_that_answers_every_stage(
    ctx: RunContext, example: Example, repo_root: Path
) -> None:
    inputs = perceived_inputs(
        example, ctx.clip_cache(), ctx.preprocess, ctx.clip_preprocessor()
    )
    pipeline = HarnessPipeline(load_bundle(repo_root / 'harness'), ctx.target())

    output = pipeline.run(example, inputs)

    assert output.parse_failures == 0
    assert output.fallback_used is False
    assert tuple(trace.stage for trace in output.traces) == STAGE_NAMES
    assert set(output.labels) <= set(example.label_set)


def test_every_request_the_target_sees_carries_its_stage(
    ctx: RunContext, example: Example, repo_root: Path
) -> None:
    seen: list[TargetRequest] = []
    inner = ctx.target()

    class _Watched:
        model_id = inner.model_id

        def generate(self, request: TargetRequest) -> object:
            seen.append(request)
            return inner.generate(request)

    inputs = perceived_inputs(
        example, ctx.clip_cache(), ctx.preprocess, ctx.clip_preprocessor()
    )
    HarnessPipeline(load_bundle(repo_root / 'harness'), _Watched()).run(  # type: ignore[arg-type]
        example, inputs
    )

    assert [request.tag for request in seen] == list(STAGE_NAMES)


def test_a_single_call_row_is_answered_with_steps_and_an_answer_block(
    ctx: RunContext, example: Example
) -> None:
    response = ctx.target().generate(
        TargetRequest(
            prompt=example.prompt, tag='direct', video_path=example.media_path
        )
    )

    assert response.text.count('Step ') == NUM_STEPS
    assert response.text.endswith('</answer>')


def test_the_teacher_target_is_a_second_model(ctx: RunContext) -> None:
    assert ctx.teacher_target('matched_protocol').model_id == FAKE_TEACHER_ID
    assert ctx.target().model_id != FAKE_TEACHER_ID


def test_a_fake_run_never_asks_a_provider_who_the_teacher_is(
    ctx: RunContext,
) -> None:
    assert ctx.teacher_provider() == 'openai_compat'


def test_the_committed_teacher_config_is_the_openai_one(ctx: RunContext) -> None:
    config = ctx.teacher_model_config()
    assert config.provider == 'openai_compat'
    assert config.api_key_env == 'OPENAI_API_KEY'


def test_the_proxy_judge_names_its_provider_and_key(ctx: RunContext) -> None:
    config = ctx.proxy_judge()
    # The committed config scores the gate with GPT 5.6 Luna (no Gemini key on
    # the run box); ADR-0007's Gemini 3 Flash default is documented in the YAML.
    assert config.provider == 'openai'
    assert config.key_env() == 'OPENAI_API_KEY'
    assert config.reasoning_effort == 'none'


def test_a_fake_run_is_priced_from_the_nominal_list(ctx: RunContext) -> None:
    assert ctx.pricing() == FAKE_PRICING


def test_the_builder_client_is_the_fake_one(ctx: RunContext) -> None:
    assert isinstance(ctx.builder_client(2), FakeBuilder)


def test_the_step_scorer_answers_from_the_table_the_context_builds(
    ctx: RunContext,
) -> None:
    table = ctx.step_score_table({'a': (1, 2, NUM_STEPS)})

    assert table['a'].example_id == 'a'
    assert table['a'].recognition == pytest.approx(0.5)
    assert table['a'].num_steps == NUM_STEPS


def test_a_salted_table_disagrees_without_moving_the_last_step(
    ctx: RunContext,
) -> None:
    plain = ctx.step_score_table({'a': (1, 2, NUM_STEPS)})
    salted = ctx.step_score_table({'a': (1, 2, NUM_STEPS)}, '#proxy')

    assert salted['a'].recognition == plain['a'].recognition
    assert salted['a'].scores != plain['a'].scores


def test_the_clip_preprocessor_fills_the_shared_cache(
    ctx: RunContext, example: Example
) -> None:
    clip = ctx.clip_preprocessor()(example, ctx.clip_cache(), ctx.preprocess)

    assert clip.frame_paths
    assert json.loads(json.dumps(clip.model_dump(mode='json')))
