from __future__ import annotations

import pytest

from emoharness.baselines.direct import (
    AUDIO_BLOCK_HEADER,
    DIRECT_AUDIO_TAG,
    DIRECT_TAG,
    HANDCRAFTED_TAG,
    direct_request,
    direct_with_audio_request,
    handcrafted_request,
    run_single_call,
)
from emoharness.common.constants import ANSWER_CLOSE, ANSWER_OPEN, STEP_TITLES
from emoharness.data.manifest import Example
from emoharness.harness.schema import AudioEvidence, PerceivedInputs
from emoharness.target.fake import FakeTarget


def _make_inputs(
    example: Example, transcript: str = 'I cannot believe this happened.'
) -> PerceivedInputs:
    return PerceivedInputs(
        example_id=example.example_id,
        video_path=example.media_path,
        duration_seconds=4.0,
        audio=AudioEvidence(
            speech_present=True,
            transcript=transcript,
            language='en',
            prosody=('raised pitch', 'fast speech rate'),
            description='a door slams in the background',
        ),
    )


def test_direct_request_contains_official_prompt_verbatim(
    benchmark_example: Example,
) -> None:
    request = direct_request(
        benchmark_example, video_path='videos/clip.mp4', max_tokens=512
    )
    assert request.prompt == benchmark_example.prompt
    assert request.video_path == 'videos/clip.mp4'
    assert request.max_tokens == 512
    # L0 never carries an audio block.
    assert AUDIO_BLOCK_HEADER not in request.prompt


def test_direct_with_audio_request_appends_delimited_audio_block(
    benchmark_example: Example,
) -> None:
    inputs = _make_inputs(benchmark_example)
    request = direct_with_audio_request(benchmark_example, inputs)
    assert request.prompt.startswith(benchmark_example.prompt)
    assert AUDIO_BLOCK_HEADER in request.prompt
    assert inputs.audio_text() in request.prompt
    assert request.prompt.index(benchmark_example.prompt) < request.prompt.index(
        AUDIO_BLOCK_HEADER
    )
    assert request.video_path == inputs.video_path


def test_handcrafted_request_has_official_prompt_audio_block_and_five_titled_steps(
    benchmark_example: Example,
) -> None:
    inputs = _make_inputs(benchmark_example)
    request = handcrafted_request(benchmark_example, inputs)
    assert benchmark_example.prompt in request.prompt
    assert AUDIO_BLOCK_HEADER in request.prompt
    assert inputs.audio_text() in request.prompt
    for index, title in enumerate(STEP_TITLES, start=1):
        assert f'Step {index}: {title}' in request.prompt
    assert ANSWER_OPEN in request.prompt
    assert ANSWER_CLOSE in request.prompt
    # No task name is named anywhere in the prompt, including the checklist wording.
    for task_name in ('ER_Wild', 'ML_ER', 'FG_ER', 'FG_SA', 'Noise_ER'):
        assert task_name not in request.prompt


def test_run_single_call_parses_answer(benchmark_example: Example) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = direct_request(benchmark_example, benchmark_example.media_path)
    record = run_single_call(target, request, benchmark_example, row='L0')
    assert record.labels == ('sadness',)
    assert record.parse_failures == 0
    assert record.target_calls == 1
    assert record.task == benchmark_example.task
    assert target.call_count == 1


def test_run_single_call_falls_back_to_first_candidate_on_parse_failure(
    benchmark_example: Example,
) -> None:
    target = FakeTarget(default_text='I decline to answer this question.')
    request = direct_request(benchmark_example, benchmark_example.media_path)
    record = run_single_call(target, request, benchmark_example, row='L0')
    assert record.labels == (benchmark_example.label_set[0],)
    assert record.parse_failures == 1


def test_run_single_call_uses_cost_fn(benchmark_example: Example) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = direct_request(benchmark_example, benchmark_example.media_path)
    record = run_single_call(
        target, request, benchmark_example, row='L0', cost_fn=lambda _response: 0.01
    )
    assert record.cost_usd == pytest.approx(0.01)


def test_every_single_call_request_names_the_row_that_issued_it(
    benchmark_example: Example,
) -> None:
    inputs = _make_inputs(benchmark_example)

    assert direct_request(benchmark_example, 'clip.mp4').tag == DIRECT_TAG
    assert direct_with_audio_request(benchmark_example, inputs).tag == DIRECT_AUDIO_TAG
    assert handcrafted_request(benchmark_example, inputs).tag == HANDCRAFTED_TAG
