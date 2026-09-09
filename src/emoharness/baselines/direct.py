"""L0-L2 single-call baselines: Direct, Direct+audio, and the frozen handcrafted prompt.

Every row here issues exactly one Target call with the official prompt or a
strict extension of it (ADR-0009's ladder). L2's prompt file is written once,
before any benchmark run, and frozen at pre-registration (ADR-0009).
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Callable
from typing import Final

from emoharness.common.constants import ANSWER_CLOSE, ANSWER_OPEN, STEP_TITLES
from emoharness.common.errors import TargetError, TargetUnavailableError
from emoharness.common.logging import get_logger
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.evaluation.schema import PredictionRecord
from emoharness.harness.schema import PerceivedInputs
from emoharness.target.base import TargetModel, TargetRequest, TargetResponse

AUDIO_BLOCK_HEADER: Final = 'Audio evidence (from external tools):\n'

#: The tag every request from this module carries, one per single-call method.
DIRECT_TAG: Final = 'direct'
DIRECT_AUDIO_TAG: Final = 'direct_audio'
HANDCRAFTED_TAG: Final = 'handcrafted'

_HANDCRAFTED_PACKAGE: Final = 'emoharness.baselines'
_HANDCRAFTED_RESOURCE: Final = 'handcrafted_prompt.md'


_LOGGER = get_logger(__name__)


def direct_request(
    example: Example, video_path: str, max_tokens: int = 1024
) -> TargetRequest:
    """L0 Qwen Direct: the official prompt verbatim, native video, one call."""
    return TargetRequest(
        prompt=example.prompt,
        tag=DIRECT_TAG,
        video_path=video_path,
        max_tokens=max_tokens,
    )


def direct_with_audio_request(
    example: Example, inputs: PerceivedInputs
) -> TargetRequest:
    """L1: L0 plus the observable-cue audio text appended to the official prompt."""
    prompt = f'{example.prompt}\n\n{AUDIO_BLOCK_HEADER}{inputs.audio_text()}'
    return TargetRequest(
        prompt=prompt, tag=DIRECT_AUDIO_TAG, video_path=inputs.video_path
    )


def handcrafted_request(example: Example, inputs: PerceivedInputs) -> TargetRequest:
    """L2: L1 plus a frozen human-written cue checklist and five-step format instruction."""
    prompt = _load_handcrafted_template().format_map(
        _handcrafted_fields(example, inputs)
    )
    return TargetRequest(
        prompt=prompt, tag=HANDCRAFTED_TAG, video_path=inputs.video_path
    )


def _handcrafted_fields(example: Example, inputs: PerceivedInputs) -> dict[str, str]:
    return {
        'official_prompt': example.prompt,
        'audio_block_header': AUDIO_BLOCK_HEADER,
        'audio_text': inputs.audio_text(),
        'step_1': STEP_TITLES[0],
        'step_2': STEP_TITLES[1],
        'step_3': STEP_TITLES[2],
        'step_4': STEP_TITLES[3],
        'step_5': STEP_TITLES[4],
        'answer_open': ANSWER_OPEN,
        'answer_close': ANSWER_CLOSE,
    }


def _load_handcrafted_template() -> str:
    resource = importlib.resources.files(_HANDCRAFTED_PACKAGE).joinpath(
        _HANDCRAFTED_RESOURCE
    )
    return resource.read_text(encoding='utf-8')


def run_single_call(
    target: TargetModel,
    request: TargetRequest,
    example: Example,
    row: str,
    cost_fn: Callable[[TargetResponse], float] = lambda _response: 0.0,
) -> PredictionRecord:
    """Issue one Target call and parse the answer; used by every single-call baseline row.

    When nothing in the response parses against `example.label_set`, the first
    candidate label is emitted and one parse failure is counted, so every row
    always produces a valid `PredictionRecord`.
    """
    try:
        response = target.generate(request)
    except TargetUnavailableError:
        raise
    except TargetError as exc:
        # An undecodable clip is one wrong answer, not a dead row.
        _LOGGER.warning('%s %s: target call failed: %s', row, example.example_id, exc)
        response = TargetResponse(
            text='',
            model_id=target.model_id,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
        )
    labels = LabelSet.from_prompt(example.prompt).parse_answer(response.text)
    parse_failures = 0
    if not labels:
        labels = (example.label_set[0],)
        parse_failures = 1
    return PredictionRecord(
        example_id=example.example_id,
        row=row,
        task=example.task,
        labels=labels,
        response_text=response.text,
        target_calls=1,
        parse_failures=parse_failures,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_s=response.latency_s,
        cost_usd=cost_fn(response),
    )
