"""Fixtures for the harness: the real Stage 1 bundle and a scripted Target.

The harness is exercised through ``HarnessPipeline.run`` alone. The Target is a
``FakeTarget`` driven by the harness's own
:func:`emoharness.harness.scripted_responder`, which reads the stage off
``TargetRequest.tag``; this module only supplies the answers this suite asserts
on. Scripts are plain JSON-shaped dicts, so a scenario says what the model
answered and asserts what came out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import Quantifier, render_prompt_for_task
from emoharness.data.tasks import Split, Task
from emoharness.harness.fake_responder import ChooseLabel
from emoharness.harness.fake_responder import scripted_responder as harness_responder
from emoharness.harness.loader import HarnessBundle, load_bundle, write_bundle_copy
from emoharness.harness.pipeline import HarnessPipeline, PipelineConfig
from emoharness.harness.schema import (
    AudioEvidence,
    HarnessOutput,
    PerceivedInputs,
)
from emoharness.target.fake import FakeTarget, Responder

#: A phrase that appears in exactly one Stage 1 prompt; ``test_loader`` asserts
#: that each one still marks exactly one prompt, so a scenario can edit by name.
STAGE_MARKERS: Mapping[str, str] = {
    'extract_visual': 'Describe only what can be seen',
    'generate_hypotheses': 'How to read the record',
    'verify': 'WHAT WAS HEARD',
    'synthesize': 'RANKED READINGS',
}

VISUAL_JSON = {
    'face': ['tearful eyes', 'downward gaze'],
    'body': ['closed posture'],
    'scene': ['conversation after bad news'],
    'temporal_change': 'neutral to distressed',
}
HYPOTHESES_JSON = {
    'coarse': 'negative, low energy',
    'hypotheses': [
        {
            'label': 'sadness',
            'support': ['v1', 'a1', 'a2'],
            'contradiction': [],
            'confidence': 0.8,
        },
        {
            'label': 'anger',
            'support': ['v3'],
            'contradiction': ['v1', 'a2'],
            'confidence': 0.3,
        },
    ],
}
VERIFY_JSON = {'modalities_agree': True, 'hypotheses': HYPOTHESES_JSON['hypotheses']}
SYNTHESIZE_JSON = {
    'steps': [
        'The clip shows a conversation after bad news [v4].',
        'The face is tearful [v1] and the gaze is lowered [v2].',
        'The voice trembles [a2] and the words stay flat [a1].',
        'Seen and heard cues line up [v1, a2]; the alternative is weaker [v3].',
    ],
    'justification': 'The tearful face outweighs the raised voice [v1].',
}

DEFAULT_SCRIPT: Mapping[str, object] = {
    'extract_visual': VISUAL_JSON,
    'generate_hypotheses': HYPOTHESES_JSON,
    'verify': VERIFY_JSON,
    'synthesize': SYNTHESIZE_JSON,
}

#: Candidate list of the shared ``benchmark_example``.
DEFAULT_LABELS: Final[tuple[str, ...]] = ('anger', 'happiness', 'sadness', 'neutral')
#: The label every scripted answer names, matching ``DEFAULT_SCRIPT``.
DEFAULT_LABEL: Final = 'sadness'

_TASK_FOR_QUANTIFIER: Final[Mapping[Quantifier, Task]] = {
    Quantifier.MOST_PROMINENT: Task.ER_WILD,
    Quantifier.ONE_OR_SEVERAL: Task.FG_ER,
    Quantifier.MULTIPLE: Task.ML_ER,
}


def fixed_label(label: str = DEFAULT_LABEL) -> ChooseLabel:
    """A label choice that does not depend on the clip, for scripted scenarios."""
    return lambda _request: label


def scripted_responder(
    script: Mapping[str, object] | None = None,
    *,
    fail_stages: frozenset[str] = frozenset(),
) -> Responder:
    """The harness's own responder, with this suite's default answers over it.

    ``script`` replaces one stage's answer, so a scenario says what the model
    answered; ``fail_stages`` makes a stage answer with prose the first time, so
    a scenario can cost a parse failure.
    """
    return harness_responder(
        fixed_label(),
        fail_stages=fail_stages,
        script={**DEFAULT_SCRIPT, **dict(script or {})},
    )


def reading(
    label: str,
    support: Sequence[str] = (),
    contradiction: Sequence[str] = (),
    confidence: float = 0.5,
) -> dict[str, Any]:
    """One hypothesis row as the Target would return it."""
    return {
        'label': label,
        'support': list(support),
        'contradiction': list(contradiction),
        'confidence': confidence,
    }


def reading_script(
    rows: Sequence[Mapping[str, Any]],
    agree: bool = True,
    coarse: str = 'negative, low energy',
) -> dict[str, object]:
    """A script whose two reading stages both return ``rows``."""
    return {
        'generate_hypotheses': {'coarse': coarse, 'hypotheses': list(rows)},
        'verify': {'modalities_agree': agree, 'hypotheses': list(rows)},
    }


def run_once(
    bundle: HarnessBundle,
    target: FakeTarget,
    example: Example,
    inputs: PerceivedInputs,
    config: PipelineConfig | None = None,
) -> HarnessOutput:
    """One item through the pipeline; every scenario starts here."""
    return HarnessPipeline(bundle, target, config).run(example, inputs)


def cited_ids(output: HarnessOutput) -> set[str]:
    """Every id the emitted steps cite in square brackets."""
    return {
        token.strip(' []')
        for step in output.steps
        for chunk in step.split('[')[1:]
        for token in chunk.split(']')[0].split(',')
    }


@pytest.fixture
def bundle(repo_root: Path) -> HarnessBundle:
    """The committed Stage 1 bundle at the repository root."""
    return load_bundle(repo_root / 'harness')


@pytest.fixture
def edited_bundle(repo_root: Path, tmp_path: Path) -> Callable[..., HarnessBundle]:
    """A copy of the Stage 1 bundle with some of its files replaced."""

    def _edit(name: str, replacements: Mapping[str, str]) -> HarnessBundle:
        return load_bundle(
            write_bundle_copy(repo_root / 'harness', tmp_path / name, replacements)
        )

    return _edit


@pytest.fixture
def audio() -> AudioEvidence:
    return AudioEvidence(
        speech_present=True,
        transcript='I am fine.',
        language='en',
        prosody=('trembling voice', 'low energy'),
    )


@pytest.fixture
def make_inputs(tmp_path: Path, audio: AudioEvidence) -> Callable[..., PerceivedInputs]:
    """Build PerceivedInputs for an example with synthetic media paths."""

    def _make(example: Example, **overrides: object) -> PerceivedInputs:
        crops = tuple(str(tmp_path / f'face_{index}.png') for index in range(4))
        frames = tuple(str(tmp_path / f'frame_{index}.png') for index in range(6))
        values: dict[str, object] = {
            'example_id': example.example_id,
            'video_path': example.media_path,
            'duration_seconds': 4.0,
            'frame_paths': frames,
            'judge_frame_paths': frames[:2],
            'face_crop_paths': crops,
            'audio': audio,
        }
        return PerceivedInputs.model_validate(values | overrides)

    return _make


@pytest.fixture
def inputs(
    benchmark_example: Example, make_inputs: Callable[..., PerceivedInputs]
) -> PerceivedInputs:
    return make_inputs(benchmark_example)


@pytest.fixture
def make_example(tmp_path: Path) -> Callable[..., Example]:
    """An item whose official prompt carries a chosen quantifier and candidate list."""

    def _make(
        quantifier: Quantifier = Quantifier.MOST_PROMINENT,
        labels: tuple[str, ...] = DEFAULT_LABELS,
        example_id: str = 'mme_scenario_000001',
    ) -> Example:
        task = _TASK_FOR_QUANTIFIER[quantifier]
        media = tmp_path / f'{example_id}.mp4'
        media.write_bytes(b'\x00' * 64)
        return Example(
            example_id=example_id,
            source_id='dia_1',
            origin='mme_emotion',
            task=task,
            split=Split.TEST,
            media_path=str(media),
            prompt=render_prompt_for_task(task, labels),
            label_set=labels,
            labels=(labels[0],),
        )

    return _make


@pytest.fixture
def target() -> FakeTarget:
    return FakeTarget(responder=scripted_responder())
