"""The scripted Target that exercises this harness, beside the harness itself.

A request carries the stage that issued it (``TargetRequest.tag``), so nothing
here guesses at a prompt's wording: a Builder may rewrite every prompt in the
bundle and this responder still answers each stage with a payload that stage can
parse, built from the harness's own schemas. Requests that carry any other tag
are single-call rows, and get five numbered steps and an ``<answer>`` block.

Which label an answer names is the caller's decision, passed in as
``choose_label``; :func:`label_chooser` is the one a ``--fake`` run uses, and it
is a deterministic function of the clip - the media path for the rows that send
video, the cached transcript for the rows that send an evidence record - so a
run produces a spread of scores instead of one constant.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Final

from emoharness.common.constants import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    NUM_STEPS,
    STEP_PREFIX,
    STEP_TITLES,
)
from emoharness.common.errors import PromptParseError
from emoharness.common.hashing import sha256_text
from emoharness.data.prompt_template import parse_prompt
from emoharness.harness._controller import MODEL_WRITTEN_STEPS
from emoharness.harness._stages import STRICT_JSON_REMINDER
from emoharness.harness.bundle_models import STAGE_NAMES
from emoharness.harness.schema import Hypothesis, HypothesisSet, VisualEvidence
from emoharness.target.base import TargetRequest
from emoharness.target.fake import Responder

__all__ = [
    'ChooseLabel',
    'example_key',
    'label_chooser',
    'label_set_of',
    'scripted_responder',
]

#: Picks the label one request is answered with, given only the request.
ChooseLabel = Callable[[TargetRequest], str]

_DEFAULT_LABELS: Final[tuple[str, ...]] = ('neutral', 'positive', 'negative')
_NOT_JSON: Final = 'Sorry - here is a plain prose answer instead of JSON.'
_COARSE: Final = 'mixed, medium energy'
_RANKED_HYPOTHESES: Final = 3

_TRANSCRIPT_RE: Final = re.compile(r'"transcript"\s*:\s*"(.*?)"', re.DOTALL)
_AUDIO_TEXT_RE: Final = re.compile(r'^Transcript: (.+)$', re.MULTILINE)
_LABEL_SET_RE: Final = re.compile(r'"label_set"\s*:\s*\[(.*?)\]', re.DOTALL)
_FRAMING_RE: Final = re.compile(r'framing note (\d+)')
FRAMING_VARIANTS: Final = 7
#: One clip in this many changes its answer when a prompt is edited, so a Builder
#: patch moves some items without rewriting every answer at once.
FRAMING_SENSITIVITY: Final = 8
#: Prompt-length bucket, so rows that append text to the same clip differ.
PROMPT_BUCKET: Final = 200
#: One clip in this many is answered with the modalities disagreeing.
DISAGREEMENT_RATE: Final = 4

_SYNTHESIS_STEPS: Final[tuple[str, ...]] = (
    'Two people sit in a small room in daylight [v4].',
    'The brows stay level and the gaze is held [v1, v2].',
    'The voice carries the line about the schedule [a1].',
    'Seen and heard cues line up [v1, a1]; the alternative is weaker [v3].',
)
_JUSTIFICATION: Final = 'The held gaze outweighs the strongest alternative [v2].'


def example_key(request: TargetRequest, variant: str = '') -> str:
    """A stable per-clip key: the media path, else the transcript, else the prompt.

    Text stages also mix in the visual stage's framing note, which is a function
    of the extraction prompt. That is what makes a Builder patch able to change
    an answer: editing a prompt changes what the model reports, which changes
    what the later stages conclude, exactly as it would with a real Target.
    """
    if request.video_path:
        base = f'{request.video_path}|p={len(request.prompt) // PROMPT_BUCKET}'
    else:
        match = _TRANSCRIPT_RE.search(request.prompt) or _AUDIO_TEXT_RE.search(
            request.prompt
        )
        base = match.group(1) if match else request.prompt
        base = _with_framing(base, request.prompt)
    if request.temperature > 0.0:
        base = f'{base}|seed={request.seed}'
    return f'{variant}|{base}' if variant else base


def _with_framing(base: str, prompt: str) -> str:
    """Mix the visual stage's framing note into one clip in FRAMING_SENSITIVITY."""
    note = _FRAMING_RE.search(prompt)
    if note is None or int(sha256_text(base)[:4], 16) % FRAMING_SENSITIVITY:
        return base
    return f'{base}|note={note.group(1)}'


def label_set_of(request: TargetRequest) -> tuple[str, ...]:
    """The candidate list, from the official prompt or from the evidence record."""
    try:
        return parse_prompt(request.prompt).label_set
    except PromptParseError:
        pass
    match = _LABEL_SET_RE.search(request.prompt)
    if match is None:
        return _DEFAULT_LABELS
    try:
        parsed = json.loads(f'[{match.group(1)}]')
    except json.JSONDecodeError:  # pragma: no cover -- the record is always valid JSON
        return _DEFAULT_LABELS
    labels = tuple(str(item) for item in parsed if str(item).strip())
    return labels or _DEFAULT_LABELS


def label_chooser(variant: str = '') -> ChooseLabel:
    """Name one candidate label per clip, the same one every time it is asked.

    ``variant`` stands in for the model's identity: two variants answer the same
    request differently, the way two real models would.
    """

    def choose(request: TargetRequest) -> str:
        labels = label_set_of(request)
        index = int(sha256_text(example_key(request, variant))[:8], 16)
        return labels[index % len(labels)]

    return choose


def scripted_responder(
    choose_label: ChooseLabel,
    *,
    fail_stages: frozenset[str] = frozenset(),
    script: Mapping[str, object] | None = None,
) -> Responder:
    """A ``FakeTarget`` responder that answers every stage and every single-call row.

    ``choose_label`` decides which label an answer names, so a caller steers the
    outcome without knowing any payload's shape. A stage in ``fail_stages``
    answers with prose the first time it is asked and normally on the retry, so a
    scenario can cost a parse failure. ``script`` replaces one stage's whole
    answer, for the shapes this module would never produce on its own.
    """
    answers = dict(script or {})

    def respond(request: TargetRequest) -> str:
        stage = request.tag or ''
        if stage in fail_stages and STRICT_JSON_REMINDER not in request.prompt:
            return _NOT_JSON
        if stage in answers:
            return _as_text(answers[stage])
        if stage == 'direct_same_tools_v1':
            payload = json.loads(_visual_answer(request))
            payload.update(
                _hypotheses(request, choose_label(request)).model_dump(mode='json')
            )
            payload['steps'] = list(_SYNTHESIS_STEPS[:MODEL_WRITTEN_STEPS])
            return json.dumps(payload)
        if stage not in STAGE_NAMES:
            return _single_call_answer(choose_label(request))
        return _stage_answer(stage, request, choose_label(request))

    return respond


def _as_text(answer: object) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer)


def _stage_answer(stage: str, request: TargetRequest, label: str) -> str:
    """One stage's answer, in the shape that stage parses."""
    if stage == STAGE_NAMES[0]:
        return _visual_answer(request)
    if stage == STAGE_NAMES[3]:
        return json.dumps(
            {
                'steps': list(_SYNTHESIS_STEPS[:MODEL_WRITTEN_STEPS]),
                'justification': _JUSTIFICATION,
            }
        )
    proposed = _hypotheses(request, label)
    if stage == STAGE_NAMES[1]:
        return json.dumps(proposed.model_dump(mode='json'))
    rows = [row.model_dump(mode='json') for row in proposed.hypotheses]
    return json.dumps({'modalities_agree': _agrees(request), 'hypotheses': rows})


def _visual_answer(request: TargetRequest) -> str:
    """Seen cues for one clip. The framing note depends on the prompt, not the clip."""
    note = int(sha256_text(request.prompt)[:6], 16) % FRAMING_VARIANTS
    shifts = int(sha256_text(example_key(request))[:2], 16) % 2 == 0
    evidence = VisualEvidence(
        face=('brows level, eyes on the other person', 'blink rate steady'),
        body=('shoulders squared toward the camera', f'framing note {note}'),
        scene=('two people in a small room, daytime',),
        temporal_change='posture shifts once, about halfway through'
        if shifts
        else None,
    )
    return json.dumps(evidence.model_dump(mode='json'))


def _hypotheses(request: TargetRequest, label: str) -> HypothesisSet:
    """Three readings, the chosen label first, citing ids the record will carry."""
    ranked = _ranked_labels(label_set_of(request), label)
    return HypothesisSet(
        coarse=_COARSE,
        hypotheses=(
            Hypothesis(label=ranked[0], support=('v1', 'a1'), confidence=0.72),
            Hypothesis(label=ranked[1], support=('v2', 'a1'), confidence=0.45),
            Hypothesis(
                label=ranked[2],
                support=('v3',),
                contradiction=('v1', 'a1'),
                confidence=0.18,
            ),
        ),
    )


def _ranked_labels(labels: tuple[str, ...], chosen: str) -> tuple[str, ...]:
    """The chosen label, then the candidates that follow it in the list."""
    start = labels.index(chosen) if chosen in labels else 0
    rotated = tuple(
        labels[(start + offset) % len(labels)] for offset in range(_RANKED_HYPOTHESES)
    )
    return (chosen, *rotated[1:])


def _agrees(request: TargetRequest) -> bool:
    """Whether this clip's modalities are reported as agreeing."""
    return int(sha256_text(example_key(request))[:2], 16) % DISAGREEMENT_RATE != 0


def _single_call_answer(label: str) -> str:
    """A row that gets one call: the five numbered steps and the answer tag."""
    bodies = (
        'two people in a small room.',
        'brows level, gaze held.',
        'an even voice, one long pause.',
        'the cues agree; the alternative is weaker.',
        f'the reading is {label}.',
    )
    lines = tuple(
        f'{STEP_PREFIX} {number}: {title}: {body}'
        for number, (title, body) in enumerate(
            zip(STEP_TITLES[:NUM_STEPS], bodies, strict=True), start=1
        )
    )
    return '\n'.join(lines) + f'\n{ANSWER_OPEN}{label}{ANSWER_CLOSE}'
