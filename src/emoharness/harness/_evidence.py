"""Turn cached inputs and the visual stage's answer into one citable record.

Implementation detail of the harness; ``HarnessPipeline.run`` is the only caller.
Ids are assigned here and nowhere else: ``v1..vn`` for what was seen, ``a1..an``
for what was heard (transcript sentences, then prosody phrases), and ``t1`` for
the wording of the question. Every later stage may only cite an id that exists
in the record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from emoharness.common.text import truncate
from emoharness.data.prompt_template import ParsedPrompt
from emoharness.harness._text import split_sentences
from emoharness.harness.schema import (
    AudioEvidence,
    EvidenceItem,
    EvidenceRecord,
    Modality,
    PerceivedInputs,
    VisualEvidence,
)

_VISUAL_ID_PREFIX: Final = 'v'
_AUDIO_ID_PREFIX: Final = 'a'
_TEXT_ID_PREFIX: Final = 't'

_SPEECH_CUE_PREFIX: Final = 'speech: '
_PROSODY_CUE_PREFIX: Final = 'prosody: '
_NO_SPEECH_CUE: Final = 'no intelligible speech in the clip'


def build_record(
    parsed: ParsedPrompt,
    visual: VisualEvidence,
    inputs: PerceivedInputs,
    transcript_max_chars: int | None = None,
) -> EvidenceRecord:
    """The full evidence record assembled before any label is requested."""
    items = (
        _visual_items(visual)
        + _audio_items(inputs.audio, transcript_max_chars)
        + _text_items(parsed.noun)
    )
    return EvidenceRecord(
        visual=visual,
        audio=_truncated_audio(inputs.audio, transcript_max_chars),
        text=parsed,
        items=items,
    )


def _visual_items(visual: VisualEvidence) -> tuple[EvidenceItem, ...]:
    """Citable items v1..vn, in the order face, body, scene, temporal change."""
    cues = tuple(visual.face) + tuple(visual.body) + tuple(visual.scene)
    if visual.temporal_change:
        cues = cues + (visual.temporal_change,)
    return _numbered(cues, _VISUAL_ID_PREFIX, 'visual')


def _audio_items(
    audio: AudioEvidence, transcript_max_chars: int | None = None
) -> tuple[EvidenceItem, ...]:
    """Citable items a1..an: transcript sentences, then prosody phrases."""
    transcript = _cut(audio.transcript, transcript_max_chars)
    speech = tuple(_SPEECH_CUE_PREFIX + part for part in split_sentences(transcript))
    if not speech and not audio.speech_present:
        speech = (_NO_SPEECH_CUE,)
    prosody = tuple(
        _PROSODY_CUE_PREFIX + part for part in audio.prosody if part.strip()
    )
    return _numbered(speech + prosody, _AUDIO_ID_PREFIX, 'audio')


def _text_items(noun: str) -> tuple[EvidenceItem, ...]:
    """The single item t1: what the question asks for, in its own words."""
    return _numbered((noun,), _TEXT_ID_PREFIX, 'text')


def _numbered(
    cues: Sequence[str], prefix: str, modality: Modality
) -> tuple[EvidenceItem, ...]:
    cleaned = tuple(' '.join(cue.split()) for cue in cues if cue and cue.strip())
    return tuple(
        EvidenceItem(id=f'{prefix}{number}', modality=modality, cue=cue)
        for number, cue in enumerate(cleaned, start=1)
    )


def _cut(text: str, limit: int | None) -> str:
    """A transcript is cut, never marked: a citation must quote the clip verbatim."""
    return truncate(text, limit, marker='')


def _truncated_audio(audio: AudioEvidence, limit: int | None) -> AudioEvidence:
    transcript = _cut(audio.transcript, limit)
    if transcript == audio.transcript:
        return audio
    return audio.model_copy(update={'transcript': transcript})
