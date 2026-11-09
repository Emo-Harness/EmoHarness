"""Evidence, hypothesis, verification, and output records shared across the harness.

Every model is frozen. Stages produce new records; nothing is mutated in place.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from pydantic.json_schema import SkipJsonSchema

from emoharness.data.prompt_template import ParsedPrompt

Modality = Literal['visual', 'audio', 'text']


class EvidenceItem(BaseModel):
    """One citable cue. Reasoning steps may only cite ids that exist in a record."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str = Field(min_length=1)
    modality: Modality
    cue: str = Field(min_length=1)
    segment: tuple[float, float] | None = Field(
        default=None, description='(start_s, end_s)'
    )


class VisualEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    face: tuple[str, ...] = ()
    body: tuple[str, ...] = ()
    scene: tuple[str, ...] = ()
    temporal_change: str | None = None


class Utterance(BaseModel):
    """One timed stretch of speech: when, which voice, what was said, how it sounded.

    ``speaker`` is a voice-cluster id (``S1``, ``S2`` ...) in order of first
    appearance, never a name or a role; ``None`` when speakers are not clustered
    or the stretch is too short to embed. ``prosody`` uses the same phrases as the
    clip-level channel, against utterance-level reference statistics.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    speaker: str | None = None
    prosody: tuple[str, ...] = ()


class AudioEvidence(BaseModel):
    """Audio cues from the selected perception protocol.

    When enabled, transcript and language come from Whisper; prosody comes from
    openSMILE functionals compared with corpus reference statistics. No model
    opinion enters here. Empty transcript means no intelligible speech.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    speech_present: bool | None = Field(
        description='None when transcription is disabled; unknown is not silence'
    )
    transcript: str = ''
    language: str | None = None
    prosody: tuple[str, ...] = ()
    # Filled only by a perception config with an ``utterances`` section; without
    # one the record serialises exactly as before the field existed, so every
    # earlier prompt and cache key is unchanged. Kept out of the exported JSON
    # schema (and out of the docstring, which becomes its description): the
    # frozen bundles ship that schema as a file, and their hashes must not move.
    utterances: SkipJsonSchema[tuple[Utterance, ...]] = ()

    @model_serializer(mode='wrap')
    def _omit_empty_utterances(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not self.utterances:
            data.pop('utterances', None)
        return data


class EvidenceRecord(BaseModel):
    """The structured record assembled before any label is requested."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    visual: VisualEvidence
    audio: AudioEvidence
    text: ParsedPrompt
    items: tuple[EvidenceItem, ...] = Field(
        default=(), description='flattened citable cues'
    )

    def item_ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self.items)

    def decision_ids(self) -> frozenset[str]:
        """Observed cues only; the question is context, never label evidence."""
        return frozenset(item.id for item in self.items if item.modality != 'text')


class Hypothesis(BaseModel):
    """A candidate label with the evidence ids that support or contradict it."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    label: str = Field(min_length=1)
    support: tuple[str, ...] = ()
    contradiction: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class HypothesisSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    coarse: str = Field(
        min_length=1, description='coarse state, e.g. negative-high-arousal'
    )
    hypotheses: tuple[Hypothesis, ...] = Field(min_length=1)


class VerificationResult(BaseModel):
    """Outcome of the verification stage: agreement flag and a ranked hypothesis list."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    modalities_agree: bool
    ranked: tuple[Hypothesis, ...] = Field(min_length=1)
    notes: str = ''


class StageTrace(BaseModel):
    """What one pipeline stage cost. Kept for parse-failure and budget reporting."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stage: str
    calls: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0.0)


class HarnessOutput(BaseModel):
    """Final per-item result: emitted labels, the five steps, and the full response text."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str
    labels: tuple[str, ...] = Field(
        min_length=1, description='verbatim from the candidate list'
    )
    steps: tuple[str, ...] = Field(min_length=1)
    response_text: str = Field(
        min_length=1, description='what the official judge receives'
    )
    evidence: EvidenceRecord
    hypotheses: HypothesisSet
    verification: VerificationResult
    traces: tuple[StageTrace, ...] = ()
    fallback_used: bool = False

    @property
    def target_calls(self) -> int:
        return sum(trace.calls for trace in self.traces)

    @property
    def parse_failures(self) -> int:
        return sum(trace.parse_failures for trace in self.traces)


class PerceivedInputs(BaseModel):
    """Everything the pipeline needs about one clip before the first Target call.

    Produced by the preprocessing cache (perceive) and consumed by the harness and
    by the single-call baselines, so every method sees identical inputs.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str
    video_path: str
    duration_seconds: float = Field(ge=0.0)
    frame_paths: tuple[str, ...] = Field(default=(), description='2 fps frames, 4-32')
    judge_frame_paths: tuple[str, ...] = Field(
        default=(), description='1 fps frames, at most 10'
    )
    face_crop_paths: tuple[str, ...] = ()
    audio: AudioEvidence

    def audio_text(self) -> str:
        """Observable-cue audio channel rendered as the text block single-call rows append."""
        speech = (
            'unknown (ASR disabled)'
            if self.audio.speech_present is None
            else 'yes'
            if self.audio.speech_present
            else 'no'
        )
        lines = [f'Speech present: {speech}']
        if self.audio.utterances:
            lines.append('Utterances:')
            lines.extend(utterance_line(u) for u in self.audio.utterances)
        elif self.audio.transcript:
            lines.append(f'Transcript: {self.audio.transcript}')
        if self.audio.language:
            lines.append(f'Language: {self.audio.language}')
        if self.audio.prosody:
            lines.append('Prosody: ' + '; '.join(self.audio.prosody))
        return '\n'.join(lines)


def utterance_line(utterance: Utterance) -> str:
    """``[12.3-15.8s] S1: text (high pitch; loud voice)``, the one rendering every reader uses."""
    who = f' {utterance.speaker}' if utterance.speaker else ''
    line = f'[{utterance.start:.1f}-{utterance.end:.1f}s]{who}: {utterance.text}'
    if utterance.prosody:
        line += ' (' + '; '.join(utterance.prosody) + ')'
    return line
