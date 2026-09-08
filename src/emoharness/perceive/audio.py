"""Audio evidence protocols, frozen records, deterministic fakes, and real adapters.

Real adapters (``WhisperTranscriber``, ``OpenSmileProsody``, ``PyannoteDiarizer``)
import their heavy dependency lazily inside the call and raise ``EmoHarnessError``
naming the missing package. ``OpenAICompatDescriber`` uses ``httpx`` (a core
dependency) but never touches the network in tests; tests inject a client built on
``httpx.MockTransport`` instead.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.harness.schema import AudioOpinion

_CUE_PROMPT: Final = (
    'Describe only the observable vocal cues and non-speech sounds in this audio '
    'clip: pitch, pace, volume, tone quality, laughter, crying, sighing, silence, '
    'and background noise. Do not name, imply, or guess any emotion.'
)
_OPINION_PROMPT: Final = (
    'Listen to this audio clip and reply with only a JSON object with three '
    'fields: "valence" (one of "negative", "neutral", "positive"), "arousal" '
    '(one of "low", "medium", "high"), and "candidates" (a list of at most two '
    'single-word candidate emotion labels).'
)


class Word(BaseModel):
    """One transcribed word with its time span, in seconds."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    text: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)


class Transcript(BaseModel):
    """A transcription result. Empty ``text`` means no intelligible speech."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    text: str
    language: str | None = None
    words: tuple[Word, ...] = ()


class SpeechSegment(BaseModel):
    """One diarized speech turn."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    speaker: str = Field(min_length=1)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, wav: Path, /) -> Transcript: ...


@runtime_checkable
class ProsodyExtractor(Protocol):
    def extract(self, wav: Path, /) -> Mapping[str, float]: ...


@runtime_checkable
class AudioDescriber(Protocol):
    def describe(self, wav: Path, /) -> str: ...

    def opine(self, wav: Path, /) -> AudioOpinion: ...


@runtime_checkable
class Diarizer(Protocol):
    def diarize(self, wav: Path, /) -> tuple[SpeechSegment, ...]: ...


class FakeTranscriber:
    """Deterministic transcriber that always returns the same transcript."""

    def __init__(
        self, text: str, language: str | None = None, words: tuple[Word, ...] = ()
    ) -> None:
        self._text = text
        self._language = language
        self._words = words

    def transcribe(self, _wav: Path) -> Transcript:
        return Transcript(text=self._text, language=self._language, words=self._words)


class FakeProsody:
    """Deterministic prosody extractor that always returns the same features."""

    def __init__(self, features: Mapping[str, float]) -> None:
        self._features = dict(features)

    def extract(self, _wav: Path) -> Mapping[str, float]:
        return dict(self._features)


class FakeDescriber:
    """Deterministic describer that always returns the same description and opinion."""

    def __init__(self, description: str, opinion: AudioOpinion) -> None:
        self._description = description
        self._opinion = opinion

    def describe(self, _wav: Path) -> str:
        return self._description

    def opine(self, _wav: Path) -> AudioOpinion:
        return self._opinion


class FakeDiarizer:
    """Deterministic diarizer that always returns the same segments."""

    def __init__(self, segments: tuple[SpeechSegment, ...]) -> None:
        self._segments = segments

    def diarize(self, _wav: Path) -> tuple[SpeechSegment, ...]:
        return self._segments


class WhisperTranscriber:
    """faster-whisper transcription with word timestamps."""

    def __init__(self, model_size: str = 'large-v3', device: str = 'auto') -> None:
        self._model_size = model_size
        self._device = device

    def transcribe(self, wav: Path) -> Transcript:
        try:
            from faster_whisper import WhisperModel  # pragma: no cover
        except ImportError as exc:
            raise EmoHarnessError(
                'faster_whisper is required for WhisperTranscriber; '
                "install the 'audio' extra"
            ) from exc
        model = WhisperModel(self._model_size, device=self._device)
        raw_segments, info = model.transcribe(str(wav), word_timestamps=True)
        segments = list(raw_segments)
        words = tuple(
            Word(text=word.word.strip(), start=word.start, end=word.end)
            for segment in segments
            for word in segment.words or ()
        )
        text = ''.join(segment.text for segment in segments).strip()
        return Transcript(text=text, language=info.language, words=words)


class OpenSmileProsody:
    """openSMILE eGeMAPSv02 functionals for one audio file."""

    def extract(self, wav: Path) -> Mapping[str, float]:
        try:
            import opensmile  # pragma: no cover
        except ImportError as exc:
            raise EmoHarnessError(
                "opensmile is required for OpenSmileProsody; install the 'audio' extra"
            ) from exc
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        frame = smile.process_file(str(wav))
        return {str(column): float(frame[column].iloc[0]) for column in frame.columns}


class PyannoteDiarizer:
    """Speaker diarization via a pretrained pyannote.audio pipeline."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def diarize(self, wav: Path) -> tuple[SpeechSegment, ...]:
        try:
            from pyannote.audio import Pipeline  # pragma: no cover
        except ImportError as exc:
            raise EmoHarnessError(
                'pyannote.audio is required for PyannoteDiarizer; '
                "install the 'audio' extra"
            ) from exc
        pipeline = Pipeline.from_pretrained(self._model_name)
        diarization = pipeline(str(wav))
        return tuple(
            SpeechSegment(
                speaker=str(speaker), start=float(segment.start), end=float(segment.end)
            )
            for segment, _, speaker in diarization.itertracks(yield_label=True)
        )


class OpenAICompatDescriber:
    """Posts audio to an OpenAI-compatible chat endpoint for cues and an opinion."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint
        self._model_id = model_id
        self._client = client
        self._timeout = timeout

    def describe(self, wav: Path) -> str:
        return self._complete(_CUE_PROMPT, wav)

    def opine(self, wav: Path) -> AudioOpinion:
        return _parse_opinion(self._complete(_OPINION_PROMPT, wav))

    def _complete(self, prompt: str, wav: Path) -> str:
        payload = _build_payload(self._model_id, prompt, _read_audio_b64(wav))
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            response = client.post(self._endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmoHarnessError(f'audio describer request failed: {exc}') from exc
        finally:
            if self._client is None:
                client.close()
        return _extract_content(_parse_json_body(response))


def _read_audio_b64(wav: Path) -> str:
    return base64.b64encode(wav.read_bytes()).decode('ascii')


def _build_payload(model_id: str, prompt: str, audio_b64: str) -> dict[str, Any]:
    return {
        'model': model_id,
        'temperature': 0.0,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'input_audio',
                        'input_audio': {'data': audio_b64, 'format': 'wav'},
                    },
                ],
            }
        ],
    }


def _parse_json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f'describer response is not JSON: {response.text!r}'
        ) from exc


def _extract_content(data: Any) -> str:
    try:
        return str(data['choices'][0]['message']['content'])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError(f'unexpected describer response shape: {data!r}') from exc


def _parse_opinion(raw: str) -> AudioOpinion:
    cleaned = raw.strip().strip('`').strip()
    if cleaned.lower().startswith('json'):
        cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValidationError(f'describer opinion is not valid JSON: {raw!r}') from exc
    try:
        return AudioOpinion.model_validate(data)
    except PydanticValidationError as exc:
        raise ValidationError(f'describer opinion failed validation: {data!r}') from exc


def dominant_speaker(segments: Sequence[SpeechSegment]) -> str | None:
    """The speaker with the greatest total speech duration, or ``None`` if none.

    Ties are broken by speaker id, ascending, for a fully deterministic result.
    """
    if not segments:
        return None
    totals: dict[str, float] = {}
    for segment in segments:
        duration = max(0.0, segment.end - segment.start)
        totals[segment.speaker] = totals.get(segment.speaker, 0.0) + duration
    return max(totals, key=lambda speaker: (totals[speaker], speaker))


def subject_transcript(
    transcript: Transcript, segments: Sequence[SpeechSegment], speaker: str | None
) -> str:
    """Words whose midpoint falls inside one of ``speaker``'s segments.

    Falls back to the full transcript text when there is no word timing, no
    speaker, or no segment for that speaker.
    """
    if not transcript.words or speaker is None:
        return transcript.text
    windows = tuple((s.start, s.end) for s in segments if s.speaker == speaker)
    if not windows:
        return transcript.text
    kept = [word.text for word in transcript.words if _midpoint_within(word, windows)]
    return ' '.join(kept).strip()


def _midpoint_within(word: Word, windows: Sequence[tuple[float, float]]) -> bool:
    midpoint = (word.start + word.end) / 2
    return any(start <= midpoint <= end for start, end in windows)
