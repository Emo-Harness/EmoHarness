from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.harness.schema import AudioOpinion
from emoharness.perceive.audio import (
    AudioDescriber,
    Diarizer,
    FakeDescriber,
    FakeDiarizer,
    FakeProsody,
    FakeTranscriber,
    OpenAICompatDescriber,
    OpenSmileProsody,
    ProsodyExtractor,
    PyannoteDiarizer,
    SpeechSegment,
    Transcriber,
    Transcript,
    WhisperTranscriber,
    Word,
    dominant_speaker,
    subject_transcript,
)


def _skip_if_installed(module_name: str) -> None:
    """Guard the 'missing dependency' tests in case the optional extra is present."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return
    pytest.skip(
        f'{module_name} is installed; the missing-dependency path does not apply'
    )


def _mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- frozen models -----------------------------------------------------------------


def test_word_is_frozen() -> None:
    word = Word(text='hi', start=0.0, end=0.5)
    with pytest.raises(PydanticValidationError):
        word.text = 'bye'  # type: ignore[misc]


def test_word_rejects_negative_times() -> None:
    with pytest.raises(PydanticValidationError):
        Word(text='hi', start=-1.0, end=0.5)


def test_transcript_defaults() -> None:
    transcript = Transcript(text='hello world')
    assert transcript.language is None
    assert transcript.words == ()


def test_speech_segment_requires_nonempty_speaker() -> None:
    with pytest.raises(PydanticValidationError):
        SpeechSegment(speaker='', start=0.0, end=1.0)


# --- fakes ---------------------------------------------------------------------------


def test_fakes_satisfy_their_protocols() -> None:
    assert isinstance(FakeTranscriber('hi'), Transcriber)
    assert isinstance(FakeProsody({}), ProsodyExtractor)
    assert isinstance(FakeDescriber('d', AudioOpinion()), AudioDescriber)
    assert isinstance(FakeDiarizer(()), Diarizer)


def test_fake_transcriber_is_deterministic(tmp_path: Path) -> None:
    fake = FakeTranscriber('hello', 'en')
    wav = tmp_path / 'a.wav'
    expected = Transcript(text='hello', language='en', words=())
    assert fake.transcribe(wav) == expected
    assert fake.transcribe(wav) == expected


def test_fake_prosody_does_not_alias_internal_state(tmp_path: Path) -> None:
    fake = FakeProsody({'a': 1.0})
    wav = tmp_path / 'a.wav'
    out = cast('dict[str, float]', fake.extract(wav))
    out['a'] = 99.0
    assert fake.extract(wav) == {'a': 1.0}


def test_fake_describer_returns_fixed_values(tmp_path: Path) -> None:
    opinion = AudioOpinion(valence='positive', arousal='high', candidates=('joy',))
    fake = FakeDescriber('calm room tone', opinion)
    wav = tmp_path / 'a.wav'
    assert fake.describe(wav) == 'calm room tone'
    assert fake.opine(wav) == opinion


def test_fake_diarizer_returns_fixed_segments(tmp_path: Path) -> None:
    segments = (SpeechSegment(speaker='A', start=0.0, end=1.0),)
    fake = FakeDiarizer(segments)
    assert fake.diarize(tmp_path / 'a.wav') == segments


# --- dominant_speaker ------------------------------------------------------------


def test_dominant_speaker_empty_is_none() -> None:
    assert dominant_speaker(()) is None


def test_dominant_speaker_single_speaker() -> None:
    segments = (SpeechSegment(speaker='A', start=0.0, end=2.0),)
    assert dominant_speaker(segments) == 'A'


def test_dominant_speaker_picks_greatest_total_duration() -> None:
    segments = (
        SpeechSegment(speaker='A', start=0.0, end=1.0),
        SpeechSegment(speaker='B', start=1.0, end=1.5),
        SpeechSegment(speaker='B', start=2.0, end=3.5),
    )
    assert dominant_speaker(segments) == 'B'


def test_dominant_speaker_tie_break_is_deterministic() -> None:
    segments = (
        SpeechSegment(speaker='A', start=0.0, end=1.0),
        SpeechSegment(speaker='B', start=0.0, end=1.0),
    )
    assert dominant_speaker(segments) == 'B'
    assert dominant_speaker(tuple(reversed(segments))) == 'B'


# --- subject_transcript ----------------------------------------------------------


def test_subject_transcript_without_words_returns_full_text() -> None:
    transcript = Transcript(text='hello there')
    assert subject_transcript(transcript, (), 'A') == 'hello there'


def test_subject_transcript_without_speaker_returns_full_text() -> None:
    words = (Word(text='hi', start=0.0, end=0.5),)
    transcript = Transcript(text='hi', words=words)
    assert subject_transcript(transcript, (), None) == 'hi'


def test_subject_transcript_filters_words_to_speaker_window() -> None:
    words = (
        Word(text='hi', start=0.0, end=0.5),
        Word(text='there', start=1.5, end=2.0),
        Word(text='bye', start=3.0, end=3.5),
    )
    transcript = Transcript(text='hi there bye', words=words)
    segments = (
        SpeechSegment(speaker='A', start=0.0, end=1.0),
        SpeechSegment(speaker='B', start=1.4, end=2.1),
    )
    assert subject_transcript(transcript, segments, 'A') == 'hi'
    assert subject_transcript(transcript, segments, 'B') == 'there'


def test_subject_transcript_unmatched_speaker_falls_back_to_full_text() -> None:
    words = (Word(text='hi', start=0.0, end=0.5),)
    transcript = Transcript(text='hi', words=words)
    segments = (SpeechSegment(speaker='A', start=0.0, end=1.0),)
    assert subject_transcript(transcript, segments, 'C') == 'hi'


# --- real adapters: missing dependency ------------------------------------------


def test_whisper_transcriber_missing_dependency_raises(tmp_path: Path) -> None:
    _skip_if_installed('faster_whisper')
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    with pytest.raises(EmoHarnessError, match='faster_whisper'):
        WhisperTranscriber().transcribe(wav)


def test_opensmile_prosody_missing_dependency_raises(tmp_path: Path) -> None:
    _skip_if_installed('opensmile')
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    with pytest.raises(EmoHarnessError, match='opensmile'):
        OpenSmileProsody().extract(wav)


def test_pyannote_diarizer_missing_dependency_raises(tmp_path: Path) -> None:
    _skip_if_installed('pyannote.audio')
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    with pytest.raises(EmoHarnessError, match='pyannote'):
        PyannoteDiarizer('some/model').diarize(wav)


# --- OpenAICompatDescriber (httpx.MockTransport; no network) --------------------


def test_describe_posts_cue_prompt_and_base64_audio(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'fake-wav-bytes')
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content)
        return httpx.Response(
            200, json={'choices': [{'message': {'content': 'a quiet room tone'}}]}
        )

    describer = OpenAICompatDescriber(
        'https://example.test/v1/chat/completions',
        'test-model',
        client=_mock_client(handler),
    )
    assert describer.describe(wav) == 'a quiet room tone'
    body = captured['body']
    assert body['model'] == 'test-model'
    content = body['messages'][0]['content']
    assert 'emotion' in content[0]['text'].lower()
    assert base64.b64decode(content[1]['input_audio']['data']) == b'fake-wav-bytes'


def test_describe_closes_a_client_it_created_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no client is injected, ``_complete`` builds one and must close it."""
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')
    real_client_cls = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})

    def fake_client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs['transport'] = httpx.MockTransport(handler)
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, 'Client', fake_client_factory)
    describer = OpenAICompatDescriber('https://example.test', 'm')
    assert describer.describe(wav) == 'ok'


def test_opine_parses_json_content_into_audio_opinion(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')
    payload = {'valence': 'positive', 'arousal': 'high', 'candidates': ['joy', 'pride']}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={'choices': [{'message': {'content': json.dumps(payload)}}]}
        )

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    opinion = describer.opine(wav)
    assert opinion == AudioOpinion(
        valence='positive', arousal='high', candidates=('joy', 'pride')
    )


def test_opine_strips_markdown_json_fence(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')
    fenced = '```json\n{"valence": "negative", "arousal": "low", "candidates": []}\n```'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'message': {'content': fenced}}]})

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    opinion = describer.opine(wav)
    assert (opinion.valence, opinion.arousal, opinion.candidates) == (
        'negative',
        'low',
        (),
    )


def test_describer_raises_emoharness_error_on_http_failure(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='boom')

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    with pytest.raises(EmoHarnessError):
        describer.describe(wav)


def test_describer_raises_validation_error_on_non_json_body(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='not json')

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    with pytest.raises(ValidationError):
        describer.describe(wav)


def test_describer_raises_validation_error_on_unexpected_shape(tmp_path: Path) -> None:
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'unexpected': True})

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    with pytest.raises(ValidationError):
        describer.describe(wav)


def test_opine_records_unknown_on_non_json_content(tmp_path: Path) -> None:
    """A prose answer must not abort a corpus: the opinion is one voter, never evidence."""
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')

    def handler(_request: httpx.Request) -> httpx.Response:
        body = {'choices': [{'message': {'content': 'not json'}}]}
        return httpx.Response(200, json=body)

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    assert describer.opine(wav) == AudioOpinion()


def test_opine_coerces_off_schema_fields_to_unknown(tmp_path: Path) -> None:
    """Qwen2.5-Omni answers 'mixed' valence on some clips; that is unknown, not a crash."""
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'x')
    payload = {
        'valence': 'mixed',
        'arousal': 'high',
        'candidates': ['sad', 'happy', 'x'],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={'choices': [{'message': {'content': json.dumps(payload)}}]}
        )

    describer = OpenAICompatDescriber(
        'https://example.test', 'm', client=_mock_client(handler)
    )
    assert describer.opine(wav) == AudioOpinion(
        valence='unknown', arousal='high', candidates=('sad', 'happy')
    )
