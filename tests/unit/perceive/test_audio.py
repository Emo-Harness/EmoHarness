from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import EmoHarnessError
from emoharness.perceive.audio import (
    Diarizer,
    FakeDiarizer,
    FakeProsody,
    FakeTranscriber,
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
