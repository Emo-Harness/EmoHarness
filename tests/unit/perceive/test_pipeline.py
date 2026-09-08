from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pytest

from emoharness.common.config import load_model
from emoharness.common.hashing import sha256_file
from emoharness.data.manifest import Example
from emoharness.data.tasks import Split, Task
from emoharness.harness.schema import AudioOpinion
from emoharness.perceive import pipeline as pipeline_module
from emoharness.perceive.audio import (
    FakeDescriber,
    FakeDiarizer,
    FakeProsody,
    FakeTranscriber,
    SpeechSegment,
    Word,
)
from emoharness.perceive.cache import PreprocessCache, PreprocessConfig
from emoharness.perceive.descriptors import ReferenceStats
from emoharness.perceive.pipeline import (
    PerceptionTools,
    preprocess_clip,
    preprocess_many,
)


class _PatchVideoIO(Protocol):
    def __call__(
        self, duration: float, *, has_audio: bool = True
    ) -> dict[str, int]: ...


@pytest.fixture
def patch_video_io(monkeypatch: pytest.MonkeyPatch) -> _PatchVideoIO:
    """Replace ffmpeg-backed video functions with counting, file-writing fakes."""
    counts = {'probe': 0, 'frames': 0, 'audio': 0}

    def _install(duration: float, *, has_audio: bool = True) -> dict[str, int]:
        def fake_probe(_path: Path) -> float:
            counts['probe'] += 1
            return duration

        def fake_extract_frames(
            _path: Path, timestamps: Sequence[float], out_dir: Path, prefix: str
        ) -> tuple[Path, ...]:
            counts['frames'] += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(len(timestamps)):
                frame_path = out_dir / f'{prefix}_{index:04d}.jpg'
                frame_path.write_bytes(b'jpeg')
                paths.append(frame_path)
            return tuple(paths)

        def fake_extract_audio(_path: Path, wav_path: Path) -> Path | None:
            counts['audio'] += 1
            return wav_path if has_audio else None

        monkeypatch.setattr(pipeline_module, 'probe_duration', fake_probe)
        monkeypatch.setattr(pipeline_module, 'extract_frames', fake_extract_frames)
        monkeypatch.setattr(pipeline_module, 'extract_audio_track', fake_extract_audio)
        return counts

    return _install


@pytest.fixture
def config(repo_root: Path) -> PreprocessConfig:
    return load_model(repo_root / 'configs' / 'preprocess.yaml', PreprocessConfig)


@pytest.fixture
def cache(tmp_path: Path) -> PreprocessCache:
    return PreprocessCache(tmp_path / 'cache')


def test_perception_tools_fakes_has_safe_defaults() -> None:
    tools = PerceptionTools.fakes()
    assert isinstance(tools, PerceptionTools)
    assert tools.reference == ReferenceStats(mean={}, std={})
    assert tools.banned == frozenset()


def test_preprocess_clip_assembles_audio_evidence_from_fakes(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    counts = patch_video_io(duration=6.0, has_audio=True)
    tools = PerceptionTools.fakes(
        transcript_text='hello world',
        language='en',
        prosody_features={'loudness_sma3_amean': 100.0},
        description='A calm room tone is present. There is an angry undertone here.',
        opinion=AudioOpinion(
            valence='negative', arousal='medium', candidates=('anger',)
        ),
        reference=ReferenceStats(
            mean={'loudness_sma3_amean': 0.0}, std={'loudness_sma3_amean': 1.0}
        ),
        banned=frozenset({'angry'}),
    )

    clip = preprocess_clip(benchmark_example, config, tools, cache)

    assert clip.duration_seconds == 6.0
    assert len(clip.frame_paths) == 12  # floor(6.0 * 2 fps)
    assert len(clip.judge_frame_paths) == 6  # 0..5 at 1 fps
    assert clip.audio.speech_present is True
    assert clip.audio.transcript == 'hello world'
    assert clip.audio.language == 'en'
    assert clip.audio.prosody == ('loud voice',)
    assert clip.audio.description == 'A calm room tone is present'
    assert clip.audio_opinion == AudioOpinion(
        valence='negative', arousal='medium', candidates=('anger',)
    )
    assert clip.transcript_words == 0
    assert counts == {'probe': 1, 'frames': 2, 'audio': 1}


def test_preprocess_clip_restricts_transcript_to_dominant_speaker(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    patch_video_io(duration=4.0, has_audio=True)
    words = (
        Word(text='hi', start=0.0, end=0.5),
        Word(text='you', start=1.0, end=1.5),
        Word(text='bye', start=2.2, end=2.6),
    )
    segments = (
        SpeechSegment(speaker='A', start=0.0, end=2.0),  # 2.0s: dominant
        SpeechSegment(speaker='B', start=2.0, end=3.0),  # 1.0s
    )
    tools = PerceptionTools(
        transcriber=FakeTranscriber('hi you bye', 'en', words),
        prosody=FakeProsody({}),
        describer=FakeDescriber('', AudioOpinion()),
        diarizer=FakeDiarizer(segments),
        reference=ReferenceStats(mean={}, std={}),
        banned=frozenset(),
    )

    clip = preprocess_clip(benchmark_example, config, tools, cache)

    assert clip.audio.transcript == 'hi you'
    assert clip.transcript_words == 3


def test_preprocess_clip_no_audio_stream_gives_speech_present_false(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    patch_video_io(duration=3.0, has_audio=False)
    tools = PerceptionTools.fakes(transcript_text='should not be used')

    clip = preprocess_clip(benchmark_example, config, tools, cache)

    assert clip.audio.speech_present is False
    assert clip.audio.transcript == ''
    assert clip.audio.prosody == ()
    assert clip.audio.description == ''
    assert clip.audio_opinion is None
    assert clip.transcript_words == 0


def test_preprocess_clip_silent_audio_track_gives_speech_present_false(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    patch_video_io(duration=3.0, has_audio=True)
    tools = PerceptionTools.fakes(transcript_text='', description='')

    clip = preprocess_clip(benchmark_example, config, tools, cache)

    assert clip.audio.speech_present is False


def test_preprocess_clip_nonspeech_description_alone_gives_speech_present_true(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    patch_video_io(duration=3.0, has_audio=True)
    tools = PerceptionTools.fakes(
        transcript_text='', description='Loud music plays throughout.'
    )

    clip = preprocess_clip(benchmark_example, config, tools, cache)

    assert clip.audio.speech_present is True
    assert clip.audio.transcript == ''
    assert clip.audio.description == 'Loud music plays throughout'


def test_preprocess_clip_uses_cache_on_second_call(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    counts = patch_video_io(duration=3.0, has_audio=False)
    tools = PerceptionTools.fakes(transcript_text='hi')

    first = preprocess_clip(benchmark_example, config, tools, cache)
    second = preprocess_clip(benchmark_example, config, tools, cache)

    assert first == second
    assert counts['probe'] == 1


def test_preprocess_clip_recomputes_when_config_hash_changes(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    counts = patch_video_io(duration=3.0, has_audio=False)
    other_frames = config.frames.model_copy(update={'fps': 1.0})
    other_config = config.model_copy(update={'frames': other_frames})
    tools = PerceptionTools.fakes(transcript_text='hi')

    preprocess_clip(benchmark_example, config, tools, cache)
    preprocess_clip(benchmark_example, other_config, tools, cache)

    assert counts['probe'] == 2


def test_preprocess_clip_recomputes_when_cache_is_missing_a_file(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    counts = patch_video_io(duration=3.0, has_audio=False)
    tools = PerceptionTools.fakes(transcript_text='hi')

    first = preprocess_clip(benchmark_example, config, tools, cache)
    for path in first.frame_paths:
        Path(path).unlink()

    preprocess_clip(benchmark_example, config, tools, cache)

    assert counts['probe'] == 2


def test_preprocess_many_preserves_order_and_returns_every_clip(
    benchmark_example: Example,
    multilabel_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
) -> None:
    patch_video_io(duration=3.0, has_audio=False)
    tools = PerceptionTools.fakes()

    results = preprocess_many(
        [benchmark_example, multilabel_example], config, tools, cache
    )

    expected_hashes = [
        sha256_file(Path(benchmark_example.media_path)),
        sha256_file(Path(multilabel_example.media_path)),
    ]
    assert [clip.media_hash for clip in results] == expected_hashes


def test_preprocess_many_logs_progress_for_the_final_item(
    benchmark_example: Example,
    config: PreprocessConfig,
    cache: PreprocessCache,
    patch_video_io: _PatchVideoIO,
    caplog: pytest.LogCaptureFixture,
) -> None:
    patch_video_io(duration=3.0, has_audio=False)
    tools = PerceptionTools.fakes()

    logger = logging.getLogger('emoharness.perceive.pipeline')
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger='emoharness.perceive.pipeline'):
            preprocess_many([benchmark_example], config, tools, cache)
    finally:
        logger.removeHandler(caplog.handler)

    assert any('1/1' in message for message in caplog.messages)


@pytest.mark.integration
def test_preprocess_clip_end_to_end_with_real_ffmpeg(
    synthetic_clip: Path,
    config: PreprocessConfig,
    cache: PreprocessCache,
) -> None:
    example = Example(
        example_id='int_test_1',
        source_id='src_1',
        origin='mme_emotion',
        task=Task.ER_WILD,
        split=Split.TEST,
        media_path=str(synthetic_clip),
        prompt='dummy prompt',
        label_set=('anger', 'happiness'),
        labels=('anger',),
    )
    tools = PerceptionTools.fakes(transcript_text='hello there', language='en')

    clip = preprocess_clip(example, config, tools, cache)

    assert clip.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert len(clip.frame_paths) >= config.frames.min_frames
    assert all(Path(p).exists() for p in clip.frame_paths)
    assert all(Path(p).exists() for p in clip.judge_frame_paths)
    assert clip.audio.speech_present is True
    assert clip.audio.transcript == 'hello there'
    assert cache.verify(clip) is True

    cached_again = preprocess_clip(example, config, tools, cache)
    assert cached_again == clip
