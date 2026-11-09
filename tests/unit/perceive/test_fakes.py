"""Cached clips without ffmpeg: what the perception fakes write and reuse."""

from __future__ import annotations

from pathlib import Path

import imagehash
import pytest
from PIL import Image

from emoharness.common.config import load_model
from emoharness.data.manifest import Example
from emoharness.perceive.cache import PreprocessCache, PreprocessConfig
from emoharness.perceive.fakes import (
    fake_audio,
    fake_perception_tools,
    fake_preprocess,
    fake_transcript,
)


@pytest.fixture
def preprocess_config(repo_root: Path) -> PreprocessConfig:
    """The committed preprocessing config; its hash keys the cache."""
    return load_model(repo_root / 'configs' / 'preprocess.yaml', PreprocessConfig)


def _phash(path: str) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(path))


def _with_media(example: Example, tmp_path: Path, index: int) -> Example:
    """The same item pointed at a clip whose bytes are its own."""
    media = tmp_path / f'clip_{index}.mp4'
    media.write_bytes(bytes([index]) * 64)
    return example.model_copy(
        update={'example_id': f'clip_{index}', 'media_path': str(media)}
    )


class TestFakeTranscriptAndAudio:
    def test_a_transcript_is_stable_in_its_seed(self) -> None:
        assert fake_transcript(7) == fake_transcript(7)

    def test_two_clips_do_not_share_a_transcript(self) -> None:
        assert fake_transcript(7) != fake_transcript(8)

    def test_audio_evidence_carries_a_transcript_and_prosody(
        self, benchmark_example: Example
    ) -> None:
        audio = fake_audio(benchmark_example, 11)

        assert audio.speech_present is True
        assert audio.transcript
        assert audio.prosody


class TestFakePreprocess:
    def test_writes_frames_and_caches_the_entry(
        self,
        tmp_path: Path,
        benchmark_example: Example,
        preprocess_config: PreprocessConfig,
    ) -> None:
        cache = PreprocessCache(tmp_path / 'derived')

        clip = fake_preprocess(benchmark_example, cache, preprocess_config)

        assert clip.frame_paths and clip.judge_frame_paths
        assert all(Path(path).is_file() for path in clip.judge_frame_paths)
        assert cache.get(clip.media_hash, clip.config_hash) == clip

    def test_reuses_a_verified_entry(
        self,
        tmp_path: Path,
        benchmark_example: Example,
        preprocess_config: PreprocessConfig,
    ) -> None:
        cache = PreprocessCache(tmp_path / 'derived')

        first = fake_preprocess(benchmark_example, cache, preprocess_config)
        second = fake_preprocess(benchmark_example, cache, preprocess_config)

        assert first == second

    def test_frames_and_transcripts_differ_between_clips(
        self,
        tmp_path: Path,
        benchmark_example: Example,
        preprocess_config: PreprocessConfig,
    ) -> None:
        cache = PreprocessCache(tmp_path / 'derived')
        clips = [
            fake_preprocess(
                _with_media(benchmark_example, tmp_path, index),
                cache,
                preprocess_config,
            )
            for index in range(2)
        ]

        first, second = clips
        assert first.audio.transcript != second.audio.transcript
        distance = _phash(first.judge_frame_paths[0]) - _phash(
            second.judge_frame_paths[0]
        )
        assert distance > 8


class TestFakePerceptionTools:
    def test_the_toolset_answers_without_any_model(self, tmp_path: Path) -> None:
        tools = fake_perception_tools()
        wav = tmp_path / 'clip.wav'

        assert tools.transcriber.transcribe(wav).text
        assert tools.prosody.extract(wav) == {}
        assert tools.diarizer.diarize(wav) == ()
