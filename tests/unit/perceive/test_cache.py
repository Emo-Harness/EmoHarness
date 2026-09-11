from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.harness.schema import AudioEvidence, PerceivedInputs
from emoharness.perceive.cache import (
    FrameConfig,
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
)


def _clip(
    *,
    media_hash: str = 'abc123',
    config_hash: str = 'cfg456',
    duration_seconds: float = 4.0,
    frame_paths: tuple[str, ...] = ('f1.jpg', 'f2.jpg'),
    judge_frame_paths: tuple[str, ...] = ('j1.jpg',),
    face_crop_paths: tuple[str, ...] = (),
    audio: AudioEvidence | None = None,
    transcript_words: int = 1,
) -> PreprocessedClip:
    return PreprocessedClip(
        media_hash=media_hash,
        config_hash=config_hash,
        duration_seconds=duration_seconds,
        frame_paths=frame_paths,
        judge_frame_paths=judge_frame_paths,
        face_crop_paths=face_crop_paths,
        audio=audio
        if audio is not None
        else AudioEvidence(speech_present=True, transcript='hi'),
        transcript_words=transcript_words,
    )


# --- PreprocessConfig --------------------------------------------------------------


def test_preprocess_config_loads_real_config_file(repo_root: Path) -> None:
    config = load_model(repo_root / 'configs' / 'preprocess.yaml', PreprocessConfig)
    assert config.frames == FrameConfig(
        fps=2.0, min_frames=4, max_frames=32, judge_fps=1.0, judge_max_frames=10
    )
    assert config.asr.model == 'large-v3'
    assert config.asr.backend == 'faster_whisper'
    assert config.prosody.feature_set == 'eGeMAPSv02'
    # This run: no diarizer, and no audio-language model at all (ADR-0004
    # amendment of 2026-09-11): every tool in the config is deterministic code.
    assert config.diarization.backend == 'none'
    assert not hasattr(config, 'describer')
    assert config.cache_root == 'data/derived'


def test_preprocess_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_model(tmp_path / 'missing.yaml', PreprocessConfig)


def test_preprocess_config_schema_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / 'bad.yaml'
    path.write_text('frames: {fps: 2.0}\n', encoding='utf-8')
    with pytest.raises(ValidationError):
        load_model(path, PreprocessConfig)


def test_frame_config_rejects_max_below_min() -> None:
    with pytest.raises(PydanticValidationError):
        FrameConfig(
            fps=2.0, min_frames=8, max_frames=4, judge_fps=1.0, judge_max_frames=10
        )


def test_config_hash_is_stable_and_sensitive_to_changes(repo_root: Path) -> None:
    config = load_model(repo_root / 'configs' / 'preprocess.yaml', PreprocessConfig)
    changed_frames = config.frames.model_copy(update={'fps': 3.0})
    other = config.model_copy(update={'frames': changed_frames})
    assert config.config_hash() == config.config_hash()
    assert config.config_hash() != other.config_hash()


# --- PreprocessedClip.to_perceived --------------------------------------------------


def test_to_perceived_maps_fields_and_drops_cache_only_fields() -> None:
    clip = _clip()
    perceived = clip.to_perceived('ex1', '/videos/ex1.mp4')
    assert perceived == PerceivedInputs(
        example_id='ex1',
        video_path='/videos/ex1.mp4',
        duration_seconds=4.0,
        frame_paths=('f1.jpg', 'f2.jpg'),
        judge_frame_paths=('j1.jpg',),
        face_crop_paths=(),
        audio=clip.audio,
    )


# --- PreprocessCache -----------------------------------------------------------------


def test_path_for_frames_dir_and_audio_path_layout(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    assert cache.path_for('h1', 'c1') == tmp_path / 'h1' / 'c1.json'
    assert cache.frames_dir('h1') == tmp_path / 'h1' / 'frames'
    assert cache.audio_path('h1') == tmp_path / 'h1' / 'audio.wav'


def test_get_missing_entry_is_none(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    assert cache.get('nope', 'nope') is None


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    clip = _clip()
    cache.put(clip)
    assert cache.get(clip.media_hash, clip.config_hash) == clip
    assert cache.path_for(clip.media_hash, clip.config_hash).exists()


def test_get_corrupt_json_raises(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    path = cache.path_for('h1', 'c1')
    path.parent.mkdir(parents=True)
    path.write_text('not json{{{', encoding='utf-8')
    with pytest.raises(ValidationError):
        cache.get('h1', 'c1')


def test_get_schema_mismatch_raises(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    path = cache.path_for('h1', 'c1')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'not': 'a clip'}), encoding='utf-8')
    with pytest.raises(ValidationError):
        cache.get('h1', 'c1')


def test_verify_true_when_all_referenced_files_exist(tmp_path: Path) -> None:
    frame = tmp_path / 'f1.jpg'
    frame.write_bytes(b'x')
    clip = _clip(frame_paths=(str(frame),), judge_frame_paths=(), face_crop_paths=())
    cache = PreprocessCache(tmp_path)
    assert cache.verify(clip) is True


def test_verify_false_when_a_referenced_file_is_missing(tmp_path: Path) -> None:
    clip = _clip(frame_paths=(str(tmp_path / 'missing.jpg'),))
    cache = PreprocessCache(tmp_path)
    assert cache.verify(clip) is False


def test_verify_true_when_no_files_are_referenced(tmp_path: Path) -> None:
    clip = _clip(frame_paths=(), judge_frame_paths=(), face_crop_paths=())
    cache = PreprocessCache(tmp_path)
    assert cache.verify(clip) is True


def test_iter_entries_yields_every_cached_clip(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path)
    clip_a = _clip(media_hash='hash-a')
    clip_b = _clip(media_hash='hash-b', config_hash='cfg-other')
    cache.put(clip_a)
    cache.put(clip_b)
    entries = {(c.media_hash, c.config_hash) for c in cache.iter_entries()}
    assert entries == {('hash-a', 'cfg456'), ('hash-b', 'cfg-other')}


def test_iter_entries_on_missing_root_yields_nothing(tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path / 'does-not-exist-yet')
    assert list(cache.iter_entries()) == []


def test_iter_entries_skips_corrupt_files_and_logs_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache = PreprocessCache(tmp_path)
    cache.put(_clip(media_hash='good-hash'))
    bad_path = cache.path_for('bad-hash', 'cfg456')
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text('not json', encoding='utf-8')

    logger = logging.getLogger('emoharness.perceive.cache')
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger='emoharness.perceive.cache'):
            entries = list(cache.iter_entries())
    finally:
        logger.removeHandler(caplog.handler)

    assert [entry.media_hash for entry in entries] == ['good-hash']
    assert any('bad-hash' in message for message in caplog.messages)
