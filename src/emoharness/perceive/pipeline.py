"""Assemble one clip's audio evidence and cache it alongside its sampled frames."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from emoharness.common.hashing import sha256_file_cached
from emoharness.common.logging import get_logger
from emoharness.data.manifest import Example
from emoharness.harness.schema import AudioEvidence
from emoharness.perceive.audio import (
    Diarizer,
    FakeDiarizer,
    FakeProsody,
    FakeTranscriber,
    ProsodyExtractor,
    SpeechSegment,
    Transcriber,
    dominant_speaker,
    subject_transcript,
)
from emoharness.perceive.cache import (
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
)
from emoharness.perceive.descriptors import ReferenceStats, describe_prosody
from emoharness.perceive.video import (
    extract_audio_track,
    extract_frames,
    frame_timestamps,
    judge_frame_timestamps,
    probe_duration,
)

_LOG_EVERY: Final = 10
_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PerceptionTools:
    """Dependency-injection point: the deterministic audio tools, plus config.

    A transcriber, a prosody feature extractor, and a diarizer; nothing here
    forms an opinion about the clip (ADR-0004, amended 2026-09-11).
    """

    transcriber: Transcriber
    prosody: ProsodyExtractor
    diarizer: Diarizer
    reference: ReferenceStats

    @classmethod
    def fakes(
        cls,
        *,
        transcript_text: str = '',
        language: str | None = None,
        prosody_features: Mapping[str, float] | None = None,
        segments: tuple[SpeechSegment, ...] = (),
        reference: ReferenceStats | None = None,
    ) -> PerceptionTools:
        """A fully deterministic toolset for tests; every argument has a safe default."""
        return cls(
            transcriber=FakeTranscriber(transcript_text, language),
            prosody=FakeProsody(prosody_features or {}),
            diarizer=FakeDiarizer(segments),
            reference=reference or ReferenceStats(mean={}, std={}),
        )


@dataclass(frozen=True)
class _AudioBundle:
    evidence: AudioEvidence
    word_count: int


def preprocess_clip(
    example: Example,
    config: PreprocessConfig,
    tools: PerceptionTools,
    cache: PreprocessCache,
) -> PreprocessedClip:
    """Preprocess one clip, reusing a verified cache entry when one already exists."""
    media_path = Path(example.media_path)
    media_hash = sha256_file_cached(media_path)
    cfg_hash = config.config_hash()
    cached = cache.get(media_hash, cfg_hash)
    if cached is not None and cache.verify(cached):
        return cached
    duration = probe_duration(media_path)
    frames_root = cache.frames_dir(media_hash)
    frame_paths = _extract_visual_frames(media_path, duration, config, frames_root)
    judge_paths = _extract_judge_frames(media_path, duration, frames_root)
    audio = _build_audio_evidence(media_path, cache.audio_path(media_hash), tools)
    clip = PreprocessedClip(
        media_hash=media_hash,
        config_hash=cfg_hash,
        duration_seconds=duration,
        frame_paths=frame_paths,
        judge_frame_paths=judge_paths,
        face_crop_paths=(),
        audio=audio.evidence,
        transcript_words=audio.word_count,
    )
    cache.put(clip)
    return clip


def preprocess_many(
    examples: Iterable[Example],
    config: PreprocessConfig,
    tools: PerceptionTools,
    cache: PreprocessCache,
) -> tuple[PreprocessedClip, ...]:
    """Preprocess every example in order, logging progress every few items."""
    items = tuple(examples)
    results: list[PreprocessedClip] = []
    for index, example in enumerate(items, start=1):
        results.append(preprocess_clip(example, config, tools, cache))
        if index % _LOG_EVERY == 0 or index == len(items):
            _LOGGER.info('preprocessed %d/%d clips', index, len(items))
    return tuple(results)


def _extract_visual_frames(
    media_path: Path, duration: float, config: PreprocessConfig, frames_root: Path
) -> tuple[str, ...]:
    timestamps = frame_timestamps(
        duration,
        config.frames.fps,
        config.frames.min_frames,
        config.frames.max_frames,
    )
    paths = extract_frames(media_path, timestamps, frames_root, prefix='frame')
    return tuple(str(p) for p in paths)


def _extract_judge_frames(
    media_path: Path, duration: float, frames_root: Path
) -> tuple[str, ...]:
    timestamps = judge_frame_timestamps(duration)
    paths = extract_frames(media_path, timestamps, frames_root, prefix='judge')
    return tuple(str(p) for p in paths)


def _build_audio_evidence(
    media_path: Path, wav_path: Path, tools: PerceptionTools
) -> _AudioBundle:
    wav = extract_audio_track(media_path, wav_path)
    if wav is None:
        return _AudioBundle(AudioEvidence(speech_present=False), 0)
    transcript = tools.transcriber.transcribe(wav)
    segments = tools.diarizer.diarize(wav)
    speaker = dominant_speaker(segments)
    subject_text = subject_transcript(transcript, segments, speaker)
    descriptors = describe_prosody(tools.prosody.extract(wav), tools.reference)
    evidence = AudioEvidence(
        speech_present=bool(subject_text.strip()) if transcript.available else None,
        transcript=subject_text,
        language=transcript.language,
        prosody=descriptors,
    )
    return _AudioBundle(evidence, len(transcript.words))
