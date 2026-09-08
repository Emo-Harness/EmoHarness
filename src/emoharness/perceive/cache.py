"""Preprocessing configuration and the content-addressed cache of clip evidence.

Storage layout: ``<root>/<media_hash>/<config_hash>.json`` for the cached record,
``<root>/<media_hash>/frames/`` for extracted JPEGs, and
``<root>/<media_hash>/audio.wav`` for the extracted audio track.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import stable_hash
from emoharness.common.io import read_json, write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.harness.schema import AudioEvidence, AudioOpinion, PerceivedInputs

_LOGGER = get_logger(__name__)


class FrameConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    fps: float = Field(gt=0.0)
    min_frames: int = Field(ge=1)
    max_frames: int = Field(ge=1)
    judge_fps: float = Field(gt=0.0)
    judge_max_frames: int = Field(ge=1)

    @model_validator(mode='after')
    def _check_bounds(self) -> FrameConfig:
        if self.max_frames < self.min_frames:
            raise ValueError('frames.max_frames must be >= frames.min_frames')
        return self


class AsrConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    word_timestamps: bool = True


class ProsodyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    feature_set: str = Field(min_length=1)
    reference_stats: str = Field(min_length=1)


class DiarizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    backend: str = Field(min_length=1)
    subject_rule: str = Field(min_length=1)


class DescriberEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', protected_namespaces=())

    model_id: str = Field(min_length=1)
    mode: str = 'audio_only'


class DescriberConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    primary: DescriberEndpoint
    fallback: DescriberEndpoint
    lexicon_path: str = Field(min_length=1)


class PreprocessConfig(BaseModel):
    """Mirrors ``configs/preprocess.yaml``; its hash keys the preprocessing cache."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    frames: FrameConfig
    asr: AsrConfig
    prosody: ProsodyConfig
    diarization: DiarizationConfig
    describer: DescriberConfig
    cache_root: str = Field(min_length=1)

    def config_hash(self) -> str:
        """Stable hash of every setting that can change cached preprocessing output."""
        return stable_hash(self.model_dump(mode='json'))


class PreprocessedClip(BaseModel):
    """One clip's cached preprocessing output; everything the harness needs about it."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    media_hash: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.0)
    frame_paths: tuple[str, ...] = ()
    judge_frame_paths: tuple[str, ...] = ()
    face_crop_paths: tuple[str, ...] = ()
    audio: AudioEvidence
    audio_opinion: AudioOpinion | None = None
    transcript_words: int = Field(ge=0)

    def to_perceived(self, example_id: str, video_path: str) -> PerceivedInputs:
        """Project this cache entry into the ``PerceivedInputs`` the harness consumes."""
        return PerceivedInputs(
            example_id=example_id,
            video_path=video_path,
            duration_seconds=self.duration_seconds,
            frame_paths=self.frame_paths,
            judge_frame_paths=self.judge_frame_paths,
            face_crop_paths=self.face_crop_paths,
            audio=self.audio,
            audio_opinion=self.audio_opinion,
        )


class PreprocessCache:
    """Filesystem-backed cache of :class:`PreprocessedClip` records."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, media_hash: str, config_hash: str) -> Path:
        return self._root / media_hash / f'{config_hash}.json'

    def frames_dir(self, media_hash: str) -> Path:
        return self._root / media_hash / 'frames'

    def audio_path(self, media_hash: str) -> Path:
        return self._root / media_hash / 'audio.wav'

    def get(self, media_hash: str, config_hash: str) -> PreprocessedClip | None:
        """The cached entry, or ``None`` if it does not exist. Raises on corruption."""
        path = self.path_for(media_hash, config_hash)
        if not path.exists():
            return None
        try:
            return PreprocessedClip.model_validate(read_json(path))
        except PydanticValidationError as exc:
            raise ValidationError(f'corrupt cache entry {path}: {exc}') from exc

    def put(self, clip: PreprocessedClip) -> None:
        """Write ``clip`` to its content-addressed path, creating directories."""
        path = self.path_for(clip.media_hash, clip.config_hash)
        write_json_atomic(path, clip.model_dump(mode='json'))

    def verify(self, clip: PreprocessedClip) -> bool:
        """True when every frame file the entry references still exists on disk."""
        all_paths = clip.frame_paths + clip.judge_frame_paths + clip.face_crop_paths
        return all(Path(p).exists() for p in all_paths)

    def iter_entries(self) -> Iterator[PreprocessedClip]:
        """Yield every cached entry under root, skipping unreadable ones with a warning."""
        if not self._root.exists():
            return
        for path in sorted(self._root.glob('*/*.json')):
            try:
                yield PreprocessedClip.model_validate(read_json(path))
            except (ValidationError, PydanticValidationError) as exc:
                _LOGGER.warning('skipping corrupt cache entry %s: %s', path, exc)
