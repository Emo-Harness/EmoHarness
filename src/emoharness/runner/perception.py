"""Building the perception layer and reading back what it cached.

The real adapters are constructed here and nowhere else, so a ``--fake`` run
never imports faster-whisper, openSMILE, or pyannote. Which of them a run uses is
not decided here either: callers pass in a :data:`ClipPreprocessor`, and
``RunContext.clip_preprocessor`` is what chooses. Everything downstream - the
harness, the single-call rows, the leakage pHashes, the judge frames - reads the
same :class:`PreprocessCache` entries, which is what makes the ladder's
"byte-identical inputs" claim true (ADR-0009).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file_cached
from emoharness.common.logging import get_logger
from emoharness.common.parallel import map_ordered, worker_count
from emoharness.data.leakage import phash_image
from emoharness.data.manifest import Example
from emoharness.harness.schema import PerceivedInputs
from emoharness.perceive.audio import (
    Diarizer,
    NullDiarizer,
    OpenSmileProsody,
    PyannoteDiarizer,
    WhisperTranscriber,
)
from emoharness.perceive.cache import (
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
)
from emoharness.perceive.descriptors import ReferenceStats, load_reference
from emoharness.perceive.pipeline import PerceptionTools
from emoharness.runner.paths import ProjectPaths, source_root

#: One clip into the shared cache. The real one drives ffmpeg and the audio
#: models; the fake one writes the same records without opening the video.
ClipPreprocessor = Callable[
    [Example, PreprocessCache, PreprocessConfig], PreprocessedClip
]

#: ``diarization.backend`` values: the pyannote pipeline, or no diarization.
PYANNOTE_BACKEND: Final = 'pyannote'
NO_DIARIZATION_BACKEND: Final = 'none'
PYANNOTE_MODEL: Final = 'pyannote/speaker-diarization-3.1'
#: Preprocessing is one ffmpeg-bound and one model-bound pass per clip, so a
#: couple of clips in flight overlap the two; more just contend for the cores.
_MAX_PREPROCESS_WORKERS: Final = 3
_LOG_EVERY: Final = 25

_LOGGER = get_logger(__name__)


def resolve_input(paths: ProjectPaths, relative: str) -> Path:
    """Resolve a config-declared repository path against the root, then the checkout."""
    candidate = paths.root / relative
    if candidate.exists():
        return candidate
    return source_root() / relative


def load_preprocess_config(paths: ProjectPaths) -> PreprocessConfig:
    """Load ``configs/preprocess.yaml`` verbatim; its hash keys the cache."""
    path = paths.preprocess_config
    if not path.is_file():
        raise ValidationError(f'preprocess config {path} does not exist')
    return load_model(path, PreprocessConfig)


def preprocess_cache(paths: ProjectPaths) -> PreprocessCache:
    """The clip cache under ``data/derived/``."""
    return PreprocessCache(paths.preprocess_cache)


def _reference_stats(paths: ProjectPaths, config: PreprocessConfig) -> ReferenceStats:
    path = resolve_input(paths, config.prosody.reference_stats)
    if not path.is_file():
        _LOGGER.warning('no prosody reference stats at %s; descriptors disabled', path)
        return ReferenceStats(mean={}, std={})
    return load_reference(path)


def real_perception_tools(
    paths: ProjectPaths, config: PreprocessConfig
) -> PerceptionTools:
    """The real audio stack. Every heavy import happens inside the adapters' calls."""
    return PerceptionTools(
        transcriber=WhisperTranscriber(config.asr.model),
        prosody=OpenSmileProsody(),
        diarizer=_diarizer_for(config.diarization.backend),
        reference=_reference_stats(paths, config),
    )


def _diarizer_for(backend: str) -> Diarizer:
    if backend == NO_DIARIZATION_BACKEND:
        return NullDiarizer()
    if backend == PYANNOTE_BACKEND:
        return PyannoteDiarizer(PYANNOTE_MODEL)
    raise ValidationError(
        f'unknown diarization backend {backend!r}; '
        f'expected {PYANNOTE_BACKEND!r} or {NO_DIARIZATION_BACKEND!r}'
    )


def perceived_inputs(
    example: Example,
    cache: PreprocessCache,
    config: PreprocessConfig,
    preprocess: ClipPreprocessor,
) -> PerceivedInputs:
    """Everything the harness and the single-call rows need about one clip."""
    clip = preprocess(example, cache, config)
    return clip.to_perceived(example.example_id, example.media_path)


def lookup_clip(
    example: Example, cache: PreprocessCache, config: PreprocessConfig
) -> PreprocessedClip | None:
    """The cache entry for one example, or None when it has not been preprocessed."""
    try:
        media_hash = sha256_file_cached(Path(example.media_path))
    except OSError:
        _LOGGER.warning('media missing for %s', example.example_id)
        return None
    return cache.get(media_hash, config.config_hash())


def judge_frames_by_id(
    examples: Iterable[Example], cache: PreprocessCache, config: PreprocessConfig
) -> Mapping[str, tuple[str, ...]]:
    """The 1 fps judge frames of every preprocessed example, keyed by example id."""
    found: dict[str, tuple[str, ...]] = {}
    for example in examples:
        clip = lookup_clip(example, cache, config)
        if clip is not None and clip.judge_frame_paths:
            found[example.example_id] = clip.judge_frame_paths
    return found


def phashes_by_id(
    examples: Iterable[Example], cache: PreprocessCache, config: PreprocessConfig
) -> Mapping[str, tuple[str, ...]]:
    """Perceptual hashes of every cached judge frame, for the leakage sweep."""
    return {
        example_id: tuple(phash_image(Path(path)) for path in paths)
        for example_id, paths in judge_frames_by_id(examples, cache, config).items()
    }


def transcripts_by_id(
    examples: Iterable[Example], cache: PreprocessCache, config: PreprocessConfig
) -> Mapping[str, str]:
    """Cached transcripts, for the transcript-overlap leakage check."""
    found: dict[str, str] = {}
    for example in examples:
        clip = lookup_clip(example, cache, config)
        if clip is not None and clip.audio.transcript:
            found[example.example_id] = clip.audio.transcript
    return found


def preprocess_all(
    examples: Sequence[Example],
    cache: PreprocessCache,
    config: PreprocessConfig,
    preprocess: ClipPreprocessor,
) -> tuple[PreprocessedClip, ...]:
    """Preprocess every example, in manifest order, reusing what the cache holds."""
    total = len(examples)
    done = 0
    lock = threading.Lock()

    def one(example: Example) -> PreprocessedClip:
        nonlocal done
        clip = preprocess(example, cache, config)
        with lock:
            done += 1
            if done % _LOG_EVERY == 0 or done == total:
                _LOGGER.info('preprocessed %d/%d clips', done, total)
        return clip

    workers = min(worker_count(), _MAX_PREPROCESS_WORKERS)
    return map_ordered(one, examples, workers)


def verify_cache(
    examples: Sequence[Example], cache: PreprocessCache, config: PreprocessConfig
) -> tuple[str, ...]:
    """Example ids whose cache entry is missing or whose frames no longer exist."""
    broken = []
    for example in examples:
        clip = lookup_clip(example, cache, config)
        if clip is None or not cache.verify(clip):
            broken.append(example.example_id)
    return tuple(broken)


def media_hashes_by_id(examples: Iterable[Example]) -> Mapping[str, str]:
    """SHA-256 of every example's media file; missing files are skipped."""
    found: dict[str, str] = {}
    for example in examples:
        path = Path(example.media_path)
        if path.is_file():
            found[example.example_id] = sha256_file_cached(path)
    return found
