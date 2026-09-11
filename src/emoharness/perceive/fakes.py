"""Perception fakes: cached clips without ffmpeg, GPUs, or audio models.

The doubles for this package's two seams sit here beside the real adapters:
``fake_perception_tools`` stands in for the audio models :func:`preprocess_clip`
drives, and ``fake_preprocess`` stands in for that whole function. Both fill the
same :class:`PreprocessCache` the real pipeline fills, so every downstream
consumer (harness, baselines, judge frames, leakage pHashes) reads its evidence
through the ordinary code path. Frames are real JPEGs written with Pillow: small,
blocky, and different per clip, so perceptual hashing over them behaves like it
does on real footage.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final

from emoharness.common.hashing import sha256_file_cached
from emoharness.data.manifest import Example
from emoharness.harness.schema import AudioEvidence
from emoharness.perceive.cache import (
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
)
from emoharness.perceive.descriptors import ReferenceStats
from emoharness.perceive.pipeline import PerceptionTools

FAKE_FRAMES: Final = 3
FAKE_JUDGE_FRAMES: Final = 2
FRAME_BLOCKS: Final = 8
FRAME_SIZE: Final = 64

# Deliberately free of emotion words: the transcript is evidence, never a label
# hint (ADR-0004).
_SUBJECTS: Final[tuple[str, ...]] = (
    'the speaker',
    'the woman on the left',
    'the man at the desk',
    'the caller',
    'the presenter',
    'the guest',
)
_PREDICATES: Final[tuple[str, ...]] = (
    'says the meeting starts soon',
    'asks about the schedule',
    'reads the number out loud',
    'repeats the address',
    'mentions the delivery',
    'describes the room',
    'counts the boxes',
    'names the street',
)
#: Neutral filler drawn per clip. The pool is wide on purpose: two clips must not
#: look like transcript near-duplicates to the leakage check, which flags a token
#: Jaccard of 0.8 or more between a construction clip and a benchmark clip.
_FILLER: Final[tuple[str, ...]] = (
    'table',
    'window',
    'ticket',
    'garden',
    'letter',
    'bicycle',
    'morning',
    'harbour',
    'ladder',
    'kettle',
    'notebook',
    'pavement',
    'orchard',
    'lantern',
    'compass',
    'satchel',
    'balcony',
    'crossing',
    'terminal',
    'workshop',
    'cabinet',
    'railing',
    'platform',
    'envelope',
    'doorway',
    'corridor',
    'staircase',
    'courtyard',
    'blanket',
    'saucer',
    'basket',
    'canvas',
    'drawer',
    'shelf',
    'timber',
    'gravel',
    'meadow',
    'ribbon',
    'anchor',
    'beacon',
    'canyon',
    'furnace',
    'granite',
    'harvest',
    'island',
    'jacket',
    'kernel',
    'lantern-post',
    'marble',
    'needle',
    'octave',
    'parcel',
    'quarry',
    'rooftop',
    'sandal',
    'thimble',
    'utensil',
    'vessel',
    'wagon',
    'yardstick',
)
FILLER_WORDS: Final = 8
_PROSODY: Final[tuple[str, ...]] = (
    'high pitch',
    'low pitch',
    'loud voice',
    'quiet voice',
    'fast speech',
    'slow speech',
    'long pauses',
)


def clip_seed(media_hash: str) -> int:
    """A small integer derived from a media hash, stable across machines."""
    return int(media_hash[:12], 16)


def write_frame(path: Path, seed: int) -> Path:
    """Write one blocky JPEG whose pattern is a function of ``seed``."""
    if path.exists():
        return path
    from PIL import Image

    rng = random.Random(seed)  # noqa: S311 -- reproducible pattern, not crypto
    pixels = bytes(rng.randrange(256) for _ in range(FRAME_BLOCKS * FRAME_BLOCKS))
    small = Image.frombytes('L', (FRAME_BLOCKS, FRAME_BLOCKS), pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    small.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.NEAREST).save(
        path, quality=95
    )
    return path


def _frames(frames_dir: Path, prefix: str, seed: int, count: int) -> tuple[str, ...]:
    return tuple(
        str(write_frame(frames_dir / f'{prefix}_{index:04d}.jpg', seed + index * 977))
        for index in range(count)
    )


def fake_transcript(seed: int) -> str:
    """A deterministic, emotion-free line whose token set is unique to one clip."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible wording, not crypto
    subject = _SUBJECTS[seed % len(_SUBJECTS)]
    predicate = _PREDICATES[(seed // 7) % len(_PREDICATES)]
    filler = ' '.join(rng.sample(_FILLER, FILLER_WORDS))
    return f'{subject} {predicate}: {filler} at minute {seed % 53}.'


def fake_audio(example: Example, seed: int) -> AudioEvidence:
    """Observable-cue evidence for one clip, distinct per clip and label-free."""
    transcript = fake_transcript(seed)
    prosody = (
        _PROSODY[seed % len(_PROSODY)],
        _PROSODY[(seed // 11) % len(_PROSODY)],
    )
    return AudioEvidence(
        speech_present=True,
        transcript=transcript,
        language=example.language or 'en',
        prosody=tuple(dict.fromkeys(prosody)),
    )


def fake_preprocess(
    example: Example, cache: PreprocessCache, config: PreprocessConfig
) -> PreprocessedClip:
    """Cache one clip's evidence without ffmpeg, reusing a verified entry.

    The cache key is the real one - the media SHA-256 and the config hash - so a
    ``--fake`` cache and a real cache never collide and ``preprocess.py --verify``
    behaves the same way in both modes.
    """
    media_hash = sha256_file_cached(Path(example.media_path))
    config_hash = config.config_hash()
    cached = cache.get(media_hash, config_hash)
    if cached is not None and cache.verify(cached):
        return cached
    seed = clip_seed(media_hash)
    frames_dir = cache.frames_dir(media_hash)
    audio = fake_audio(example, seed)
    clip = PreprocessedClip(
        media_hash=media_hash,
        config_hash=config_hash,
        duration_seconds=float(2 + seed % 7),
        frame_paths=_frames(frames_dir, 'frame', seed, FAKE_FRAMES),
        judge_frame_paths=_frames(frames_dir, 'judge', seed + 1, FAKE_JUDGE_FRAMES),
        face_crop_paths=(),
        audio=audio,
        transcript_words=len(audio.transcript.split()),
    )
    cache.put(clip)
    return clip


def fake_perception_tools() -> PerceptionTools:
    """Deterministic audio tools, for the real pipeline run without models."""
    return PerceptionTools.fakes(
        transcript_text=fake_transcript(0),
        language='en',
        prosody_features={},
        reference=ReferenceStats(mean={}, std={}),
    )
