"""Reuse code-produced cache artifacts without retaining model-derived speech."""

from __future__ import annotations

from emoharness.common.errors import ValidationError
from emoharness.perceive.cache import PreprocessConfig, PreprocessedClip


def without_asr(
    clip: PreprocessedClip,
    source: PreprocessConfig,
    target: PreprocessConfig,
    *,
    audio_present: bool,
) -> PreprocessedClip:
    """Project only compatible caches; no model or media processing occurs here."""
    if target.asr.backend != 'none' or source.diarization.backend != 'none':
        raise ValidationError(
            'projection requires a code-only target and no source diarizer'
        )
    source_settings = source.model_dump(mode='json', exclude={'asr'})
    target_settings = target.model_dump(mode='json', exclude={'asr'})
    if source_settings != target_settings or clip.config_hash != source.config_hash():
        raise ValidationError('projection source or non-ASR settings do not match')
    audio = clip.audio.model_copy(
        update={
            'speech_present': None if audio_present else False,
            'transcript': '',
            'language': None,
        }
    )
    return clip.model_copy(
        update={
            'config_hash': target.config_hash(),
            'audio': audio,
            'transcript_words': 0,
        }
    )
