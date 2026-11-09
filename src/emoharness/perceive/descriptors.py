"""Turn openSMILE eGeMAPS functionals into short, corpus-relative prosody phrases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json, write_json_atomic

#: z-score magnitude beyond which a feature is called out. Named per engineering.md
#: ("no hardcoded thresholds outside constants/configs"); this module owns it because
#: it is purely a describer presentation rule, not a benchmark protocol fact.
Z_THRESHOLD: Final = 1.0

#: eGeMAPS functional -> (phrase when z <= -Z_THRESHOLD, phrase when z >= +Z_THRESHOLD).
#: Order is preserved in ``describe_prosody``'s output.
_FEATURE_PHRASES: Final[Mapping[str, tuple[str, str]]] = {
    'F0semitoneFrom27.5Hz_sma3nz_amean': ('low pitch', 'high pitch'),
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': ('flat pitch', 'variable pitch'),
    'loudness_sma3_amean': ('quiet voice', 'loud voice'),
    'VoicedSegmentsPerSec': ('slow speech', 'fast speech'),
    'MeanUnvoicedSegmentLength': ('short pauses', 'long pauses'),
}


class ReferenceStats(BaseModel):
    """Per-feature corpus mean and standard deviation used to compute z-scores."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    mean: Mapping[str, float]
    std: Mapping[str, float]


def fit_reference(rows: Iterable[Mapping[str, float]]) -> ReferenceStats:
    """Per-feature mean and population std over ``rows``; missing features are skipped.

    Each feature's statistics are computed only from the rows that contain it, so
    heterogeneous feature sets across rows are handled without error.
    """
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(float(value))
    mean = {key: float(np.mean(values)) for key, values in columns.items()}
    std = {key: float(np.std(values)) for key, values in columns.items()}
    return ReferenceStats(mean=mean, std=std)


def describe_prosody(
    features: Mapping[str, float], reference: ReferenceStats
) -> tuple[str, ...]:
    """Short phrases for the eGeMAPS functionals whose z-score exceeds ``±Z_THRESHOLD``.

    Unknown or missing features (absent from ``features`` or from ``reference``, or
    with zero reference std) are silently skipped rather than raising.
    """
    phrases: list[str] = []
    for name, (low_phrase, high_phrase) in _FEATURE_PHRASES.items():
        phrase = _phrase_for(name, low_phrase, high_phrase, features, reference)
        if phrase is not None:
            phrases.append(phrase)
    return tuple(phrases)


def _phrase_for(
    name: str,
    low_phrase: str,
    high_phrase: str,
    features: Mapping[str, float],
    reference: ReferenceStats,
) -> str | None:
    if name not in features or name not in reference.mean or name not in reference.std:
        return None
    std = reference.std[name]
    if std <= 0:
        return None
    z = (features[name] - reference.mean[name]) / std
    if z >= Z_THRESHOLD:
        return high_phrase
    if z <= -Z_THRESHOLD:
        return low_phrase
    return None


def load_reference(path: Path) -> ReferenceStats:
    """Load reference statistics written by :func:`save_reference`."""
    data = read_json(path)
    try:
        return ReferenceStats.model_validate(data)
    except PydanticValidationError as exc:
        raise ValidationError(f'invalid reference stats at {path}: {exc}') from exc


def save_reference(path: Path, stats: ReferenceStats) -> None:
    """Write reference statistics atomically as JSON."""
    write_json_atomic(path, stats.model_dump(mode='json'))
