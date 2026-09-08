from __future__ import annotations

import json
from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.perceive.descriptors import (
    ReferenceStats,
    describe_prosody,
    fit_reference,
    load_reference,
    save_reference,
)

_FEATURES = (
    'F0semitoneFrom27.5Hz_sma3nz_amean',
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    'loudness_sma3_amean',
    'VoicedSegmentsPerSec',
    'MeanUnvoicedSegmentLength',
)


def _reference(std: float = 1.0) -> ReferenceStats:
    return ReferenceStats(
        mean={name: 0.0 for name in _FEATURES}, std={name: std for name in _FEATURES}
    )


def test_fit_reference_computes_mean_and_std() -> None:
    rows = [{'a': 1.0, 'b': 10.0}, {'a': 3.0, 'b': 20.0}]
    stats = fit_reference(rows)
    assert stats.mean == {'a': 2.0, 'b': 15.0}
    assert stats.std['a'] == pytest.approx(1.0)
    assert stats.std['b'] == pytest.approx(5.0)


def test_fit_reference_handles_heterogeneous_rows() -> None:
    rows = [{'a': 1.0}, {'a': 3.0, 'b': 100.0}]
    stats = fit_reference(rows)
    assert stats.mean['a'] == pytest.approx(2.0)
    assert stats.mean['b'] == pytest.approx(100.0)
    assert stats.std['b'] == pytest.approx(0.0)


def test_fit_reference_empty_rows_yields_empty_stats() -> None:
    stats = fit_reference([])
    assert stats.mean == {}
    assert stats.std == {}


def test_describe_prosody_high_extreme_phrases_in_declared_order() -> None:
    features = {name: 2.0 for name in _FEATURES}
    assert describe_prosody(features, _reference()) == (
        'high pitch',
        'variable pitch',
        'loud voice',
        'fast speech',
        'long pauses',
    )


def test_describe_prosody_low_extreme_phrases() -> None:
    features = {name: -2.0 for name in _FEATURES}
    assert describe_prosody(features, _reference()) == (
        'low pitch',
        'flat pitch',
        'quiet voice',
        'slow speech',
        'short pauses',
    )


def test_describe_prosody_within_one_z_is_silent() -> None:
    features = {name: 0.5 for name in _FEATURES}
    assert describe_prosody(features, _reference()) == ()


def test_describe_prosody_boundary_z_is_inclusive() -> None:
    features = {name: 1.0 for name in _FEATURES}
    assert len(describe_prosody(features, _reference())) == 5


def test_describe_prosody_skips_features_missing_from_input() -> None:
    features = {'F0semitoneFrom27.5Hz_sma3nz_amean': 5.0}
    assert describe_prosody(features, _reference()) == ('high pitch',)


def test_describe_prosody_skips_unknown_features() -> None:
    features = {'totally_unknown_feature': 999.0}
    assert describe_prosody(features, _reference()) == ()


def test_describe_prosody_skips_zero_std_reference() -> None:
    reference = ReferenceStats(
        mean={'loudness_sma3_amean': 0.0}, std={'loudness_sma3_amean': 0.0}
    )
    features = {'loudness_sma3_amean': 50.0}
    assert describe_prosody(features, reference) == ()


def test_reference_round_trips_through_disk(tmp_path: Path) -> None:
    stats = ReferenceStats(mean={'a': 1.0}, std={'a': 2.0})
    path = tmp_path / 'ref.json'
    save_reference(path, stats)
    assert load_reference(path) == stats


def test_load_reference_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_reference(tmp_path / 'missing.json')


def test_load_reference_schema_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'mean': 'not-a-mapping'}), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_reference(path)
