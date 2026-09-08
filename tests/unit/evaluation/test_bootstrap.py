from __future__ import annotations

import pytest

from emoharness.common.errors import ValidationError
from emoharness.data.tasks import Kind, Task
from emoharness.evaluation.bootstrap import (
    _aggregate,
    detectable_effect,
    gap_closed,
    holm_adjust,
    paired_bootstrap,
)

_TASKS = (Task.ER_LAB, Task.ER_WILD, Task.SA, Task.IR)


def test_paired_bootstrap_constant_difference_is_exact() -> None:
    strata = {f'item{i}': _TASKS[i % 4] for i in range(100)}
    a = dict.fromkeys(strata, 0.0)
    b = dict.fromkeys(strata, 1.0)
    result = paired_bootstrap(a, b, strata, n_resamples=500, seed=0)
    assert result.estimate == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)
    assert result.p_value == 0.0
    assert result.n_items == 100


def test_paired_bootstrap_negative_difference_is_also_significant() -> None:
    strata = {f'item{i}': _TASKS[i % 4] for i in range(100)}
    a = dict.fromkeys(strata, 1.0)
    b = dict.fromkeys(strata, 0.0)
    result = paired_bootstrap(a, b, strata, n_resamples=500, seed=0)
    assert result.estimate == pytest.approx(-1.0)
    assert result.p_value == 0.0


def test_paired_bootstrap_no_effect_gives_large_p_value() -> None:
    strata = {f'item{i}': _TASKS[i % 4] for i in range(80)}
    a = {item_id: 0.5 for item_id in strata}
    b = dict(a)
    result = paired_bootstrap(a, b, strata, n_resamples=500, seed=0)
    assert result.estimate == 0.0
    assert result.p_value == 1.0
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)


def test_paired_bootstrap_ci_contains_the_known_synthetic_mean() -> None:
    strata = {f'item{i}': _TASKS[i % 4] for i in range(200)}
    a = dict.fromkeys(strata, 0.0)
    b = {
        item_id: 0.4 + (0.1 if i % 2 == 0 else -0.1) for i, item_id in enumerate(strata)
    }
    result = paired_bootstrap(a, b, strata, n_resamples=3000, seed=0)
    assert result.estimate == pytest.approx(0.4)
    assert result.ci_low <= 0.4 <= result.ci_high
    assert result.p_value < 0.01


def test_paired_bootstrap_macro_vs_pooled_differ_under_imbalance() -> None:
    strata = {f'a{i}': Task.ER_LAB for i in range(10)}
    strata.update({f'b{i}': Task.SA for i in range(90)})
    a = dict.fromkeys(strata, 0.0)
    b = {item_id: (1.0 if item_id.startswith('a') else 0.0) for item_id in strata}

    macro = paired_bootstrap(a, b, strata, n_resamples=200, seed=1, macro=True)
    pooled = paired_bootstrap(a, b, strata, n_resamples=200, seed=1, macro=False)
    assert macro.estimate == pytest.approx(0.5)
    assert pooled.estimate == pytest.approx(0.1)


def test_paired_bootstrap_requires_shared_items() -> None:
    with pytest.raises(ValidationError):
        paired_bootstrap({'x': 1.0}, {'y': 1.0}, {'x': Task.SA, 'y': Task.SA})


def test_paired_bootstrap_accepts_any_hashable_stratum() -> None:
    """Construction data strata by capability kind, and a pooled key by nothing."""
    strata = {f'item{i}': Kind.LAB_ER if i % 2 else Kind.SENTIMENT for i in range(40)}
    a = dict.fromkeys(strata, 0.0)
    b = dict.fromkeys(strata, 0.5)
    by_kind = paired_bootstrap(a, b, strata, n_resamples=200, seed=0)
    pooled = paired_bootstrap(
        a, b, dict.fromkeys(strata, 'pooled'), n_resamples=200, seed=0
    )
    assert by_kind.estimate == pytest.approx(0.5)
    assert pooled.estimate == pytest.approx(0.5)


def test_holm_adjust_known_vector() -> None:
    p_values = {'a': 0.01, 'b': 0.02, 'c': 0.03, 'd': 0.04}
    adjusted = holm_adjust(p_values)
    assert adjusted['a'] == pytest.approx(0.04)
    assert adjusted['b'] == pytest.approx(0.06)
    assert adjusted['c'] == pytest.approx(0.06)
    assert adjusted['d'] == pytest.approx(0.06)


def test_holm_adjust_caps_at_one_and_stays_monotone() -> None:
    adjusted = holm_adjust({'x': 0.9, 'y': 0.95})
    assert adjusted['x'] == 1.0
    assert adjusted['y'] == 1.0


def test_detectable_effect_matches_adr_0013_figures() -> None:
    assert detectable_effect(3_900, 0.30) == pytest.approx(2.5, abs=0.1)
    assert detectable_effect(2_600, 0.30) == pytest.approx(3.0, abs=0.15)
    assert detectable_effect(1_000, 0.30) == pytest.approx(5.0, abs=0.2)
    assert detectable_effect(125, 0.30) == pytest.approx(14.0, abs=0.5)
    with pytest.raises(ValidationError):
        detectable_effect(0, 0.30)


def test_gap_closed() -> None:
    assert gap_closed(0.0, 0.5, 1.0) == 0.5
    assert gap_closed(0.5, 0.5, 0.5) is None
    assert gap_closed(0.5, 0.6, 0.4) is None


def test_aggregate_with_no_tasks_is_zero() -> None:
    """Defensive branch: `_aggregate` is total on an empty mapping, unlike `paired_bootstrap`."""
    assert _aggregate({}, macro=True) == 0.0
    assert _aggregate({}, macro=False) == 0.0
