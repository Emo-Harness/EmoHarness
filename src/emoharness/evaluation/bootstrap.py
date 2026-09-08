"""Paired bootstrap CIs, Holm correction, and pre-registered detectable effects (ADR-0013)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.errors import ValidationError

FloatArray = NDArray[np.float64]
Stratum = Hashable


class BootstrapResult(BaseModel):
    """95% paired bootstrap CI and two-sided p-value for one difference (ADR-0013 §2)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float = Field(ge=0.0, le=1.0)
    n_resamples: int = Field(gt=0)
    n_items: int = Field(gt=0)


def paired_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    strata: Mapping[str, Stratum],
    n_resamples: int = 10_000,
    seed: int = 0,
    macro: bool = True,
) -> BootstrapResult:
    """95% CI and two-sided p-value for the item-paired difference b - a.

    Item ids shared by `a` and `b` are grouped by `strata`; every resample
    draws, independently within each stratum, as many ids with replacement as
    that stratum has, so stratum proportions are preserved (ADR-0013 §2). A
    stratum is a task on the benchmark and a capability kind on construction
    data; any hashable key works, and one pooled key reduces this to an
    unstratified paired bootstrap. Each resample's statistic is the macro
    average of per-stratum means (`macro=True`, matching the primary metric's
    definition) or the pooled mean over every resampled item (`macro=False`).
    The two-sided p-value is the fraction of resamples whose sign is opposite
    the point estimate, doubled and capped at 1.
    """
    shared_ids = tuple(item_id for item_id in a if item_id in b)
    if not shared_ids:
        raise ValidationError('paired_bootstrap requires an item id shared by a and b')

    diffs: dict[Stratum, list[float]] = {}
    for item_id in shared_ids:
        diffs.setdefault(strata[item_id], []).append(b[item_id] - a[item_id])
    arrays = {
        stratum: np.asarray(values, dtype=float) for stratum, values in diffs.items()
    }

    estimate = _aggregate(arrays, macro)
    rng = np.random.default_rng(seed)
    resamples = np.fromiter(
        (_aggregate(_resample(arrays, rng), macro) for _ in range(n_resamples)),
        dtype=float,
        count=n_resamples,
    )
    ci_low, ci_high = (float(x) for x in np.percentile(resamples, [2.5, 97.5]))
    return BootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=_two_sided_p_value(resamples, estimate),
        n_resamples=n_resamples,
        n_items=len(shared_ids),
    )


def _resample(
    arrays: Mapping[Stratum, FloatArray], rng: np.random.Generator
) -> dict[Stratum, FloatArray]:
    return {
        stratum: values[rng.integers(0, len(values), size=len(values))]
        for stratum, values in arrays.items()
    }


def _aggregate(by_stratum: Mapping[Stratum, FloatArray], macro: bool) -> float:
    non_empty = [values for values in by_stratum.values() if values.size]
    if not non_empty:
        return 0.0
    if macro:
        return float(np.mean([float(values.mean()) for values in non_empty]))
    return float(np.concatenate(non_empty).mean())


def _two_sided_p_value(resamples: FloatArray, estimate: float) -> float:
    if estimate > 0:
        opposite = float(np.mean(resamples <= 0))
    elif estimate < 0:
        opposite = float(np.mean(resamples >= 0))
    else:
        return 1.0
    return float(min(1.0, 2.0 * opposite))


def holm_adjust(p_values: Mapping[str, float]) -> Mapping[str, float]:
    """Holm-Bonferroni step-down adjustment (ADR-0013 §6); monotone by construction."""
    names = list(p_values)
    order = sorted(range(len(names)), key=lambda i: p_values[names[i]])
    total = len(names)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, index in enumerate(order):
        name = names[index]
        running_max = max(running_max, min(1.0, (total - rank) * p_values[name]))
        adjusted[name] = running_max
    return adjusted


def detectable_effect(n_items: int, discordance: float, power_z: float = 2.8) -> float:
    """Minimum detectable paired accuracy difference, in points (ADR-0013 §4)."""
    if n_items <= 0:
        raise ValidationError('detectable_effect requires a positive item count')
    return power_z * math.sqrt(discordance / n_items) * 100.0


def gap_closed(base: float, method: float, upper: float) -> float | None:
    """Gap Closed = (method - base) / (upper - base); None when the denominator isn't positive.

    ADR-0012 H3 defines this only when the upper bound beats the baseline.
    """
    denominator = upper - base
    if denominator <= 0:
        return None
    return (method - base) / denominator
