"""The pre-registered tests, computed from raw predictions.

Every test is reported whatever its outcome, including a missing row: a test that
could not be computed says so rather than disappearing. Unadjusted CIs and
Holm-adjusted bootstrap p-values are reported side by side, as pre-registered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from emoharness.data.tasks import Group
from emoharness.evaluation.bootstrap import BootstrapResult, gap_closed, holm_adjust
from emoharness.runner.items import FULL, SUBSET
from emoharness.runner.results import RowData, RowIndex

#: The pre-registered H3 tiers. They live here rather than in ``common.constants`` because
#: they are reporting thresholds for this analysis, not inference-time protocol.
GAP_TIER_1: Final = 0.25
GAP_TIER_2: Final = 0.5

MISSING: Final = 'missing rows'
NO_ITEMS: Final = 'no shared items'
OK: Final = 'computed'


@dataclass(frozen=True)
class TestSpec:
    """One pre-registered difference test."""

    name: str
    method: str
    baseline: str
    group: Group
    item_set: str
    statement: str


PRE_REGISTERED: Final[tuple[TestSpec, ...]] = (
    TestSpec(
        'H1', 'L5', 'L0', Group.SEEN, FULL, 'Rec-S(L5) - Rec-S(L0) > 0 on seen macro'
    ),
    TestSpec(
        'H2',
        'L5',
        'L0',
        Group.HELD_OUT,
        FULL,
        'Rec-S(L5) - Rec-S(L0) > 0 on held-out macro',
    ),
    TestSpec(
        'H-B_seen',
        'L5',
        'L3',
        Group.SEEN,
        FULL,
        'Rec-S(L5) - Rec-S(L3) > 0 on seen macro',
    ),
    TestSpec(
        'H-B_held_out',
        'L5',
        'L3',
        Group.HELD_OUT,
        FULL,
        'Rec-S(L5) - Rec-S(L3) > 0 on held-out macro',
    ),
)


def _compare(
    method: RowData, baseline: RowData, group: Group | None, resamples: int
) -> BootstrapResult | None:
    """Paired bootstrap of method minus baseline over the items both scored.

    Restricting to ``group`` first drops every item whose task is elsewhere, so a
    comparison with nothing left to pair comes back as None rather than as zero.
    """
    left, right = baseline.scoreboard(), method.scoreboard()
    if group is not None:
        left, right = left.in_group(group), right.in_group(group)
    return left.compare(right, resamples)


def run_test(spec: TestSpec, index: RowIndex, resamples: int) -> dict[str, Any]:
    """One pre-registered difference test, or a record of why it is unavailable."""
    head = {
        'id': spec.name,
        'statement': spec.statement,
        'rows': [spec.baseline, spec.method],
        'item_set': spec.item_set,
        'group': spec.group.value,
    }
    method = index.get(spec.method, spec.item_set)
    baseline = index.get(spec.baseline, spec.item_set)
    if method is None or baseline is None:
        absent = [
            n
            for n, d in ((spec.method, method), (spec.baseline, baseline))
            if d is None
        ]
        return {**head, 'status': MISSING, 'missing': absent}
    result = _compare(method, baseline, spec.group, resamples)
    if result is None:
        return {**head, 'status': NO_ITEMS}
    return {
        **head,
        'status': OK,
        'source': method.source,
        'estimate': result.estimate,
        'ci_low': result.ci_low,
        'ci_high': result.ci_high,
        'p_value': result.p_value,
        'n_items': result.n_items,
        'ci_excludes_zero': result.ci_low > 0.0 or result.ci_high < 0.0,
    }


def _tier(value: float) -> int:
    if value >= GAP_TIER_2:
        return 2
    if value >= GAP_TIER_1:
        return 1
    return 0


def gap_closed_test(index: RowIndex, resamples: int) -> dict[str, Any]:
    """H3: how much of the teacher-minus-baseline gap the harness closes."""
    head = {
        'id': 'H3',
        'statement': 'Gap Closed = (Rec-S(L5) - Rec-S(L0)) / (Rec-S(L6) - Rec-S(L0))',
        'rows': ['L0', 'L5', 'L6'],
        'item_set': SUBSET,
        'group': 'overall',
        'tiers': {'tier_1': GAP_TIER_1, 'tier_2': GAP_TIER_2},
    }
    found = {name: index.get(name, SUBSET) for name in ('L0', 'L5', 'L6')}
    absent = [name for name, data in found.items() if data is None]
    if absent:
        return {**head, 'status': MISSING, 'missing': absent}
    rows = {name: data for name, data in found.items() if data is not None}
    macros = {name: data.scoreboard().macro() for name, data in rows.items()}
    base, method, upper = macros['L0'], macros['L5'], macros['L6']
    if base is None or method is None or upper is None:
        return {**head, 'status': NO_ITEMS}
    ratio = gap_closed(base, method, upper)
    body = _gap_body(rows['L5'], rows['L0'], ratio, upper - base, resamples)
    return {**head, **body}


def _gap_body(
    method: RowData,
    baseline: RowData,
    ratio: float | None,
    span: float,
    resamples: int,
) -> dict[str, Any]:
    result = _compare(method, baseline, None, resamples)
    if result is None:
        return {'status': NO_ITEMS}
    scale = span if span > 0 else None
    return {
        'status': OK,
        'source': method.source,
        'gap_closed': ratio,
        'tier_reached': _tier(ratio) if ratio is not None else None,
        'gain_estimate': result.estimate,
        'gain_ci_low': result.ci_low,
        'gain_ci_high': result.ci_high,
        'gap_ci_low': result.ci_low / scale if scale else None,
        'gap_ci_high': result.ci_high / scale if scale else None,
        'p_value': result.p_value,
        'n_items': result.n_items,
    }


def hypotheses_report(index: RowIndex, resamples: int) -> Mapping[str, Any]:
    """Every pre-registered test plus the Holm-adjusted p-values across the family."""
    tests: list[dict[str, Any]] = [
        run_test(spec, index, resamples) for spec in PRE_REGISTERED
    ]
    tests.append(gap_closed_test(index, resamples))
    return {
        'family': [test['id'] for test in tests],
        'holm_adjusted': dict(_holm(tests)),
        'tests': tests,
    }


def _holm(tests: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    p_values = {
        str(test['id']): float(test['p_value'])
        for test in tests
        if test.get('status') == OK and 'p_value' in test
    }
    return holm_adjust(p_values) if p_values else {}
