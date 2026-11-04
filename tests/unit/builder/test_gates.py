from __future__ import annotations

from collections.abc import Callable

from emoharness.builder.gates import GateConfig, evaluate_gates, select_candidate
from emoharness.builder.schema import CandidateScores

Scores = Callable[..., CandidateScores]
CFG = GateConfig(
    regression_rate_limit=0.05,
    reasoning_margin_points=2.0,
    max_mean_calls=4.0,
    max_p95_latency_s=10.0,
)


def test_a_strictly_better_candidate_passes(make_scores: Scores) -> None:
    result = evaluate_gates(make_scores('c', recognition=0.6), make_scores('i'), CFG)
    assert result.passed is True
    assert result.reasons == ()
    assert result.name == 'c'


def test_equal_recognition_is_rejected(make_scores: Scores) -> None:
    result = evaluate_gates(make_scores('c', recognition=0.5), make_scores('i'), CFG)
    assert result.passed is False
    assert 'does not exceed incumbent' in result.reasons[0]


def test_min_recognition_gain_is_applied(make_scores: Scores) -> None:
    cfg = CFG.model_copy(update={'min_recognition_gain': 0.02})
    small = evaluate_gates(make_scores('c', recognition=0.51), make_scores('i'), cfg)
    large = evaluate_gates(make_scores('c', recognition=0.53), make_scores('i'), cfg)
    assert small.passed is False
    assert large.passed is True


def test_reasoning_may_not_fall_below_the_margin(make_scores: Scores) -> None:
    incumbent = make_scores('i', reasoning=0.50)
    inside = make_scores('c', recognition=0.6, reasoning=0.485)
    outside = make_scores('c', recognition=0.6, reasoning=0.479)
    assert evaluate_gates(inside, incumbent, CFG).passed is True
    result = evaluate_gates(outside, incumbent, CFG)
    assert result.passed is False
    assert 'reasoning' in result.reasons[0]


def test_reasoning_gate_is_skipped_when_a_score_is_missing(make_scores: Scores) -> None:
    incumbent = make_scores('i', reasoning=None)
    candidate = make_scores('c', recognition=0.6, reasoning=0.0)
    assert evaluate_gates(candidate, incumbent, CFG).passed is True
    flipped = evaluate_gates(
        make_scores('c', recognition=0.6, reasoning=None),
        make_scores('i', reasoning=0.9),
        CFG,
    )
    assert flipped.passed is True


def test_regression_rate_is_capped(make_scores: Scores) -> None:
    at_limit = make_scores('c', recognition=0.6, regression_rate=0.05)
    over = make_scores('c', recognition=0.6, regression_rate=0.06)
    assert evaluate_gates(at_limit, make_scores('i'), CFG).passed is True
    result = evaluate_gates(over, make_scores('i'), CFG)
    assert result.passed is False
    assert 'regression rate' in result.reasons[0]


def test_call_and_latency_budgets_are_enforced(make_scores: Scores) -> None:
    calls = make_scores('c', recognition=0.6, mean_calls=4.5)
    latency = make_scores('c', recognition=0.6, p95_latency_s=11.0)
    assert 'mean calls' in evaluate_gates(calls, make_scores('i'), CFG).reasons[0]
    assert 'p95 latency' in evaluate_gates(latency, make_scores('i'), CFG).reasons[0]


def test_latency_is_unbounded_when_unset(make_scores: Scores) -> None:
    cfg = GateConfig()
    candidate = make_scores('c', recognition=0.6, mean_calls=1.0, p95_latency_s=999.0)
    assert evaluate_gates(candidate, make_scores('i'), cfg).passed is True


def test_all_failing_reasons_are_reported(make_scores: Scores) -> None:
    candidate = make_scores(
        'c',
        recognition=0.4,
        reasoning=0.1,
        regression_rate=0.9,
        mean_calls=9.0,
        p95_latency_s=99.0,
    )
    result = evaluate_gates(candidate, make_scores('i'), CFG)
    assert len(result.reasons) == 5


def test_select_takes_the_highest_recognition(make_scores: Scores) -> None:
    candidates = (
        make_scores('a', recognition=0.55),
        make_scores('b', recognition=0.70),
        make_scores('c', recognition=0.40),
    )
    best, results = select_candidate(candidates, make_scores('i'), CFG)
    assert best is not None
    assert best.name == 'b'
    assert [r.name for r in results] == ['a', 'b', 'c']
    assert [r.passed for r in results] == [True, True, False]


def test_ties_break_on_calls_then_name(make_scores: Scores) -> None:
    candidates = (
        make_scores('z', recognition=0.7, mean_calls=2.0),
        make_scores('a', recognition=0.7, mean_calls=3.0),
        make_scores('b', recognition=0.7, mean_calls=2.0),
    )
    best, _ = select_candidate(candidates, make_scores('i'), CFG)
    assert best is not None
    assert best.name == 'b'


def test_select_returns_none_when_nothing_passes(make_scores: Scores) -> None:
    best, results = select_candidate(
        (make_scores('a', recognition=0.1),), make_scores('i'), CFG
    )
    assert best is None
    assert results[0].passed is False


def test_select_of_nothing_is_none(make_scores: Scores) -> None:
    best, results = select_candidate((), make_scores('i'), CFG)
    assert best is None
    assert results == ()


def test_context_quality_guards_plain_accuracy(make_scores: Scores) -> None:
    from emoharness.evaluation.quality import QualityMetrics

    def q(recognition: float) -> QualityMetrics:
        return QualityMetrics(
            n=10,
            recognition=recognition,
            balanced=0.4,
            mean_confidence=0.9,
            brier=0.4,
            ece=0.4,
            objective=0.4,
        )

    cfg = CFG.model_copy(
        update={'selection_protocol': 'context_quality', 'regression_rate_limit': 0.10}
    )
    incumbent = make_scores('i').model_copy(update={'quality': q(0.50)})
    held = make_scores('c', recognition=0.6).model_copy(update={'quality': q(0.495)})
    fell = make_scores('d', recognition=0.6).model_copy(update={'quality': q(0.48)})
    assert evaluate_gates(held, incumbent, cfg).passed is True
    result = evaluate_gates(fell, incumbent, cfg)
    assert result.passed is False
    assert 'plain accuracy' in result.reasons[0]
    # other protocols never look at it
    assert evaluate_gates(fell, incumbent, CFG).passed is True
