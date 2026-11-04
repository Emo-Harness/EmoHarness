"""Context quality: class-balanced accuracy and the calibration of the emitted label."""

from __future__ import annotations

import pytest

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import parse_prompt
from emoharness.data.tasks import Task
from emoharness.evaluation.quality import context_quality, emitted_confidence
from emoharness.harness.schema import (
    AudioEvidence,
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    VerificationResult,
    VisualEvidence,
)

from .test_scoreboard import make_example


def stated(example: Example, label: str, confidence: float) -> HarnessOutput:
    hypotheses = HypothesisSet(
        coarse='x', hypotheses=(Hypothesis(label=label, confidence=confidence),)
    )
    return HarnessOutput(
        example_id=example.example_id,
        labels=(label,),
        steps=('Step 5: Decision.',),
        response_text=f'<answer>{label}</answer>',
        evidence=EvidenceRecord(
            visual=VisualEvidence(),
            audio=AudioEvidence(speech_present=True),
            text=parse_prompt(example.prompt),
        ),
        hypotheses=hypotheses,
        verification=VerificationResult(
            modalities_agree=True, ranked=hypotheses.hypotheses
        ),
    )


@pytest.fixture
def items() -> dict[str, Example]:
    """Three neutral clips and one angry clip, one task."""
    rows = [make_example(f'n{i}', ('neutral',), task=Task.ER_WILD) for i in range(3)]
    rows.append(make_example('a0', ('anger',), task=Task.ER_WILD))
    return {e.example_id: e for e in rows}


def test_a_label_that_absorbs_everything_gains_nothing_balanced(
    items: dict[str, Example],
) -> None:
    outputs = [stated(e, 'neutral', 0.9) for e in items.values()]
    q = context_quality(outputs, items)
    assert q.recognition == pytest.approx(0.75)
    assert q.balanced == pytest.approx(0.5)  # neutral recall 1, anger recall 0


def test_overconfidence_costs_brier_and_ece(items: dict[str, Example]) -> None:
    sure = context_quality([stated(e, 'neutral', 1.0) for e in items.values()], items)
    honest = context_quality(
        [stated(e, 'neutral', 0.75) for e in items.values()], items
    )
    assert sure.brier == pytest.approx(0.25)  # one wrong at 1.0, three right at 1.0
    assert honest.brier == pytest.approx((3 * 0.25**2 + 0.75**2) / 4)
    assert honest.ece == pytest.approx(0.0)
    assert sure.ece == pytest.approx(0.25)
    assert honest.objective > sure.objective
    assert honest.balanced == sure.balanced


def test_accuracy_outweighs_calibration(items: dict[str, Example]) -> None:
    lazy = context_quality([stated(e, 'neutral', 0.75) for e in items.values()], items)
    right = [
        stated(e, 'anger' if e.example_id == 'a0' else 'neutral', 0.9)
        for e in items.values()
    ]
    assert context_quality(right, items).objective > lazy.objective


def test_the_emitted_label_carries_its_own_confidence(
    items: dict[str, Example],
) -> None:
    out = stated(items['a0'], 'anger', 0.4)
    assert emitted_confidence(out) == pytest.approx(0.4)
