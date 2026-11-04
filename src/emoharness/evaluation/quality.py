"""How good an emotion context is: class-balanced accuracy and calibrated confidence.

Plain Recognition rewards a label that absorbs every uncertain item: on ER_Wild
the frozen harnesses predict ``neutral`` almost twice as often as it is the
gold label. Class-balanced Recognition averages the recall of each gold label
within a task, then averages the tasks, so that absorption gains nothing.

A context is only useful to a reader who can tell how far to trust it. The
emitted label's confidence is scored as a probability of being right: the
Brier score, plus ECE over ten equal-width bins for reporting. The frozen
harnesses state 0.74-0.90 on average while being right about 42% of the time.

``objective`` weights accuracy twice as heavily as calibration,
``(2 * balanced + (1 - brier)) / 3``, so a candidate cannot buy its way up by
lowering every confidence while its labels get worse.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.harness.schema import HarnessOutput

ECE_BINS: Final = 10
ACCURACY_WEIGHT: Final = 2.0


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    low: float
    high: float
    n: int = Field(ge=0)
    mean_confidence: float | None = None
    accuracy: float | None = None


class QualityMetrics(BaseModel):
    """One bundle's context quality on one item set."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    n: int = Field(ge=1)
    recognition: float = Field(ge=0.0, le=1.0)
    balanced: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0, le=1.0)
    objective: float = Field(ge=0.0, le=1.0)
    reliability: tuple[ReliabilityBin, ...] = ()


def emitted_confidence(output: HarnessOutput) -> float:
    """The confidence the harness attached to the label it emitted (0 if none)."""
    top = output.labels[0] if output.labels else None
    for hypothesis in output.hypotheses.hypotheses:
        if hypothesis.label == top:
            return float(hypothesis.confidence)
    return 0.0


def context_quality(
    outputs: Sequence[HarnessOutput], examples: Mapping[str, Example]
) -> QualityMetrics:
    """Score single-label items; multi-label items have no single emitted claim."""
    rows: list[tuple[str, str, float, float]] = []
    for output in outputs:
        example = examples[output.example_id]
        if len(example.labels) != 1:
            continue
        labels = LabelSet.from_prompt(example.prompt)
        correct = labels.recognition(list(output.labels[:1]), list(example.labels))
        gold = labels.canonical(example.labels[0]) or example.labels[0]
        task = example.task.value if example.task else '?'
        rows.append((task, gold, float(correct), emitted_confidence(output)))
    if not rows:
        raise ValueError('no single-label item to score')
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task, gold, hit, _ in rows:
        per_task[task][gold].append(hit)
    balanced = sum(
        sum(sum(v) / len(v) for v in golds.values()) / len(golds)
        for golds in per_task.values()
    ) / len(per_task)
    hits = [r[2] for r in rows]
    conf = [r[3] for r in rows]
    brier = sum((c - k) ** 2 for c, k in zip(conf, hits, strict=True)) / len(rows)
    bins = []
    ece = 0.0
    for b in range(ECE_BINS):
        low, high = b / ECE_BINS, (b + 1) / ECE_BINS
        members = [
            i
            for i, c in enumerate(conf)
            if low <= c < high or (b == ECE_BINS - 1 and c == 1.0)
        ]
        if not members:
            bins.append(ReliabilityBin(low=low, high=high, n=0))
            continue
        mc = sum(conf[i] for i in members) / len(members)
        acc = sum(hits[i] for i in members) / len(members)
        ece += abs(mc - acc) * len(members) / len(rows)
        bins.append(
            ReliabilityBin(
                low=low, high=high, n=len(members), mean_confidence=mc, accuracy=acc
            )
        )
    return QualityMetrics(
        n=len(rows),
        recognition=sum(hits) / len(rows),
        balanced=balanced,
        mean_confidence=sum(conf) / len(rows),
        brier=brier,
        ece=min(1.0, ece),
        objective=(ACCURACY_WEIGHT * balanced + (1.0 - brier)) / (ACCURACY_WEIGHT + 1),
        reliability=tuple(bins),
    )
