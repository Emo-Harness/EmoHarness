"""Best-of-n test-time-scaling control: n samples, majority vote."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from emoharness.common.errors import ValidationError
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.evaluation.schema import PredictionRecord
from emoharness.target.base import TargetModel, TargetRequest, TargetResponse


def majority_vote(
    samples: Sequence[tuple[str, ...]], label_set: LabelSet
) -> tuple[str, ...]:
    """Majority vote over normalized label sets; ties go to the earliest sample.

    ``label_set`` decides which samples count as the same answer, so two
    spellings of one candidate vote together. The winning sample's verbatim
    labels come back unmodified.
    """
    if not samples:
        raise ValidationError('majority_vote requires at least one sample')
    keys = tuple(frozenset(label_set.canonical_many(sample)) for sample in samples)
    counts = Counter(keys)
    best_index = max(range(len(samples)), key=lambda i: (counts[keys[i]], -i))
    return samples[best_index]


def _sample_responses(
    target: TargetModel,
    request: TargetRequest,
    temperature: float,
    top_p: float,
    seeds: Sequence[int],
) -> tuple[TargetResponse, ...]:
    """One Target call per seed, each a copy of `request` with the sampling params set."""
    return tuple(
        target.generate(
            request.model_copy(
                update={'temperature': temperature, 'top_p': top_p, 'seed': seed}
            )
        )
        for seed in seeds
    )


def _parse_or_fallback(text: str, label_set: LabelSet) -> tuple[tuple[str, ...], bool]:
    """Parsed candidate labels and whether parsing failed (falls back to the first candidate)."""
    labels = label_set.parse_answer(text)
    return (labels, False) if labels else ((label_set.labels[0],), True)


def run_best_of_n(
    target: TargetModel,
    request: TargetRequest,
    example: Example,
    n: int,
    temperature: float,
    top_p: float,
    seeds: tuple[int, ...] | None,
    row: str,
    cost_fn: Callable[[TargetResponse], float] | None = None,
) -> PredictionRecord:
    """Issue n samples over identical inputs, vote, and report the winner.

    Each sample is a copy of `request` with `temperature`, `top_p`, and one seed
    from `seeds` (default `0..n-1`, matching best-of-4's seeds 0-3). A sample
    that parses to no candidate label falls back to the first candidate and
    counts one parse failure, exactly as `run_single_call` does. ``cost_fn`` prices
    one response; the record's cost is the sum over all n samples.
    """
    sample_seeds = seeds if seeds is not None else tuple(range(n))
    if len(sample_seeds) != n:
        raise ValidationError('run_best_of_n requires exactly one seed per sample')

    label_set = LabelSet.from_prompt(example.prompt)
    responses = _sample_responses(target, request, temperature, top_p, sample_seeds)
    parsed = [_parse_or_fallback(r.text, label_set) for r in responses]
    parsed_labels = [labels for labels, _failed in parsed]
    parse_failures = sum(1 for _labels, failed in parsed if failed)

    winning_labels = majority_vote(tuple(parsed_labels), label_set)
    winning_response = responses[parsed_labels.index(winning_labels)]

    return PredictionRecord(
        example_id=example.example_id,
        row=row,
        task=example.task,
        labels=winning_labels,
        response_text=winning_response.text,
        target_calls=n,
        parse_failures=parse_failures,
        input_tokens=sum(r.input_tokens for r in responses),
        output_tokens=sum(r.output_tokens for r in responses),
        latency_s=sum(r.latency_s for r in responses),
        cost_usd=sum(cost_fn(r) for r in responses) if cost_fn else 0.0,
    )
