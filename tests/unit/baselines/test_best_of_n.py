from __future__ import annotations

import pytest

from emoharness.baselines.best_of_n import majority_vote, run_best_of_n
from emoharness.common.errors import ValidationError
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.target.base import TargetRequest
from emoharness.target.fake import FakeTarget

LABELS = LabelSet.from_labels(('anger', 'happiness', 'sadness', 'neutral'))


def test_majority_vote_picks_the_larger_group() -> None:
    samples = (('anger',), ('anger',), ('sadness',))
    assert majority_vote(samples, LABELS) == ('anger',)


def test_majority_vote_ties_go_to_the_earliest_sample() -> None:
    samples = (('anger',), ('sadness',))
    assert majority_vote(samples, LABELS) == ('anger',)
    assert majority_vote(tuple(reversed(samples)), LABELS) == ('sadness',)


def test_majority_vote_normalizes_for_counting_but_returns_verbatim_winner() -> None:
    samples = (('Happy',), ('happiness',), ('sad',))
    # 'Happy' and 'happiness' normalize to the same candidate (2 votes) and beat
    # 'sad' (1 vote); the *verbatim* text of the earliest such sample is returned.
    assert majority_vote(samples, LABELS) == ('Happy',)


def test_majority_vote_requires_at_least_one_sample() -> None:
    with pytest.raises(ValidationError):
        majority_vote((), LABELS)


def _responder_by_seed(request: TargetRequest) -> str:
    return (
        '<answer>anger</answer>'
        if request.seed % 2 == 0
        else '<answer>sadness</answer>'
    )


def test_run_best_of_n_issues_n_requests_with_distinct_seeds_and_votes(
    benchmark_example: Example,
) -> None:
    target = FakeTarget(responder=_responder_by_seed)
    request = TargetRequest(prompt=benchmark_example.prompt, video_path='clip.mp4')
    record = run_best_of_n(
        target,
        request,
        benchmark_example,
        n=4,
        temperature=0.7,
        top_p=0.9,
        seeds=None,
        row='best_of_4',
    )
    assert target.call_count == 4
    seeds_used = [r.seed for r in target.requests]
    assert seeds_used == [0, 1, 2, 3]
    assert len({r.seed for r in target.requests}) == 4
    for sent in target.requests:
        assert sent.temperature == pytest.approx(0.7)
        assert sent.top_p == pytest.approx(0.9)

    # seeds 0 and 2 answer 'anger', seeds 1 and 3 answer 'sadness': a 2-2 tie
    # broken by the earliest sample (seed 0, 'anger').
    assert record.labels == ('anger',)
    assert record.response_text == '<answer>anger</answer>'
    assert record.target_calls == 4
    assert record.parse_failures == 0
    assert record.input_tokens > 0
    assert record.output_tokens > 0
    assert record.latency_s >= 0.0
    assert record.task == benchmark_example.task


def test_run_best_of_n_accepts_explicit_seeds(benchmark_example: Example) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = TargetRequest(prompt=benchmark_example.prompt, video_path='clip.mp4')
    run_best_of_n(
        target,
        request,
        benchmark_example,
        n=3,
        temperature=0.5,
        top_p=1.0,
        seeds=(10, 20, 30),
        row='best_of_3',
    )
    assert [r.seed for r in target.requests] == [10, 20, 30]


def test_run_best_of_n_rejects_mismatched_seed_count(
    benchmark_example: Example,
) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = TargetRequest(prompt=benchmark_example.prompt, video_path='clip.mp4')
    with pytest.raises(ValidationError):
        run_best_of_n(
            target,
            request,
            benchmark_example,
            n=4,
            temperature=0.7,
            top_p=0.9,
            seeds=(1, 2),
            row='best_of_4',
        )


def test_run_best_of_n_counts_parse_failures_and_still_votes(
    benchmark_example: Example,
) -> None:
    def responder(request: TargetRequest) -> str:
        return (
            'no parseable label here'
            if request.seed == 0
            else '<answer>sadness</answer>'
        )

    target = FakeTarget(responder=responder)
    request = TargetRequest(prompt=benchmark_example.prompt, video_path='clip.mp4')
    record = run_best_of_n(
        target,
        request,
        benchmark_example,
        n=3,
        temperature=0.7,
        top_p=0.9,
        seeds=None,
        row='best_of_3',
    )
    assert record.parse_failures == 1
    # 2 of 3 samples say 'sadness'; the fallback sample counted the first
    # candidate label, which does not happen to be 'sadness' here.
    assert record.labels == ('sadness',)
    assert record.target_calls == 3


def test_run_best_of_n_sums_the_cost_of_every_sample(
    benchmark_example: Example,
) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = TargetRequest(prompt=benchmark_example.prompt, video_path='clip.mp4')
    record = run_best_of_n(
        target,
        request,
        benchmark_example,
        n=4,
        temperature=0.7,
        top_p=0.9,
        seeds=None,
        row='best_of_4',
        cost_fn=lambda _response: 0.5,
    )
    assert record.cost_usd == 2.0
    assert record.target_calls == 4


def test_every_sample_keeps_the_tag_of_the_row_it_resamples(
    benchmark_example: Example,
) -> None:
    target = FakeTarget(default_text='<answer>sadness</answer>')
    request = TargetRequest(
        prompt=benchmark_example.prompt, tag='direct', video_path='clip.mp4'
    )

    run_best_of_n(
        target,
        request,
        benchmark_example,
        n=3,
        temperature=0.7,
        top_p=0.9,
        seeds=None,
        row='best_of_4',
    )

    assert [sample.tag for sample in target.requests] == ['direct'] * 3
