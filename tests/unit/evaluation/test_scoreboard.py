"""The Scoreboard: three constructors, the empty-set rule, and every aggregate."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from emoharness.common.constants import COT_ALPHA
from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.data.prompt_template import (
    Quantifier,
    parse_prompt,
    render_prompt,
    render_prompt_for_kind,
    render_prompt_for_task,
)
from emoharness.data.tasks import Group, Kind, Split, Task
from emoharness.evaluation.schema import PredictionRecord, StepScores
from emoharness.evaluation.scoreboard import Scoreboard
from emoharness.harness.schema import (
    AudioEvidence,
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    VerificationResult,
    VisualEvidence,
)

ER_LABELS = ('anger', 'happiness', 'sadness', 'neutral')
SA_LABELS = ('positive', 'negative', 'neutral')
RESAMPLES = 200


def make_example(
    example_id: str,
    labels: tuple[str, ...],
    *,
    task: Task | None = None,
    kind: Kind | None = None,
    label_set: tuple[str, ...] = ER_LABELS,
) -> Example:
    if task is not None:
        prompt = render_prompt_for_task(task, label_set)
    elif kind is not None:
        prompt = render_prompt_for_kind(kind, label_set)
    else:
        prompt = render_prompt(Quantifier.MOST_PROMINENT, 'emotion', label_set)
    return Example(
        example_id=example_id,
        source_id=example_id,
        origin='synthetic',
        task=task,
        kind=kind,
        split=Split.TEST if task is not None else Split.BUILD,
        media_path=f'/clips/{example_id}.mp4',
        prompt=prompt,
        label_set=label_set,
        labels=labels,
    )


def prediction(example: Example, labels: tuple[str, ...]) -> PredictionRecord:
    return PredictionRecord(
        example_id=example.example_id,
        row='L5',
        task=example.task,
        labels=labels,
        response_text=f'<answer>{", ".join(labels)}</answer>',
        target_calls=1,
    )


def output(example: Example, labels: tuple[str, ...]) -> HarnessOutput:
    hypotheses = HypothesisSet(
        coarse='mixed', hypotheses=(Hypothesis(label=labels[0], confidence=0.9),)
    )
    return HarnessOutput(
        example_id=example.example_id,
        labels=labels,
        steps=('Step 5: Decision.',),
        response_text=f'<answer>{", ".join(labels)}</answer>',
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


def by_id(examples: tuple[Example, ...]) -> Mapping[str, Example]:
    return {example.example_id: example for example in examples}


@pytest.fixture
def benchmark_items() -> tuple[Example, ...]:
    """Two seen-group tasks and one held-out task, four items in all."""
    return (
        make_example('lab_1', ('anger',), task=Task.ER_LAB),
        make_example('lab_2', ('sadness',), task=Task.ER_LAB),
        make_example('sa_1', ('positive',), task=Task.SA, label_set=SA_LABELS),
        make_example('ir_1', ('anger',), task=Task.IR),
    )


@pytest.fixture
def scored(benchmark_items: tuple[Example, ...]) -> Scoreboard:
    """lab_1 right, lab_2 wrong, sa_1 right, ir_1 wrong."""
    right = {'lab_1', 'sa_1'}
    predictions = tuple(
        prediction(
            example,
            example.labels if example.example_id in right else (example.label_set[-1],),
        )
        for example in benchmark_items
    )
    return Scoreboard.from_predictions(predictions, by_id(benchmark_items))


class TestConstructors:
    def test_from_predictions_scores_locally(self, scored: Scoreboard) -> None:
        assert scored.source == 'local'
        assert scored.count == 4
        assert scored.ids == {'lab_1', 'lab_2', 'sa_1', 'ir_1'}
        assert scored.items['lab_1'].recognition == 1.0
        assert scored.items['lab_2'].recognition == 0.0
        assert scored.items['lab_1'].reasoning is None
        assert scored.items['lab_1'].num_steps is None

    def test_from_predictions_rejects_an_unknown_example(self) -> None:
        example = make_example('lab_1', ('anger',), task=Task.ER_LAB)
        with pytest.raises(ValidationError, match='no example for prediction'):
            Scoreboard.from_predictions((prediction(example, ('anger',)),), {})

    def test_from_outputs_agrees_with_from_predictions(
        self, benchmark_items: tuple[Example, ...]
    ) -> None:
        examples = by_id(benchmark_items)
        labels = {'lab_1': ('anger',), 'lab_2': ('anger',)}
        chosen = tuple(benchmark_items[:2])
        from_outputs = Scoreboard.from_outputs(
            tuple(output(e, labels[e.example_id]) for e in chosen), examples
        )
        from_predictions = Scoreboard.from_predictions(
            tuple(prediction(e, labels[e.example_id]) for e in chosen), examples
        )
        assert from_outputs.items == from_predictions.items

    def test_from_outputs_rejects_an_unknown_example(self) -> None:
        example = make_example('lab_1', ('anger',), task=Task.ER_LAB)
        with pytest.raises(ValidationError, match='no example for prediction'):
            Scoreboard.from_outputs((output(example, ('anger',)),), {})

    def test_from_step_scores_reads_the_judge(
        self, benchmark_items: tuple[Example, ...]
    ) -> None:
        scores = (
            StepScores(example_id='lab_1', scores=(1.0, 0.0, 1.0)),
            StepScores(example_id='lab_2', scores=(0.0,)),
        )
        board = Scoreboard.from_step_scores(scores, by_id(benchmark_items))
        assert board.source == 'official'
        assert board.items['lab_1'].recognition == 1.0
        assert board.items['lab_1'].reasoning == pytest.approx(0.5)
        assert board.items['lab_1'].num_steps == 3
        # A single-step item has no reasoning steps to average.
        assert board.items['lab_2'].reasoning == 0.0
        assert board.items['lab_2'].task is Task.ER_LAB

    def test_from_step_scores_takes_prf_from_a_matching_prediction(self) -> None:
        example = make_example(
            'ml_1', ('anger', 'sadness'), task=Task.ML_ER, label_set=ER_LABELS
        )
        scores = (StepScores(example_id='ml_1', scores=(1.0, 0.5)),)
        examples = by_id((example,))
        without = Scoreboard.from_step_scores(scores, examples)
        with_prediction = Scoreboard.from_step_scores(
            scores, examples, (prediction(example, ('anger',)),)
        )
        assert without.items['ml_1'].prf is None
        attached = with_prediction.items['ml_1'].prf
        assert attached is not None and attached.recall == 0.5

    def test_a_judged_item_with_no_example_joins_no_macro(self) -> None:
        board = Scoreboard.from_step_scores(
            (StepScores(example_id='ghost', scores=(1.0,)),), {}
        )
        assert board.count == 1
        assert board.items['ghost'].task is None
        assert board.macro() is None
        assert board.mean() == 1.0


class TestAggregates:
    def test_mean_is_flat_over_every_item(self, scored: Scoreboard) -> None:
        assert scored.mean() == pytest.approx(0.5)

    def test_by_task_averages_within_each_task(self, scored: Scoreboard) -> None:
        by_task = scored.by_task()
        assert by_task[Task.ER_LAB] == pytest.approx(0.5)
        assert by_task[Task.SA] == 1.0
        assert by_task[Task.IR] == 0.0
        assert Task.ER_WILD not in by_task

    def test_macro_weights_tasks_not_items(self, scored: Scoreboard) -> None:
        assert scored.macro(Group.SEEN) == pytest.approx(0.75)
        assert scored.macro(Group.HELD_OUT) == 0.0
        assert scored.macro() == pytest.approx(0.5)

    def test_by_kind_covers_construction_items(self) -> None:
        items = (
            make_example('b1', ('positive',), kind=Kind.SENTIMENT, label_set=SA_LABELS),
            make_example('b2', ('negative',), kind=Kind.SENTIMENT, label_set=SA_LABELS),
            make_example('b3', ('anger',), kind=Kind.LAB_ER),
        )
        board = Scoreboard.from_predictions(
            (
                prediction(items[0], ('positive',)),
                prediction(items[1], ('positive',)),
                prediction(items[2], ('anger',)),
            ),
            by_id(items),
        )
        assert board.by_kind() == {Kind.SENTIMENT: 0.5, Kind.LAB_ER: 1.0}
        assert board.by_task() == {}


class TestTheEmptySetRule:
    def test_an_empty_scoreboard_has_no_mean_and_no_macro(self) -> None:
        empty = Scoreboard(items={}, source='local')
        assert empty.count == 0
        assert empty.mean() is None
        assert empty.macro() is None
        assert empty.macro(Group.SEEN) is None
        assert empty.ci(RESAMPLES) is None

    def test_a_group_with_no_task_scored_is_none_not_zero(self) -> None:
        example = make_example('sa_1', ('positive',), task=Task.SA, label_set=SA_LABELS)
        board = Scoreboard.from_predictions(
            (prediction(example, ('negative',)),), by_id((example,))
        )
        # SA is seen and scored zero; nothing held-out was scored at all.
        assert board.macro(Group.SEEN) == 0.0
        assert board.macro(Group.HELD_OUT) is None

    def test_items_without_a_task_have_no_macro_but_do_have_a_mean(self) -> None:
        example = make_example('b1', ('anger',), kind=Kind.LAB_ER)
        board = Scoreboard.from_predictions(
            (prediction(example, ('anger',)),), by_id((example,))
        )
        assert board.macro() is None
        assert board.mean() == 1.0


class TestNarrowing:
    def test_restrict_keeps_only_the_named_ids(self, scored: Scoreboard) -> None:
        narrowed = scored.restrict({'lab_1', 'sa_1', 'not-here'})
        assert narrowed.ids == {'lab_1', 'sa_1'}
        assert narrowed.source == scored.source
        assert scored.count == 4

    def test_in_group_keeps_only_that_group(self, scored: Scoreboard) -> None:
        seen = scored.in_group(Group.SEEN)
        assert seen.ids == {'lab_1', 'lab_2', 'sa_1'}
        assert scored.in_group(Group.HELD_OUT).ids == {'ir_1'}
        assert seen.macro(Group.HELD_OUT) is None


class TestBootstrap:
    def test_ci_brackets_the_macro(self, scored: Scoreboard) -> None:
        interval = scored.ci(RESAMPLES)
        macro = scored.macro()
        assert interval is not None and macro is not None
        assert interval[0] <= macro <= interval[1]

    def test_ci_is_none_without_task_bearing_items(self) -> None:
        example = make_example('b1', ('anger',), kind=Kind.LAB_ER)
        board = Scoreboard.from_predictions(
            (prediction(example, ('anger',)),), by_id((example,))
        )
        assert board.ci(RESAMPLES) is None

    def test_compare_estimates_the_macro_difference_by_task(
        self, benchmark_items: tuple[Example, ...]
    ) -> None:
        examples = by_id(benchmark_items)
        wrong = Scoreboard.from_predictions(
            tuple(prediction(e, (e.label_set[-1],)) for e in benchmark_items), examples
        )
        right = Scoreboard.from_predictions(
            tuple(prediction(e, e.labels) for e in benchmark_items), examples
        )
        result = wrong.compare(right, RESAMPLES)
        assert result is not None
        assert result.estimate == pytest.approx(1.0)
        assert result.n_items == 4

    def test_compare_strata_by_kind_when_there_is_no_task(self) -> None:
        items = tuple(
            make_example(f'b{i}', ('anger',), kind=Kind.LAB_ER) for i in range(6)
        )
        examples = by_id(items)
        wrong = Scoreboard.from_predictions(
            tuple(prediction(e, ('neutral',)) for e in items), examples
        )
        right = Scoreboard.from_predictions(
            tuple(prediction(e, ('anger',)) for e in items), examples
        )
        result = wrong.compare(right, RESAMPLES)
        assert result is not None and result.estimate == pytest.approx(1.0)

    def test_compare_pools_items_with_neither_task_nor_kind(self) -> None:
        items = tuple(make_example(f'x{i}', ('anger',)) for i in range(6))
        examples = by_id(items)
        wrong = Scoreboard.from_predictions(
            tuple(prediction(e, ('neutral',)) for e in items), examples
        )
        right = Scoreboard.from_predictions(
            tuple(prediction(e, ('anger',)) for e in items), examples
        )
        result = wrong.compare(right, RESAMPLES)
        assert result is not None and result.estimate == pytest.approx(1.0)

    def test_compare_is_none_without_shared_ids(self, scored: Scoreboard) -> None:
        assert scored.compare(scored.restrict(()), RESAMPLES) is None

    def test_compare_only_pairs_shared_ids(self, scored: Scoreboard) -> None:
        result = scored.compare(scored.restrict({'lab_1', 'lab_2'}), RESAMPLES)
        assert result is not None and result.n_items == 2


class TestMethodScores:
    def test_reproduces_the_official_cal_metrics_arithmetic(self) -> None:
        examples = by_id(
            tuple(
                make_example(name, ('anger',), task=Task.ER_LAB)
                for name in ('1', '2', '3')
            )
        )
        scores = (
            StepScores(example_id='1', scores=(1.0, 0.0, 1.0)),
            StepScores(example_id='2', scores=(0.0,)),
            StepScores(example_id='3', scores=(1.0, 1.0, 1.0, 0.5)),
        )
        result = Scoreboard.from_step_scores(scores, examples).method_scores(
            row='L5', item_set='full'
        )
        assert result.count == 3
        assert result.recognition == pytest.approx(0.5)
        assert result.reasoning == pytest.approx(0.5)
        assert result.cot == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
        assert result.avg_steps == pytest.approx(8 / 3)

    def test_alpha_moves_cot_only(self) -> None:
        examples = by_id((make_example('1', ('anger',), task=Task.ER_LAB),))
        scores = (StepScores(example_id='1', scores=(1.0, 0.0)),)
        board = Scoreboard.from_step_scores(scores, examples)
        default = board.method_scores(row='L5', item_set='full')
        weighted = board.method_scores(row='L5', item_set='full', alpha=1.0)
        assert default.recognition == weighted.recognition == 0.0
        assert default.reasoning == weighted.reasoning == 1.0
        assert weighted.cot == 0.0
        assert default.cot == pytest.approx(COT_ALPHA * 0.0 + (1 - COT_ALPHA) * 1.0)

    def test_empty_is_zero_because_a_table_cell_must_still_print(self) -> None:
        result = Scoreboard(items={}, source='official').method_scores('L5', 'subset')
        assert result.count == 0
        assert (result.recognition, result.reasoning, result.avg_steps) == (
            0.0,
            0.0,
            0.0,
        )

    def test_a_local_scoreboard_has_no_reasoning_component(
        self, scored: Scoreboard
    ) -> None:
        result = scored.method_scores('L5', 'full')
        assert result.recognition == pytest.approx(0.5)
        assert (result.reasoning, result.avg_steps) == (0.0, 0.0)


class TestMultilabel:
    def test_reports_only_the_multi_label_items(self) -> None:
        multi = make_example(
            'ml_1', ('anger', 'sadness'), task=Task.ML_ER, label_set=ER_LABELS
        )
        single = make_example('lab_1', ('anger',), task=Task.ER_LAB)
        board = Scoreboard.from_predictions(
            (prediction(multi, ('anger',)), prediction(single, ('anger',))),
            by_id((multi, single)),
        )
        report = board.multilabel()
        assert set(report) == {Task.ML_ER}
        assert report[Task.ML_ER].precision == 1.0
        assert report[Task.ML_ER].recall == 0.5
        assert report[Task.ML_ER].f1 == pytest.approx(2 / 3)

    def test_averages_over_the_items_of_one_task(self) -> None:
        items = (
            make_example('ml_1', ('anger', 'sadness'), task=Task.ML_ER),
            make_example('ml_2', ('anger', 'sadness'), task=Task.ML_ER),
        )
        board = Scoreboard.from_predictions(
            (
                prediction(items[0], ('anger', 'sadness')),
                prediction(items[1], ('anger',)),
            ),
            by_id(items),
        )
        assert board.multilabel()[Task.ML_ER].recall == pytest.approx(0.75)

    def test_nothing_multi_label_reports_nothing(self, scored: Scoreboard) -> None:
        assert scored.multilabel() == {}
