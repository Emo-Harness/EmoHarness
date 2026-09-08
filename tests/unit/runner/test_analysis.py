"""The tables: ladder rows, per-task and multi-label breakdowns, hypotheses, cost."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json, write_jsonl_atomic
from emoharness.data.manifest import Example
from emoharness.data.tasks import Task
from emoharness.evaluation.schema import PredictionRecord
from emoharness.runner.analysis import (
    COST_JSON,
    HYPOTHESES_JSON,
    LADDER_JSON,
    LADDER_MD,
    MULTILABEL_JSON,
    PER_TASK_JSON,
    analyze,
    ladder_entry,
    ladder_markdown,
    multilabel_report,
    per_task_report,
)
from emoharness.runner.context import RunContext, build_context
from emoharness.runner.hypotheses import (
    GAP_TIER_1,
    GAP_TIER_2,
    PRE_REGISTERED,
    _tier,
    gap_closed_test,
    hypotheses_report,
    run_test,
)
from emoharness.runner.items import examples_for
from emoharness.runner.results import (
    RowIndex,
    discover_rows,
    load_row_data,
    unslug,
)


def synthetic_predictions(
    row: str, examples: Sequence[Example], correct_every: int
) -> tuple[PredictionRecord, ...]:
    """One prediction per example; every ``correct_every``-th one is right."""
    return tuple(
        PredictionRecord(
            example_id=example.example_id,
            row=row,
            task=example.task,
            labels=(
                example.labels[0]
                if index % correct_every == 0
                else example.label_set[-1],
            ),
            response_text='Step 1: a\nStep 5: b\n<answer>x</answer>',
            target_calls=index % 3 + 1,
            input_tokens=100,
            output_tokens=20,
            latency_s=0.5,
            cost_usd=0.001,
        )
        for index, example in enumerate(examples)
    )


def write_row(row: str, ctx: RunContext, correct_every: int) -> None:
    examples = examples_for('full', ctx.paths)
    records = synthetic_predictions(row, examples, correct_every)
    path = ctx.paths.predictions(row, 'full')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(path, (r.model_dump(mode='json') for r in records))


@pytest.fixture
def scored(prepared_root: Path, tmp_path: Path) -> RunContext:
    """A context with three synthetic rows of increasing accuracy."""
    ctx = build_context(tmp_path, fake=True, force=True, resamples=50)
    for name in ('mme_emotion.jsonl',):
        (ctx.paths.manifests / name).write_bytes(
            (prepared_root / 'data' / 'manifests' / name).read_bytes()
        )
    (ctx.paths.splits / 'subset_1000.json').write_bytes(
        (prepared_root / 'data' / 'splits' / 'subset_1000.json').read_bytes()
    )
    write_row('L0', ctx, correct_every=4)
    write_row('L3', ctx, correct_every=3)
    write_row('L5', ctx, correct_every=2)
    return ctx


class TestLadderEntry:
    def test_reports_local_scores_when_nothing_was_judged(
        self, scored: RunContext
    ) -> None:
        entry = ladder_entry(load_row_data('L5', 'full', scored), scored.resamples)
        assert entry.source == 'local'
        assert entry.reasoning is None and entry.cot is None

    def test_macros_sit_between_zero_and_one(self, scored: RunContext) -> None:
        entry = ladder_entry(load_row_data('L5', 'full', scored), scored.resamples)
        for value in (entry.seen_macro, entry.held_out_macro, entry.overall_macro):
            assert value is not None and 0.0 <= value <= 1.0

    def test_the_confidence_interval_brackets_the_estimate(
        self, scored: RunContext
    ) -> None:
        entry = ladder_entry(load_row_data('L5', 'full', scored), scored.resamples)
        assert entry.rec_ci_low is not None and entry.rec_ci_high is not None
        assert entry.rec_ci_low <= entry.rec_ci_high

    def test_budget_columns_come_from_the_predictions(self, scored: RunContext) -> None:
        entry = ladder_entry(load_row_data('L0', 'full', scored), scored.resamples)
        assert entry.mean_calls > 0
        assert entry.cost_per_1000_usd == pytest.approx(1.0)

    def test_a_better_row_scores_higher(self, scored: RunContext) -> None:
        weak = ladder_entry(load_row_data('L0', 'full', scored), scored.resamples)
        strong = ladder_entry(load_row_data('L5', 'full', scored), scored.resamples)
        assert strong.overall_macro is not None and weak.overall_macro is not None
        assert strong.overall_macro > weak.overall_macro


class TestTables:
    def test_the_markdown_table_has_one_line_per_row(self, scored: RunContext) -> None:
        entries = [
            ladder_entry(load_row_data(name, 'full', scored), scored.resamples)
            for name in ('L0', 'L3', 'L5')
        ]
        body = ladder_markdown(entries)
        assert all(f'| {entry.row} |' in body for entry in entries)
        assert 'Rec-S 95% CI' in body

    def test_per_task_reports_every_scored_task(self, scored: RunContext) -> None:
        rows = [load_row_data('L5', 'full', scored)]
        report: Mapping[str, Any] = per_task_report(rows)
        assert set(report['L5@full']['tasks']) == {task.value for task in Task}

    def test_multilabel_covers_only_the_multi_label_tasks(
        self, scored: RunContext
    ) -> None:
        report = multilabel_report([load_row_data('L5', 'full', scored)])
        assert set(report['L5@full']) == {'FG_ER', 'ML_ER'}


class TestHypotheses:
    def test_reports_every_pre_registered_test(self, scored: RunContext) -> None:
        rows = [load_row_data(name, 'full', scored) for name in ('L0', 'L3', 'L5')]
        report = hypotheses_report(RowIndex(rows, frozenset()), scored.resamples)
        assert [test['id'] for test in report['tests']] == [
            'H1',
            'H2',
            'H-B_seen',
            'H-B_held_out',
            'H3',
        ]

    def test_a_missing_row_is_recorded_not_dropped(self, scored: RunContext) -> None:
        rows = [load_row_data('L0', 'full', scored)]
        report = hypotheses_report(RowIndex(rows, frozenset()), scored.resamples)
        first = report['tests'][0]
        assert first['status'] == 'missing rows' and first['missing'] == ['L5']

    def test_holm_adjustment_covers_the_computed_tests(
        self, scored: RunContext
    ) -> None:
        rows = [load_row_data(name, 'full', scored) for name in ('L0', 'L3', 'L5')]
        report = hypotheses_report(RowIndex(rows, frozenset()), scored.resamples)
        computed = {t['id'] for t in report['tests'] if t['status'] == 'computed'}
        assert set(report['holm_adjusted']) == computed
        assert all(0.0 <= p <= 1.0 for p in report['holm_adjusted'].values())

    def test_the_builder_gain_is_positive_for_a_better_row(
        self, scored: RunContext
    ) -> None:
        rows = [load_row_data(name, 'full', scored) for name in ('L0', 'L3', 'L5')]
        report = hypotheses_report(RowIndex(rows, frozenset()), scored.resamples)
        gain = next(t for t in report['tests'] if t['id'] == 'H-B_seen')
        assert gain['estimate'] > 0

    def test_gap_closed_needs_the_upper_bound_row(self, scored: RunContext) -> None:
        from emoharness.data.splits import read_subset

        ids = frozenset(read_subset(scored.paths.subset).ids)
        rows = [load_row_data(name, 'full', scored) for name in ('L0', 'L5')]
        result = gap_closed_test(RowIndex(rows, ids), scored.resamples)
        assert result['status'] == 'missing rows' and result['missing'] == ['L6']

    def test_gap_closed_reports_a_ratio_once_the_upper_bound_exists(
        self, scored: RunContext
    ) -> None:
        from emoharness.data.splits import read_subset

        write_row('L6', scored, correct_every=1)
        ids = frozenset(read_subset(scored.paths.subset).ids)
        rows = [load_row_data(n, 'full', scored) for n in ('L0', 'L5', 'L6')]
        result = gap_closed_test(RowIndex(rows, ids), scored.resamples)
        assert result['status'] == 'computed'
        assert result['gap_closed'] is not None
        assert 0.0 < result['gap_closed'] <= 1.0
        assert result['tier_reached'] in (0, 1, 2)

    def test_a_test_with_no_item_in_its_group_says_so(self, scored: RunContext) -> None:
        """The empty-set rule reaching the report: no shared item, no number."""
        seen_only = frozenset(
            example_id
            for example_id, example in load_row_data(
                'L5', 'full', scored
            ).examples.items()
            if example.task in (Task.ER_LAB, Task.ER_WILD, Task.SA)
        )
        rows = [
            load_row_data(name, 'full', scored).restricted(seen_only)
            for name in ('L0', 'L5')
        ]
        held_out = next(s for s in PRE_REGISTERED if s.name == 'H2')
        result = run_test(held_out, RowIndex(rows, frozenset()), scored.resamples)
        assert result['status'] == 'no shared items'


@pytest.mark.parametrize(
    ('ratio', 'tier'),
    [(1.0, 2), (GAP_TIER_2, 2), (0.4, 1), (GAP_TIER_1, 1), (0.1, 0), (0.0, 0)],
)
def test_gap_tiers_follow_adr_0012(ratio: float, tier: int) -> None:
    assert _tier(ratio) == tier


class TestAnalyze:
    def test_writes_every_metrics_artifact(self, scored: RunContext) -> None:
        written = analyze(scored)
        assert set(written) == {
            LADDER_JSON,
            LADDER_MD,
            PER_TASK_JSON,
            HYPOTHESES_JSON,
            MULTILABEL_JSON,
            COST_JSON,
        }
        assert all(path.is_file() for path in written.values())

    def test_the_ladder_carries_every_requested_row(self, scored: RunContext) -> None:
        written = analyze(scored)
        document = read_json(written[LADDER_JSON])
        assert {row['row'] for row in document['rows']} == {'L0', 'L3', 'L5'}
        assert 'local' in document['note']

    def test_cost_groups_the_ledger_and_the_rows(self, scored: RunContext) -> None:
        document = read_json(analyze(scored)[COST_JSON])
        assert set(document) >= {'total', 'by_role', 'by_model_id', 'per_row'}
        assert set(document['per_row']) == {'L0@full', 'L3@full', 'L5@full'}

    def test_a_row_filter_narrows_the_tables(self, scored: RunContext) -> None:
        written = analyze(scored, [('L0', 'full')])
        assert [r['row'] for r in read_json(written[LADDER_JSON])['rows']] == ['L0']

    def test_an_empty_project_is_rejected(self, tmp_path: Path) -> None:
        empty = build_context(tmp_path, fake=True, force=True)
        with pytest.raises(ValidationError, match='no raw predictions'):
            analyze(empty)


class TestResults:
    def test_discover_rows_skips_the_harness_output_files(
        self, scored: RunContext
    ) -> None:
        assert set(discover_rows(scored.paths)) == {
            ('L0', 'full'),
            ('L3', 'full'),
            ('L5', 'full'),
        }

    def test_unslug_restores_an_external_item_set(self) -> None:
        assert unslug('external_ext_test') == 'external:ext_test'
        assert unslug('subset') == 'subset'

    def test_a_full_run_answers_for_the_subset(self, scored: RunContext) -> None:
        data = load_row_data('L5', 'full', scored)
        ids = frozenset(list(data.examples)[:4])
        index = RowIndex([data], ids)
        narrowed = index.get('L5', 'subset')
        assert narrowed is not None and len(narrowed.predictions) == 4

    def test_an_unknown_row_is_not_invented(self) -> None:
        assert RowIndex([], frozenset()).get('L9', 'full') is None
