"""Official-judge inputs, the fake judge that stands in for it, and the join back."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json
from emoharness.data.tasks import Task
from emoharness.runner.context import RunContext
from emoharness.runner.items import examples_for
from emoharness.runner.judging import (
    EVAL_FILE,
    INDEX_FILE,
    RESPONSES_FILE,
    collect_step_scores,
    count_steps,
    fake_judge,
    group_by_task,
    official_commands,
    prepare_official_inputs,
    read_predictions,
    run_official,
    task_dir,
)
from emoharness.runner.rows import run_row

ROW = 'L3'
ITEMS = 'full'


@pytest.fixture(scope='module')
def judged(prepared_root: Path) -> RunContext:
    """A context whose L3 row has been run over every benchmark item."""
    from emoharness.runner.context import build_context

    ctx = build_context(prepared_root, fake=True, force=True, resamples=50)
    run_row(ctx.ladder.rows[ROW], examples_for(ITEMS, ctx.paths), ctx, item_set=ITEMS)
    return ctx


class TestPrepareInputs:
    def test_writes_one_response_file_per_task(self, judged: RunContext) -> None:
        written = prepare_official_inputs(ROW, ITEMS, judged)
        assert set(written) == set(Task)
        assert all(path.name == RESPONSES_FILE for path in written.values())

    def test_response_rows_carry_the_official_fields(self, judged: RunContext) -> None:
        prepare_official_inputs(ROW, ITEMS, judged)
        rows = read_json(task_dir(ROW, ITEMS, Task.SA, judged.paths) / RESPONSES_FILE)
        assert all(
            set(row) == {'video_id', 'ground_truth', 'model_response'} for row in rows
        )

    def test_the_index_maps_video_ids_back_to_example_ids(
        self, judged: RunContext
    ) -> None:
        prepare_official_inputs(ROW, ITEMS, judged)
        index = read_json(task_dir(ROW, ITEMS, Task.SA, judged.paths) / INDEX_FILE)
        assert all(value.startswith('mme_') for value in index.values())

    def test_a_missing_row_is_reported(self, judged: RunContext) -> None:
        with pytest.raises(ValidationError, match='no predictions'):
            read_predictions('L9', ITEMS, judged.paths)


class TestCommands:
    def test_one_command_per_task_naming_the_three_scripts(
        self, judged: RunContext, tmp_path: Path
    ) -> None:
        commands = official_commands(
            ROW, ITEMS, judged, tmp_path / 'repo', None, tmp_path / 'videos'
        )
        assert len(commands) == len(Task)
        assert len(commands[0].commands()) == 3

    def test_the_evidence_directory_is_only_ever_an_argument(
        self, judged: RunContext, tmp_path: Path
    ) -> None:
        evidence = tmp_path / 'evidence'
        commands = official_commands(
            ROW, ITEMS, judged, tmp_path / 'repo', evidence, tmp_path / 'videos'
        )
        forwarded = commands[0].eval_cot_command()
        assert str(evidence / f'{commands[0].task.value}.json') in forwarded
        assert not evidence.exists()

    def test_run_official_uses_the_injected_runner_and_bills_the_ledger(
        self, judged: RunContext, tmp_path: Path
    ) -> None:
        prepare_official_inputs(ROW, ITEMS, judged)
        seen: list[Sequence[str]] = []

        def runner(command: Sequence[str]) -> int:
            seen.append(command)
            return 0

        commands = official_commands(
            ROW, ITEMS, judged, tmp_path / 'repo', None, tmp_path / 'videos'
        )
        results = run_official(commands[:1], judged, runner)
        assert len(seen) == 3
        assert results[commands[0].task.value] == (0, 0, 0)
        assert 'judge_official' in judged.ledger.read_text(encoding='utf-8')

    def test_a_failing_script_stops_the_rest(
        self, judged: RunContext, tmp_path: Path
    ) -> None:
        prepare_official_inputs(ROW, ITEMS, judged)
        commands = official_commands(
            ROW, ITEMS, judged, tmp_path / 'repo', None, tmp_path / 'videos'
        )
        results = run_official(commands[:1], judged, lambda _command: 3)
        assert results[commands[0].task.value] == (3,)


class TestFakeJudge:
    def test_writes_one_eval_file_per_task(self, judged: RunContext) -> None:
        written = fake_judge(ROW, ITEMS, judged)
        assert set(written) == set(Task)
        assert all(path.name == EVAL_FILE for path in written.values())

    def test_scores_parse_with_the_official_parser(self, judged: RunContext) -> None:
        fake_judge(ROW, ITEMS, judged)
        scores = collect_step_scores(ROW, ITEMS, judged)
        assert len(scores) == len(examples_for(ITEMS, judged.paths))
        assert all(0.0 <= score.recognition <= 1.0 for score in scores)

    def test_scores_come_back_keyed_by_example_id(self, judged: RunContext) -> None:
        fake_judge(ROW, ITEMS, judged)
        ids = {score.example_id for score in collect_step_scores(ROW, ITEMS, judged)}
        assert ids == {e.example_id for e in examples_for(ITEMS, judged.paths)}

    def test_multi_label_items_get_an_official_ratio(self, judged: RunContext) -> None:
        fake_judge(ROW, ITEMS, judged)
        gold = {
            e.example_id: e.labels
            for e in examples_for(ITEMS, judged.paths)
            if e.task is Task.ML_ER
        }
        scores = [
            s for s in collect_step_scores(ROW, ITEMS, judged) if s.example_id in gold
        ]
        assert scores
        assert all(s.recognition * len(gold[s.example_id]) % 1 == 0 for s in scores)

    def test_nothing_is_collected_before_the_judge_has_run(
        self, prepared_root: Path
    ) -> None:
        from emoharness.runner.context import build_context

        fresh = build_context(prepared_root, fake=True, force=True)
        assert collect_step_scores('never_run', ITEMS, fresh) == ()


class TestHelpers:
    def test_group_by_task_drops_items_without_a_task(self, judged: RunContext) -> None:
        predictions = read_predictions(ROW, ITEMS, judged.paths)
        grouped = group_by_task(predictions)
        assert sum(len(rows) for rows in grouped.values()) == len(predictions)

    def test_count_steps_matches_the_response_shape(self) -> None:
        assert count_steps('Step 1: a\nStep 2: b\n<answer>x</answer>') == 2

    def test_count_steps_never_returns_zero(self) -> None:
        assert count_steps('no steps here') == 1
