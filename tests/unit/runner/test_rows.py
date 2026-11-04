"""Row dispatch: every ladder method, its artifacts, and its budget accounting."""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from emoharness.common.errors import ValidationError
from emoharness.common.io import iter_jsonl
from emoharness.evaluation.ladder import RowSpec
from emoharness.evaluation.schema import PredictionRecord
from emoharness.runner.context import RunContext
from emoharness.runner.items import SUBSET, examples_for
from emoharness.runner.rows import (
    _teacher_mode,
    bundle_dir_for,
    cost_fn,
    pipeline_config,
    run_row,
)


def read_rows(path: Path) -> tuple[PredictionRecord, ...]:
    return tuple(PredictionRecord.model_validate(obj) for obj in iter_jsonl(path))


@pytest.fixture
def items(ctx: RunContext) -> tuple[object, ...]:
    return examples_for('full', ctx.paths)[:4]


class TestSingleCallRows:
    @pytest.mark.parametrize('name', ['L0', 'L1', 'L2'])
    def test_writes_one_prediction_per_item(
        self, name: str, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        row = ctx.ladder.rows[name]
        predictions = run_row(row, items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert len(predictions) == len(items)
        assert read_rows(ctx.paths.predictions(name, SUBSET)) == predictions

    def test_labels_come_from_the_candidate_list(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L0'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        for prediction, example in zip(predictions, items, strict=True):
            assert set(prediction.labels) <= set(example.label_set)  # type: ignore[attr-defined]

    def test_a_single_call_row_costs_one_call(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L1'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert all(p.target_calls == 1 for p in predictions)

    def test_predictions_are_charged_a_cost(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L2'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert all(p.cost_usd > 0 for p in predictions)


class TestBestOfN:
    def test_issues_n_samples_and_charges_them(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        row = ctx.ladder.rows['best_of_4']
        predictions = run_row(row, items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert all(p.target_calls == 4 for p in predictions)
        assert all(p.cost_usd > 0 for p in predictions)

    def test_needs_a_known_base_row(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        row = RowSpec(
            name='bad',
            method='best_of_n',
            item_set=SUBSET,
            priority='P1',
            n=2,
            base='LX',
        )
        with pytest.raises(ValidationError, match='known base row'):
            run_row(row, items, ctx, item_set=SUBSET)  # type: ignore[arg-type]

    def test_needs_a_single_call_base_row(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        row = RowSpec(
            name='bad',
            method='best_of_n',
            item_set=SUBSET,
            priority='P1',
            n=2,
            base='L3',
        )
        with pytest.raises(ValidationError, match='known base row'):
            run_row(row, items, ctx, item_set=SUBSET)  # type: ignore[arg-type]


class TestHarnessRow:
    def test_writes_predictions_and_the_records_behind_them(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L3'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        outputs = tuple(iter_jsonl(ctx.paths.harness_outputs('L3', SUBSET)))
        assert len(outputs) == len(predictions)
        assert {o['example_id'] for o in outputs} == {p.example_id for p in predictions}

    def test_uses_the_four_stage_pipeline(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L3'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert all(p.target_calls == 4 for p in predictions)

    def test_emits_the_five_step_response_the_judge_reads(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L3'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert all(p.response_text.count('Step ') == 5 for p in predictions)

    def test_a_missing_bundle_says_which_stage_to_build(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        with pytest.raises(ValidationError, match='no harness bundle'):
            run_row(ctx.ladder.rows['L5'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]


class TestTeacherRow:
    def test_the_upper_bound_row_runs_with_a_second_fake_model(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        predictions = run_row(ctx.ladder.rows['L6'], items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert len(predictions) == len(items)

    def test_it_sends_the_matched_protocol(self, ctx: RunContext) -> None:
        row = ctx.ladder.rows['L6']
        assert row.method == 'teacher_direct'
        assert _teacher_mode(row, ctx) == 'matched_protocol'

    def test_native_video_falls_back_to_the_matched_protocol_off_gemini(
        self, ctx: RunContext, items: tuple[object, ...]
    ) -> None:
        row = ctx.ladder.rows['L6'].model_copy(
            update={'name': 'L6_native', 'inputs': 'native_video'}
        )

        assert _teacher_mode(row, ctx) == 'matched_protocol'
        predictions = run_row(row, items, ctx, item_set=SUBSET)  # type: ignore[arg-type]
        assert len(predictions) == len(items)

    def test_a_gemini_teacher_keeps_the_native_video_variant(
        self, ctx: RunContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(RunContext, 'teacher_provider', lambda _self: 'gemini')
        row = ctx.ladder.rows['L6'].model_copy(update={'inputs': 'native_video'})

        assert _teacher_mode(row, ctx) == 'native_video'


class TestDispatchGuards:
    def test_an_unknown_method_never_reaches_dispatch(self) -> None:
        with pytest.raises(pydantic.ValidationError, match='method'):
            RowSpec(
                name='x',
                method='telepathy',  # type: ignore[arg-type]
                item_set=SUBSET,
                priority='P2',
            )

    def test_an_empty_item_set_is_rejected(self, ctx: RunContext) -> None:
        with pytest.raises(ValidationError, match='no items selected'):
            run_row(ctx.ladder.rows['L0'], (), ctx)


class TestBundleResolution:
    def test_stage_one_is_the_committed_scaffold(self, ctx: RunContext) -> None:
        assert bundle_dir_for(1, ctx) == ctx.paths.bundle

    def test_later_stages_read_a_construction_run(self, ctx: RunContext) -> None:
        assert bundle_dir_for(3, ctx) == ctx.paths.stage_dir(3, 0) / 'final'

    def test_an_override_wins(self, ctx: RunContext, tmp_path: Path) -> None:
        overridden = ctx.__class__(**{**ctx.__dict__, 'bundle_override': tmp_path})
        assert bundle_dir_for(2, overridden) == tmp_path


class TestRunSettings:
    def test_the_fake_pipeline_uses_the_default_budget(self, ctx: RunContext) -> None:
        assert pipeline_config(ctx).call_hard_cap >= 4

    def test_the_cost_function_prices_tokens(self, ctx: RunContext) -> None:
        from emoharness.target.base import TargetResponse

        response = TargetResponse(
            text='x',
            model_id='m',
            input_tokens=1_000_000,
            output_tokens=0,
            latency_s=0.0,
        )
        assert cost_fn(ctx)(response) > 0
