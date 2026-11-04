"""Builder-loop wiring: diagnosis, candidate scoring, the gates, and the freeze."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emoharness.builder.loop import run_stage
from emoharness.builder.schema import CandidateScores
from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json
from emoharness.harness.allowlist import check_verification_source
from emoharness.runner.construction import (
    RegressionTracker,
    base_bundle_for,
    evaluate_external_test,
    forbidden_ids,
    gate_sample_ids,
    git_head,
    load_construction_data,
    make_diagnose,
    make_evaluate,
    run_construction,
    write_freeze_record,
)
from emoharness.runner.context import RunContext, build_context
from emoharness.runner.items import examples_for


class ScriptedEvaluate:
    """Scores the incumbent low and the first candidate high, then stops improving."""

    def __init__(self) -> None:
        self.seen: list[Path] = []

    def __call__(self, bundle: Path) -> CandidateScores:
        self.seen.append(bundle)
        first_candidate = len(self.seen) == 2
        return CandidateScores(
            name=bundle.name,
            recognition=0.6 if first_candidate else 0.4,
            reasoning=0.5,
            regression_rate=0.0,
            mean_calls=4.0,
            p95_latency_s=1.0,
        )


class TestGateSample:
    def test_is_deterministic_and_bounded(self, ctx: RunContext) -> None:
        val = examples_for('external:val', ctx.paths)
        first = gate_sample_ids(val, 2)
        assert first == gate_sample_ids(val, 2)
        assert len(first) == 2

    def test_takes_everything_when_the_split_is_smaller(self, ctx: RunContext) -> None:
        val = examples_for('external:val', ctx.paths)
        assert len(gate_sample_ids(val, 10_000)) == len(val)


class TestRegressionTracker:
    def test_starts_empty_and_charges_nothing(self) -> None:
        tracker = RegressionTracker()
        assert tracker.rate(Path('a'), frozenset({'x'})) == 0.0
        assert tracker.regression_set == frozenset()

    def test_only_a_promoted_bundle_enters_the_regression_set(self) -> None:
        tracker = RegressionTracker()
        tracker.rate(Path('a'), frozenset({'x', 'y'}))
        tracker.promote(Path('a'))
        assert tracker.regression_set == frozenset({'x', 'y'})

    def test_charges_a_candidate_that_loses_a_solved_item(self) -> None:
        tracker = RegressionTracker()
        tracker.rate(Path('a'), frozenset({'x', 'y'}))
        tracker.promote(Path('a'))
        assert tracker.rate(Path('b'), frozenset({'x'})) == pytest.approx(0.5)

    def test_promoting_an_unknown_bundle_changes_nothing(self) -> None:
        tracker = RegressionTracker()
        tracker.promote(Path('never-evaluated'))
        assert tracker.regression_set == frozenset()

    def test_the_set_only_grows(self) -> None:
        tracker = RegressionTracker()
        tracker.rate(Path('a'), frozenset({'x'}))
        tracker.promote(Path('a'))
        tracker.rate(Path('b'), frozenset({'y'}))
        tracker.promote(Path('b'))
        assert tracker.regression_set == frozenset({'x', 'y'})


class TestDiagnoseAndEvaluate:
    def test_a_packet_carries_the_bundle_and_the_editable_paths(
        self, ctx: RunContext
    ) -> None:
        data = load_construction_data(ctx)
        diagnose = make_diagnose(ctx, data, 2, RegressionTracker())
        packet = diagnose(ctx.paths.bundle, 0)
        assert 'prompts/extract_visual.md' in packet.bundle_files
        assert 'prompts/extract_visual.md' in packet.editable_paths
        assert 0.0 <= packet.build_recognition <= 1.0

    def test_a_packet_never_names_a_benchmark_item(self, ctx: RunContext) -> None:
        from emoharness.builder.packet import (
            assert_no_forbidden_ids,
            render_packet_text,
        )

        data = load_construction_data(ctx)
        packet = make_diagnose(ctx, data, 2, RegressionTracker())(ctx.paths.bundle, 0)
        text = render_packet_text(packet, ctx.builder.packet.target_input_tokens)
        assert_no_forbidden_ids(text, forbidden_ids(ctx))

    def test_evaluate_reports_every_gate_input(self, ctx: RunContext) -> None:
        data = load_construction_data(ctx)
        scores = make_evaluate(ctx, data, RegressionTracker())(ctx.paths.bundle)
        assert 0.0 <= scores.recognition <= 1.0
        assert scores.reasoning is not None
        assert scores.mean_calls == pytest.approx(4.0)
        assert scores.p95_latency_s >= 0.0

    def test_forbidden_ids_cover_ids_and_media_names(self, ctx: RunContext) -> None:
        found = set(forbidden_ids(ctx))
        example = examples_for('full', ctx.paths)[0]
        assert example.example_id in found
        assert Path(example.media_path).name in found


class TestLoop:
    def test_two_rounds_accept_one_patch(self, ctx: RunContext, tmp_path: Path) -> None:
        data = load_construction_data(ctx)
        evaluate = ScriptedEvaluate()
        result = run_stage(
            2,
            0,
            ctx.paths.bundle,
            tmp_path / 'loop',
            ctx.builder_client(2),
            ctx.builder.model_copy(update={'max_rounds_per_stage': 2}),
            ctx.gate_config(),
            make_diagnose(ctx, data, 2, RegressionTracker()),
            evaluate,
            check_verification_source,
            forbidden_ids(ctx),
        )
        accepted = [record.accepted for record in result.rounds]
        assert len(result.rounds) == 2
        assert accepted[0] is not None and accepted[1] is None

    def test_the_accepted_patch_is_what_the_final_bundle_holds(
        self, ctx: RunContext, tmp_path: Path
    ) -> None:
        data = load_construction_data(ctx)
        result = run_stage(
            2,
            0,
            ctx.paths.bundle,
            tmp_path / 'loop',
            ctx.builder_client(2),
            ctx.builder.model_copy(update={'max_rounds_per_stage': 2}),
            ctx.gate_config(),
            make_diagnose(ctx, data, 2, RegressionTracker()),
            ScriptedEvaluate(),
            check_verification_source,
            forbidden_ids(ctx),
        )
        text = (result.final_bundle / 'prompts' / 'extract_visual.md').read_text()
        assert 'Revision 0' in text
        assert '{noun}' in text

    def test_the_round_records_every_rejection_reason(
        self, ctx: RunContext, tmp_path: Path
    ) -> None:
        data = load_construction_data(ctx)
        run_stage(
            2,
            0,
            ctx.paths.bundle,
            tmp_path / 'loop',
            ctx.builder_client(2),
            ctx.builder.model_copy(update={'max_rounds_per_stage': 2}),
            ctx.gate_config(),
            make_diagnose(ctx, data, 2, RegressionTracker()),
            ScriptedEvaluate(),
            check_verification_source,
            forbidden_ids(ctx),
        )
        gates = read_json(tmp_path / 'loop' / 'round_01' / 'candidates.json')['gates']
        assert gates and not gates[0]['passed']


class TestRunConstruction:
    def test_stage_one_may_not_be_built(self, ctx: RunContext) -> None:
        with pytest.raises(ValidationError, match='human scaffold'):
            run_construction(1, 0, ctx)

    def test_stage_two_starts_from_the_committed_scaffold(
        self, ctx: RunContext
    ) -> None:
        assert base_bundle_for(2, ctx) == ctx.paths.bundle

    def test_stage_three_starts_from_the_first_stage_two_replicate(
        self, ctx: RunContext
    ) -> None:
        assert base_bundle_for(3, ctx) == ctx.paths.stage_dir(2, 0) / 'final'

    def test_a_missing_parent_bundle_is_reported(self, tmp_path: Path) -> None:
        fresh = build_context(tmp_path, fake=True, force=True)
        with pytest.raises(ValidationError, match='no parent harness bundle'):
            run_construction(3, 0, fresh)

    def test_writes_provenance_and_a_final_bundle(self, ctx: RunContext) -> None:
        result = run_construction(2, 0, ctx)
        out_dir = ctx.paths.stage_dir(2, 0)
        assert result.final_bundle.is_dir()
        assert (out_dir / 'loop_result.json').is_file()
        assert (out_dir / 'round_00' / 'packet.txt').is_file()
        assert result.rounds

    def test_final_bundle_manifest_records_the_construction_stage(
        self, ctx: RunContext
    ) -> None:
        result = run_construction(2, 0, ctx)
        manifest = json.loads(
            (result.final_bundle / 'manifest.json').read_text(encoding='utf-8')
        )
        assert manifest['stage'] == 2
        assert manifest['version'] == 'stage2-r0'
        assert len(manifest['parent']) == 64
        assert manifest['builder'] != 'human'
        assert 'accepted patch' in manifest['notes']


class TestFreeze:
    def test_records_the_content_hash_and_the_commit(self, ctx: RunContext) -> None:
        path = write_freeze_record(ctx.paths.bundle, ctx.paths, force=True)
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert len(payload['content_hash']) == 64
        assert payload['manifest_stage'] == 1
        assert 'frozen_at' in payload

    def test_git_head_tolerates_a_directory_that_is_not_a_checkout(
        self, tmp_path: Path
    ) -> None:
        assert git_head(tmp_path / 'nowhere') is None

    def test_external_test_is_evaluated_once_against_a_bundle(
        self, ctx: RunContext
    ) -> None:
        scores = evaluate_external_test(ctx.paths.bundle, ctx)
        assert scores.count == len(examples_for('external:ext_test', ctx.paths))
        assert 0.0 <= scores.recognition <= 1.0


def test_code_preflight_catches_runtime_fallback_before_review(
    ctx: RunContext,
    tmp_path: Path,
) -> None:
    from emoharness.harness.loader import write_bundle_copy
    from emoharness.runner.construction import make_bundle_check

    broken = write_bundle_copy(
        ctx.paths.bundle,
        tmp_path / 'broken_code',
        {
            'verification/alternative_hypotheses.py': "def rank(record, hypotheses):\n    return hypotheses['missing_key']\n",
        },
    )
    check = make_bundle_check(ctx, load_construction_data(ctx))
    assert 'contract probe' in check(broken)[0]


def test_builder_diagnosis_has_only_build_preservation_examples(
    ctx: RunContext,
) -> None:
    data = load_construction_data(ctx)
    packet = make_diagnose(ctx, data, 3, RegressionTracker())(ctx.paths.bundle, 0)
    build_ids = {example.example_id for example in data.build}
    assert all(
        item.example_id in build_ids and item.gold == item.predicted
        for item in packet.correct_examples
    )
    assert len(packet.correct_examples) <= ctx.builder.packet.max_correct_examples
    assert 'unavailable' in packet.tool_availability['face_crops']


def test_candidate_build_assessment_never_reads_validation(
    ctx: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from emoharness.runner import construction

    data = load_construction_data(ctx)
    allowed = {example.example_id for example in data.build}
    run = construction.run_harness

    def build_only(examples, *args, **kwargs):
        assert {example.example_id for example in examples} <= allowed
        return run(examples, *args, **kwargs)

    monkeypatch.setattr(construction, 'run_harness', build_only)
    assessment = construction.make_build_assessment(ctx, data)(
        ctx.paths.bundle, ctx.paths.bundle
    )
    assert assessment.before == assessment.after
    assert assessment.improved_count == assessment.regressed_count == 0
