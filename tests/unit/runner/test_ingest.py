"""Ingestion and the leakage gate: manifests, splits, checksums, and the four checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json
from emoharness.data.fakes import make_fake_benchmark, make_fake_external
from emoharness.data.tasks import Kind, Split, Task
from emoharness.runner.calibration import (
    calibrate_proxy_judge,
    gate_margin_points,
    write_calibration,
)
from emoharness.runner.context import RunContext, build_context
from emoharness.runner.ingest import (
    SourcesConfig,
    benchmark_root,
    draw_and_write_subset,
    label_histograms,
    load_sources,
    prepare_benchmark,
    prepare_external,
    write_checksums,
)
from emoharness.runner.items import examples_for, load_benchmark, load_external
from emoharness.runner.perception import verify_cache
from emoharness.runner.rows import run_row


class TestSourcesConfig:
    def test_parses_the_committed_registry(self, ctx: RunContext) -> None:
        config = load_sources(ctx)
        assert config.benchmark.origin == 'mme_emotion'
        assert {spec.origin for spec in config.external} >= {'meld', 'mosei'}

    def test_split_targets_map_onto_the_enum(self, ctx: RunContext) -> None:
        assert set(load_sources(ctx).splits.targets()) == {
            Split.BUILD,
            Split.VAL,
            Split.EXT_TEST,
        }

    def test_an_unknown_split_name_is_rejected(self, ctx: RunContext) -> None:
        config = load_sources(ctx)
        broken = config.splits.model_copy(update={'per_kind': {'holdout': 1}})
        with pytest.raises(ValidationError, match='unknown split names'):
            broken.targets()

    def test_a_malformed_registry_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'sources.yaml'
        path.write_text('benchmark: {}\n', encoding='utf-8')
        with pytest.raises(ValidationError, match=str(path)):
            load_model(path, SourcesConfig)

    def test_an_iemocap_style_source_passes_session_numbers(
        self, ctx: RunContext
    ) -> None:
        spec = next(s for s in load_sources(ctx).external if s.origin == 'iemocap')
        assert spec.split_names() == ('1', '2', '3', '4')


class TestPrepareBenchmark:
    def test_writes_the_manifest_checksums_and_report(self, ctx: RunContext) -> None:
        written = prepare_benchmark(ctx, benchmark_root(ctx))
        assert set(written) == {'manifest', 'checksums', 'report'}
        report = read_json(written['report'])
        assert set(report['task_counts']) == {task.value for task in Task}

    def test_the_checksum_file_is_sha256sum_shaped(self, ctx: RunContext) -> None:
        written = prepare_benchmark(ctx, benchmark_root(ctx))
        first = written['checksums'].read_text(encoding='utf-8').splitlines()[0]
        digest, name = first.split('  ', 1)
        assert len(digest) == 64 and name.endswith('.json')

    def test_a_missing_source_is_reported(
        self, ctx: RunContext, tmp_path: Path
    ) -> None:
        with pytest.raises(ValidationError, match='does not exist'):
            prepare_benchmark(ctx, tmp_path / 'nowhere')

    def test_a_source_without_annotations_is_reported(
        self, ctx: RunContext, tmp_path: Path
    ) -> None:
        (tmp_path / 'empty').mkdir()
        with pytest.raises(ValidationError, match='cannot read JSON'):
            prepare_benchmark(ctx, tmp_path / 'empty')

    def test_checksums_are_written_sorted(self, tmp_path: Path) -> None:
        path = write_checksums(tmp_path / 'sums', {'b': 'y' * 64, 'a': 'x' * 64})
        assert path.read_text(encoding='utf-8').splitlines()[0].endswith('  a')


class TestPrepareExternal:
    def test_assigns_all_three_splits(self, ctx: RunContext) -> None:
        splits = read_json(ctx.paths.external_splits)
        assert set(splits['counts']) == {'build', 'val', 'ext_test'}
        assert all(count > 0 for count in splits['counts'].values())

    def test_reports_per_source_label_histograms(self, ctx: RunContext) -> None:
        report = read_json(ctx.paths.external_report)
        assert set(report['label_histograms']) >= {'meld', 'mosei'}
        assert report['assigned'] <= report['loaded']

    def test_every_seen_kind_is_represented(self, ctx: RunContext) -> None:
        kinds = {example.kind for example in load_external(ctx.paths)}
        assert kinds == set(Kind)

    def test_label_histograms_normalize(self, ctx: RunContext) -> None:
        histograms = label_histograms(load_external(ctx.paths))
        assert all(key == key.lower() for row in histograms.values() for key in row)

    def test_a_missing_corpus_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        ctx = build_context(tmp_path, fake=True, force=True)
        roots = make_fake_external(ctx.paths.fake_raw, per_source=2)
        prepare_external(
            ctx, {'meld': roots['meld']}, {'build': 1, 'val': 1, 'ext_test': 1}
        )
        assert {e.origin for e in load_external(ctx.paths)} == {'meld'}

    def test_no_corpus_at_all_is_fatal(self, tmp_path: Path) -> None:
        ctx = build_context(tmp_path, fake=True, force=True)
        with pytest.raises(ValidationError, match='no external corpus'):
            prepare_external(ctx, {})


class TestSubset:
    def test_draws_ids_from_the_manifest(self, ctx: RunContext) -> None:
        path = draw_and_write_subset(ctx)
        ids = set(read_json(path)['ids'])
        assert ids <= {e.example_id for e in load_benchmark(ctx.paths)}

    def test_is_reproducible(self, ctx: RunContext) -> None:
        first = read_json(draw_and_write_subset(ctx))['subset_hash']
        second = read_json(draw_and_write_subset(ctx))['subset_hash']
        assert first == second

    def test_needs_a_manifest(self, tmp_path: Path) -> None:
        ctx = build_context(tmp_path, fake=True, force=True)
        with pytest.raises(ValidationError, match='manifest'):
            draw_and_write_subset(ctx)


class TestPreprocessVerification:
    def test_the_prepared_cache_verifies(self, ctx: RunContext) -> None:
        examples = load_benchmark(ctx.paths)
        assert verify_cache(examples, ctx.clip_cache(), ctx.preprocess) == ()

    def test_a_fresh_root_has_nothing_cached(
        self, ctx: RunContext, tmp_path: Path
    ) -> None:
        from emoharness.perceive.cache import PreprocessCache

        examples = load_benchmark(ctx.paths)[:2]
        broken = verify_cache(examples, PreprocessCache(tmp_path), ctx.preprocess)
        assert len(broken) == 2


class TestCalibration:
    def test_compares_two_judges_over_a_sample(self, ctx: RunContext) -> None:
        run_row(
            ctx.ladder.rows['L3'], examples_for('full', ctx.paths), ctx, item_set='full'
        )
        payload = calibrate_proxy_judge('L3', 'full', ctx, sample=6)
        assert payload['sample_size'] == 6
        assert 0.0 <= payload['report']['step_agreement'] <= 1.0
        assert payload['gate_margin_points'] >= 2.0

    def test_writes_the_calibration_artifact(self, ctx: RunContext) -> None:
        run_row(
            ctx.ladder.rows['L3'], examples_for('full', ctx.paths), ctx, item_set='full'
        )
        path = write_calibration(calibrate_proxy_judge('L3', 'full', ctx, 4), ctx)
        assert path.name == 'proxy_judge.json'
        assert read_json(path)['proxy_model']

    def test_the_gate_margin_never_falls_below_the_floor(self) -> None:
        from emoharness.evaluation.judge import CalibrationReport

        report = CalibrationReport(
            n=1,
            step_agreement=1.0,
            recognition_agreement=1.0,
            reasoning_offset=0.0,
            reasoning_std=0.0,
        )
        assert gate_margin_points(report) == 2.0


class TestFakeBenchmarkRoot:
    def test_points_into_the_fake_tree(self, ctx: RunContext) -> None:
        assert benchmark_root(ctx).parent == ctx.paths.fake_raw

    def test_an_override_wins(self, ctx: RunContext, tmp_path: Path) -> None:
        assert benchmark_root(ctx, tmp_path) == tmp_path

    def test_a_real_run_reads_the_registry(self, tmp_path: Path) -> None:
        real = build_context(tmp_path, fake=False, force=True)
        make_fake_benchmark(real.paths.fake_raw / 'mme_emotion', per_task=1)
        assert benchmark_root(real).name == 'mme_emotion'
