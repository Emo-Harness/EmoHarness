"""The runner's leakage sweep: gathering evidence, running the checks, writing artifacts."""

from __future__ import annotations

from emoharness.common.io import read_json
from emoharness.data.leakage import LeakageFlag, LeakageReport
from emoharness.runner.context import RunContext
from emoharness.runner.items import load_external
from emoharness.runner.leakage import drop_flagged_examples, leakage_sweep


class TestLeakageSweep:
    def test_writes_a_report_and_a_coverage_summary(self, ctx: RunContext) -> None:
        report, written = leakage_sweep(ctx)
        assert set(written) == {'report', 'coverage'}
        assert report.construction_count and report.benchmark_count

    def test_synthetic_corpora_are_clean(self, ctx: RunContext) -> None:
        report, _ = leakage_sweep(ctx)
        assert report.flags == ()

    def test_coverage_records_what_each_check_saw(self, ctx: RunContext) -> None:
        _, written = leakage_sweep(ctx)
        coverage = read_json(written['coverage'])
        assert coverage['construction']['phash'] > 0
        assert coverage['benchmark']['transcript'] > 0


class TestDropFlaggedExamples:
    def test_dropping_flagged_clips_rewrites_the_manifest(
        self, ctx: RunContext
    ) -> None:
        before = load_external(ctx.paths)
        report = LeakageReport(
            flags=(
                LeakageFlag(
                    construction_id=before[0].example_id,
                    benchmark_id='mme_sa_000001',
                    check='media_hash',
                    score=1.0,
                ),
            ),
            checked_pairs=1,
            construction_count=len(before),
            benchmark_count=1,
            thresholds={},
        )
        drop_flagged_examples(ctx, report)
        after = load_external(ctx.paths)
        assert len(after) == len(before) - 1
        assert before[0].example_id not in {e.example_id for e in after}
