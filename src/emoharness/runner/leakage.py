"""The four construction-versus-benchmark leakage checks, run over the manifests.

Exact media hashes come from the files themselves; the visual and transcript
checks read the shared preprocessing cache, so a clip that has not been
preprocessed simply contributes nothing to those two checks rather than being
silently treated as clean - the report records how many clips each check saw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, get_args

from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.data.leakage import (
    CheckName,
    LeakageInputs,
    LeakageReport,
    LeakageThresholds,
    drop_flagged,
    run_leakage_checks,
    write_report,
)
from emoharness.data.manifest import Example, write_manifest
from emoharness.runner.context import RunContext
from emoharness.runner.ingest import load_sources
from emoharness.runner.items import load_benchmark, load_external
from emoharness.runner.paths import guard_write
from emoharness.runner.perception import (
    media_hashes_by_id,
    phashes_by_id,
    transcripts_by_id,
)

COVERAGE_FILE: Final = 'leakage_coverage.json'

# The check names, typed once from CheckName rather than re-spelled as string
# literals wherever the coverage summary keys its two per-side mappings.
_CHECK_NAMES: Final[tuple[CheckName, ...]] = get_args(CheckName)
MEDIA_HASH_CHECK: Final[CheckName] = _CHECK_NAMES[0]
PHASH_CHECK: Final[CheckName] = _CHECK_NAMES[1]
TRANSCRIPT_CHECK: Final[CheckName] = _CHECK_NAMES[2]

_LOGGER = get_logger(__name__)


def materialize_cache(examples: Sequence[Example], ctx: RunContext) -> int:
    """Fill the preprocessing cache for these examples; returns how many were done."""
    cache, preprocess = ctx.clip_cache(), ctx.clip_preprocessor()
    for example in examples:
        preprocess(example, cache, ctx.preprocess)
    return len(examples)


def _evidence(
    examples: Sequence[Example], ctx: RunContext
) -> tuple[Mapping[str, str], Mapping[str, tuple[str, ...]], Mapping[str, str]]:
    cache = ctx.clip_cache()
    return (
        media_hashes_by_id(examples),
        phashes_by_id(examples, cache, ctx.preprocess),
        transcripts_by_id(examples, cache, ctx.preprocess),
    )


def coverage(
    construction: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> Mapping[str, Any]:
    """How many clips each side contributed to each check."""
    return {
        'construction': {name: len(value) for name, value in construction.items()},
        'benchmark': {name: len(value) for name, value in benchmark.items()},
    }


def leakage_sweep(
    ctx: RunContext, materialize: bool | None = None
) -> tuple[LeakageReport, Mapping[str, Path]]:
    """Run every available check over the two manifests and write the report."""
    config = load_sources(ctx)
    benchmark = load_benchmark(ctx.paths)
    external = load_external(ctx.paths)
    fill = ctx.fake if materialize is None else materialize
    if fill:
        materialize_cache(benchmark, ctx)
        materialize_cache(external, ctx)
    c_hashes, c_phashes, c_text = _evidence(external, ctx)
    b_hashes, b_phashes, b_text = _evidence(benchmark, ctx)
    inputs = LeakageInputs(
        construction=external,
        benchmark=benchmark,
        media_hashes={**c_hashes, **b_hashes},
        phashes={**c_phashes, **b_phashes},
        transcripts={**c_text, **b_text},
    )
    thresholds = LeakageThresholds(
        phash_hamming=config.leakage.phash_hamming_threshold,
        transcript_jaccard=config.leakage.transcript_jaccard_threshold,
        transcript_min_words=config.leakage.transcript_min_words,
    )
    report = run_leakage_checks(inputs, thresholds)
    path = guard_write(ctx.paths.leakage_report, ctx.force)
    write_report(path, report)
    coverage_path = guard_write(ctx.paths.checksums / COVERAGE_FILE, ctx.force)
    write_json_atomic(
        coverage_path,
        coverage(
            {
                MEDIA_HASH_CHECK: c_hashes,
                PHASH_CHECK: c_phashes,
                TRANSCRIPT_CHECK: c_text,
            },
            {
                MEDIA_HASH_CHECK: b_hashes,
                PHASH_CHECK: b_phashes,
                TRANSCRIPT_CHECK: b_text,
            },
        ),
    )
    _LOGGER.info(
        'leakage: %d flags over %d pairs', len(report.flags), report.checked_pairs
    )
    return report, {'report': path, 'coverage': coverage_path}


def drop_flagged_examples(ctx: RunContext, report: LeakageReport) -> Path:
    """Rewrite the construction manifest without any flagged clip."""
    kept = drop_flagged(load_external(ctx.paths), report)
    write_manifest(ctx.paths.external_manifest, kept)
    _LOGGER.info('dropped %d flagged clips', len(report.flagged_construction_ids()))
    return ctx.paths.external_manifest
