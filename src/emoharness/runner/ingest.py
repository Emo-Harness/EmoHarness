"""Turning downloaded corpora into the manifests every later phase reads.

The benchmark side builds ``data/manifests/mme_emotion.jsonl`` plus its checksum
and annotation report; the construction side loads whichever external corpora are
actually present, assigns the three splits by source id, and reports per-source
label histograms (ADR-0003). A corpus whose root is missing is skipped with a
warning rather than failing the run: CREMA-D standing in for IEMOCAP is the
declared plan, not an accident.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.config import load_model
from emoharness.common.constants import SUBSET_SEED, SUBSET_SIZE_PER_TASK
from emoharness.common.errors import ValidationError
from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.data.benchmark import (
    annotation_report,
    build_benchmark_manifest,
    checksum_manifest,
)
from emoharness.data.label_set import normalize_label
from emoharness.data.manifest import Example, write_manifest
from emoharness.data.sources import ADAPTERS
from emoharness.data.splits import assign_external_splits, draw_subset, write_subset
from emoharness.data.tasks import Split
from emoharness.runner.context import RunContext
from emoharness.runner.items import EXTERNAL_SPLITS, load_benchmark
from emoharness.runner.paths import guard_write

_LOGGER = get_logger(__name__)


class ExternalSource(BaseModel):
    """One row of ``configs/data/sources.yaml``'s ``external`` list."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    origin: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    root: str = Field(min_length=1)
    splits: tuple[str, ...] = ()
    sessions: tuple[int, ...] = ()
    fallback_for: str | None = None

    def split_names(self) -> tuple[str, ...]:
        """What the adapter is asked to load: split names, or session numbers."""
        return self.splits or tuple(str(session) for session in self.sessions)


class BenchmarkSource(BaseModel):
    """The ``benchmark`` block of ``configs/data/sources.yaml``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    origin: str = Field(min_length=1)
    hf_repo: str = Field(min_length=1)
    root: str = Field(min_length=1)
    tasks: tuple[str, ...] = Field(min_length=1)


class SplitPlan(BaseModel):
    """Target sizes per seen capability kind, and the assignment seed (ADR-0003)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    per_kind: Mapping[str, int]
    seed: int

    def targets(self) -> Mapping[Split, int]:
        """The per-split targets, keyed by the enum the assigner expects."""
        unknown = tuple(name for name in self.per_kind if name not in EXTERNAL_SPLITS)
        if unknown:
            raise ValidationError(f'unknown split names in per_kind: {unknown}')
        return {EXTERNAL_SPLITS[name]: count for name, count in self.per_kind.items()}


class LeakageThresholds(BaseModel):
    """The four-check thresholds, mirrored from ``configs/data/sources.yaml``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    phash_hamming_threshold: int = Field(ge=0)
    transcript_jaccard_threshold: float = Field(ge=0.0, le=1.0)
    transcript_min_words: int = Field(ge=1)


class SourcesConfig(BaseModel):
    """``configs/data/sources.yaml`` as typed objects."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    benchmark: BenchmarkSource
    external: tuple[ExternalSource, ...]
    splits: SplitPlan
    leakage: LeakageThresholds


def load_sources(ctx: RunContext) -> SourcesConfig:
    """The corpus registry for this run."""
    return load_model(ctx.paths.sources_config, SourcesConfig)


def benchmark_root(ctx: RunContext, override: Path | None = None) -> Path:
    """Where the MME-Emotion annotation files live for this run."""
    if override is not None:
        return override
    if ctx.fake:
        return ctx.paths.fake_raw / 'mme_emotion'
    return ctx.paths.root / load_sources(ctx).benchmark.root


def write_checksums(path: Path, digests: Mapping[str, str]) -> Path:
    """Write a ``sha256sum``-style file: one ``<digest>  <relative path>`` per line."""
    body = ''.join(f'{digest}  {name}\n' for name, digest in sorted(digests.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')
    return path


def prepare_benchmark(
    ctx: RunContext, source: Path, hash_media: bool = False
) -> Mapping[str, Path]:
    """Build the benchmark manifest, its checksums, and its annotation report."""
    if not source.is_dir():
        raise ValidationError(f'benchmark root {source} does not exist')
    examples = build_benchmark_manifest(source, hash_media=hash_media)
    if not examples:
        raise ValidationError(f'no annotations found under {source}')
    manifest = guard_write(ctx.paths.benchmark_manifest, ctx.force)
    write_manifest(manifest, examples)
    checksums = write_checksums(
        guard_write(ctx.paths.benchmark_checksums, ctx.force), checksum_manifest(source)
    )
    report = guard_write(ctx.paths.benchmark_report, ctx.force)
    write_json_atomic(report, annotation_report(examples).model_dump(mode='json'))
    _LOGGER.info('benchmark manifest: %d items from %s', len(examples), source)
    return {'manifest': manifest, 'checksums': checksums, 'report': report}


def _load_source(spec: ExternalSource, root: Path) -> tuple[Example, ...]:
    adapter = ADAPTERS.get(spec.origin)
    if adapter is None:
        raise ValidationError(f'no adapter registered for origin {spec.origin!r}')
    return adapter.load(root, spec.split_names())


def load_external_sources(
    ctx: RunContext,
    config: SourcesConfig,
    roots: Mapping[str, Path] | None = None,
) -> tuple[Example, ...]:
    """Load every declared corpus whose root exists on this machine."""
    loaded: list[Example] = []
    for spec in config.external:
        root = (roots or {}).get(spec.origin) or ctx.paths.root / spec.root
        if not root.is_dir():
            _LOGGER.warning('skipping %s: %s does not exist', spec.origin, root)
            continue
        rows = _load_source(spec, root)
        _LOGGER.info('%s: %d examples from %s', spec.origin, len(rows), root)
        loaded.extend(rows)
    if not loaded:
        raise ValidationError(
            'no external corpus was found; check configs/data/sources.yaml'
        )
    return tuple(loaded)


def label_histograms(examples: Sequence[Example]) -> Mapping[str, Mapping[str, int]]:
    """Normalized label counts per origin, for the construction-corpus report."""
    counters: dict[str, Counter[str]] = {}
    for example in examples:
        counter = counters.setdefault(example.origin, Counter())
        counter.update(normalize_label(label) for label in example.labels)
    return {
        origin: dict(sorted(counter.items())) for origin, counter in counters.items()
    }


def _split_summary(examples: Sequence[Example]) -> Mapping[str, Any]:
    by_split = Counter(example.split.value for example in examples)
    by_kind: dict[str, Counter[str]] = {}
    for example in examples:
        kind = example.kind.value if example.kind is not None else 'unknown'
        by_kind.setdefault(kind, Counter())[example.split.value] += 1
    return {
        'counts': dict(sorted(by_split.items())),
        'by_kind': {k: dict(sorted(v.items())) for k, v in sorted(by_kind.items())},
    }


def prepare_external(
    ctx: RunContext,
    roots: Mapping[str, Path] | None = None,
    per_kind: Mapping[str, int] | None = None,
) -> Mapping[str, Path]:
    """Build the construction manifest, assign its splits, and report its labels.

    ``per_kind`` overrides the split targets in the config; a small corpus needs
    small targets, because a split whose target is never reached starves the ones
    below it in the assigner's preference order.
    """
    config = load_sources(ctx)
    if per_kind is not None:
        config = config.model_copy(
            update={'splits': config.splits.model_copy(update={'per_kind': per_kind})}
        )
    loaded = load_external_sources(ctx, config, roots)
    assigned = assign_external_splits(
        loaded, config.splits.targets(), config.splits.seed
    )
    if not assigned:
        raise ValidationError('split assignment produced no examples')
    manifest = guard_write(ctx.paths.external_manifest, ctx.force)
    write_manifest(manifest, assigned)
    splits = guard_write(ctx.paths.external_splits, ctx.force)
    write_json_atomic(
        splits,
        {
            'seed': config.splits.seed,
            'per_kind': dict(config.splits.per_kind),
            'sources': sorted({e.origin for e in assigned}),
            **_split_summary(assigned),
        },
    )
    report = guard_write(ctx.paths.external_report, ctx.force)
    write_json_atomic(
        report,
        {
            'loaded': len(loaded),
            'assigned': len(assigned),
            'label_histograms': label_histograms(assigned),
        },
    )
    _LOGGER.info(
        'external manifest: %d of %d items assigned', len(assigned), len(loaded)
    )
    return {'manifest': manifest, 'splits': splits, 'report': report}


def draw_and_write_subset(
    ctx: RunContext,
    size_per_task: int = SUBSET_SIZE_PER_TASK,
    seed: int = SUBSET_SEED,
) -> Path:
    """Draw the pre-registered benchmark subset and commit its id list."""
    examples = load_benchmark(ctx.paths)
    ids = draw_subset(examples, size_per_task, seed)
    if not ids:
        raise ValidationError('the benchmark manifest produced no subset ids')
    path = guard_write(ctx.paths.subset, ctx.force)
    write_subset(path, ids, seed, size_per_task)
    _LOGGER.info('subset: %d ids at %s', len(ids), path)
    return path
