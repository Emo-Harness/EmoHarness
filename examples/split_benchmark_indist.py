"""In-distribution construction split: carve build/val out of the benchmark's seen tasks.

    python examples/split_benchmark_indist.py
    python examples/split_benchmark_indist.py --per-kind 130,130,40 --seed 20260907

This is the original README's protocol (build 10% / val 10% of the three seen
task families, test on the rest), kept as a same-distribution check of the
Builder loop. It is *not* the pre-registered protocol of ADR-0001/ADR-0003,
whose construction data comes from outside MME-Emotion; scores from a run
prepared this way are not leaderboard-comparable and must be labelled as
in-distribution wherever they are reported.

What it writes, all under the run root:

- ``data/manifests/external.jsonl`` and ``data/splits/external_splits.json``:
  the seen-task items assigned to build / val / ext_test, tagged with their
  seen ``kind`` so the construction loop reads them exactly as it would read
  an external corpus;
- ``data/manifests/mme_emotion.jsonl``: rewritten without those items, so
  every benchmark row, the subset draw, and the packet firewall see only the
  test portion. The untouched manifest is kept as ``mme_emotion.full.jsonl``.

Run ``prepare_benchmark`` first; run this before ``leakage_report`` and
``preregister``.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from typing import Any

from emoharness.common.constants import SUBSET_SEED
from emoharness.common.errors import ValidationError
from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.data.manifest import Example, read_manifest, write_manifest
from emoharness.data.splits import assign_external_splits
from emoharness.data.tasks import TASK_KIND
from emoharness.runner.cli import announce, build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.ingest import label_histograms
from emoharness.runner.items import EXTERNAL_SPLITS
from emoharness.runner.paths import guard_write

#: Marks these rows as benchmark-derived wherever ``origin`` is inspected; no
#: leakage source-id rule is registered for it, on purpose.
INDIST_ORIGIN = 'mme_emotion_indist'
FULL_MANIFEST_NAME = 'mme_emotion.full.jsonl'
#: 10% / 10% of the 3,900 seen-task items per kind, plus a small once-only
#: ext_test the freeze step evaluates; 130 + 130 + 40 per kind.
DEFAULT_PER_KIND: Mapping[str, int] = {'build': 130, 'val': 130, 'ext_test': 40}

_LOGGER = get_logger(__name__)


def parse_per_kind(raw: str | None) -> Mapping[str, int]:
    """Parse ``--per-kind build,val,ext_test``."""
    if raw is None:
        return DEFAULT_PER_KIND
    parts = tuple(part.strip() for part in raw.split(','))
    if len(parts) != len(EXTERNAL_SPLITS):
        raise ValidationError('--per-kind takes three counts: build,val,ext_test')
    try:
        counts = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValidationError(f'--per-kind takes integers, got {raw!r}') from exc
    return dict(zip(EXTERNAL_SPLITS, counts, strict=True))


def seen_as_construction(examples: Sequence[Example]) -> tuple[Example, ...]:
    """Seen-task benchmark rows re-tagged with their construction ``kind``."""
    return tuple(
        example.model_copy(
            update={'kind': TASK_KIND[example.task], 'origin': INDIST_ORIGIN}
        )
        for example in examples
        if example.task is not None and example.task in TASK_KIND
    )


def split_summary(assigned: Sequence[Example]) -> Mapping[str, Any]:
    """Counts per split, and per split and kind."""
    by_split: dict[str, int] = {}
    by_split_kind: dict[str, dict[str, int]] = {}
    for example in assigned:
        split = example.split.value
        kind = example.kind.value if example.kind is not None else 'unknown'
        by_split[split] = by_split.get(split, 0) + 1
        by_split_kind.setdefault(split, {})
        by_split_kind[split][kind] = by_split_kind[split].get(kind, 0) + 1
    return {'by_split': by_split, 'by_split_kind': by_split_kind}


def split_benchmark(
    ctx: RunContext, per_kind: Mapping[str, int], seed: int
) -> Mapping[str, Any]:
    """Assign the construction splits and rewrite both manifests."""
    manifest = ctx.paths.benchmark_manifest
    if not manifest.is_file():
        raise ValidationError(
            f'no benchmark manifest at {manifest}; run prepare_benchmark'
        )
    full_copy = manifest.with_name(FULL_MANIFEST_NAME)
    if not full_copy.exists():
        shutil.copyfile(manifest, full_copy)
    examples = read_manifest(full_copy)
    targets = {EXTERNAL_SPLITS[name]: count for name, count in per_kind.items()}
    assigned = assign_external_splits(seen_as_construction(examples), targets, seed)
    if not assigned:
        raise ValidationError('split assignment produced no construction items')
    taken = frozenset(example.example_id for example in assigned)
    remaining = tuple(e for e in examples if e.example_id not in taken)
    if len(remaining) + len(assigned) != len(examples):
        raise ValidationError(
            'construction ids and test ids do not partition the manifest'
        )

    external = guard_write(ctx.paths.external_manifest, ctx.force)
    write_manifest(external, assigned)
    splits = guard_write(ctx.paths.external_splits, ctx.force)
    write_json_atomic(
        splits,
        {
            'seed': seed,
            'per_kind': dict(per_kind),
            'sources': [INDIST_ORIGIN],
            'protocol': 'in-distribution (original README 10/10/80); '
            'not ADR-0001/ADR-0003',
            **split_summary(assigned),
        },
    )
    report = guard_write(ctx.paths.external_report, ctx.force)
    write_json_atomic(
        report,
        {
            'loaded': len(examples),
            'assigned': len(assigned),
            'test_remaining': len(remaining),
            'label_histograms': label_histograms(assigned),
        },
    )
    write_manifest(manifest, remaining)
    _LOGGER.info(
        'in-distribution split: %d construction items, %d test items remain',
        len(assigned),
        len(remaining),
    )
    return {
        'external_manifest': external,
        'external_splits': splits,
        'external_report': report,
        'benchmark_manifest_test_only': manifest,
        'benchmark_manifest_full': full_copy,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Carve build/val/ext_test out of the benchmark's seen tasks."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--per-kind',
        default=None,
        help='split targets per seen kind: build,val,ext_test',
    )
    parser.add_argument('--seed', type=int, default=SUBSET_SEED)
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'split_benchmark_indist')
        announce(split_benchmark(ctx, parse_per_kind(args.per_kind), args.seed))
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
