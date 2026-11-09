"""Phase 1: build the benchmark manifest, its checksums, and its annotation report.

    python examples/data/prepare_benchmark.py --source /data/mme_emotion
    python examples/data/prepare_benchmark.py --fake --root /tmp/emoharness-fake

With ``--fake`` a synthetic MME-Emotion tree is generated under ``data/fake/``
first, so the whole ingestion path runs without the 6.3 GB download.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from emoharness.data.fakes import DEFAULT_PER_TASK, make_fake_benchmark
from emoharness.runner.cli import announce, build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.ingest import benchmark_root, prepare_benchmark

BENCHMARK_ROOT_ENV = 'MME_EMOTION_ROOT'


def resolve_source(source: Path | None, ctx: RunContext) -> Path:
    """The annotation root: the flag, then the environment, then the config."""
    if source is not None:
        return source
    from_env = os.environ.get(BENCHMARK_ROOT_ENV)
    if from_env and not ctx.fake:
        return Path(from_env)
    return benchmark_root(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    """Build ``data/manifests/mme_emotion.jsonl`` and its companions."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--source',
        type=Path,
        default=None,
        help=f'annotation root (${BENCHMARK_ROOT_ENV})',
    )
    parser.add_argument(
        '--per-task',
        type=int,
        default=DEFAULT_PER_TASK,
        help='items per task to synthesize with --fake',
    )
    parser.add_argument(
        '--hash-media', action='store_true', help='record a SHA-256 per media file'
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'prepare_benchmark')
        source = resolve_source(args.source, ctx)
        if args.fake:
            make_fake_benchmark(source, args.per_task)
        announce(prepare_benchmark(ctx, source, hash_media=args.hash_media))
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
