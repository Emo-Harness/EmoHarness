"""Phase 5: draw the pre-registered subset and write ``docs/preregistration.md``.

    python examples/preregister.py
    python examples/preregister.py --fake --root /tmp/emoharness-fake

The document is generated from the artifacts it freezes, so its hashes always
match the files on disk. Commit it and tag ``prereg-v1`` before the first
Builder round (ADR-0012). With ``--fake`` it is written to
``results/fake/preregistration.md`` instead, so a smoke run never touches docs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from emoharness.common.constants import SUBSET_SEED, SUBSET_SIZE_PER_TASK
from emoharness.runner.cli import build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.ingest import draw_and_write_subset
from emoharness.runner.paths import PREREGISTRATION_FILE
from emoharness.runner.preregistration import (
    build_preregistration,
    write_preregistration,
)


def target_path(ctx: RunContext) -> Path:
    """Where the document goes: ``docs/`` normally, a scratch file with ``--fake``."""
    if ctx.fake:
        return ctx.paths.fake_results / PREREGISTRATION_FILE
    return ctx.paths.preregistration


def main(argv: Sequence[str] | None = None) -> int:
    """Commit the subset id list, then render the pre-registration."""
    parser = build_parser(__doc__ or '')
    parser.add_argument('--size-per-task', type=int, default=SUBSET_SIZE_PER_TASK)
    parser.add_argument('--seed', type=int, default=SUBSET_SEED)
    parser.add_argument(
        '--skip-subset',
        action='store_true',
        help='reuse the committed subset instead of drawing it again',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'preregister')
        if not args.skip_subset:
            print(draw_and_write_subset(ctx, args.size_per_task, args.seed))
        print(
            write_preregistration(
                target_path(ctx), build_preregistration(ctx), ctx.force
            )
        )
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
