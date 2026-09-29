"""Phase 8: build every table from the raw predictions on disk.

    python examples/analyze.py
    python examples/analyze.py --rows L0,L3,L5 --items full
    python examples/analyze.py --fake --root /tmp/emoharness-fake --resamples 200

Writes ``ladder.json``, ``ladder.md``, ``per_task.json``, ``hypotheses.json``,
``multilabel.json``, and ``cost.json`` under ``results/metrics/``. Every number
traces back to a row, an item set, and a raw-prediction file.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.runner.analysis import analyze
from emoharness.runner.cli import (
    DEFAULT_RESAMPLES,
    announce,
    build_parser,
    context_from,
    guard,
)
from emoharness.runner.context import RunContext
from emoharness.runner.results import discover_rows


def selected(
    rows: str | None, items: str | None, ctx: RunContext
) -> tuple[tuple[str, str], ...] | None:
    """The row/item-set pairs to analyze, or None for everything on disk."""
    if rows is None and items is None:
        return None
    wanted = {part.strip() for part in (rows or '').split(',') if part.strip()}
    return tuple(
        (row, item_set)
        for row, item_set in discover_rows(ctx.paths)
        if (not wanted or row in wanted) and (items is None or item_set == items)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the metrics artifacts."""
    parser = build_parser(__doc__ or '')
    parser.add_argument('--rows', default=None, help='comma-separated row names')
    parser.add_argument('--items', default=None, help='restrict to one item set')
    parser.add_argument(
        '--resamples',
        type=int,
        default=DEFAULT_RESAMPLES,
        help='bootstrap resamples (the reported tables use 10,000)',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'analyze')
        announce(analyze(ctx, selected(args.rows, args.items, ctx)))
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
