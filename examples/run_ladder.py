"""Phases 4 and 7: run ladder rows over an item set and write raw predictions.

    python examples/run_ladder.py --rows L0,L3 --items full
    python examples/run_ladder.py --rows L3 --items external:val --bundle harness
    python examples/run_ladder.py --rows L0,L1,L2,L3 --items subset --fake

Rows are named as in ``configs/experiments/ladder.yaml``. ``--items`` overrides
the item set the row declares, which is how the same row is run on D_val during
construction and on the benchmark afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.common.errors import ValidationError
from emoharness.runner.cli import (
    add_target_options,
    build_parser,
    context_from,
    guard,
)
from emoharness.runner.context import RunContext
from emoharness.runner.items import examples_for
from emoharness.runner.rows import run_row


def row_names(raw: str) -> tuple[str, ...]:
    """Split a ``--rows`` value such as ``L0,L3`` into row names."""
    names = tuple(part.strip() for part in raw.split(',') if part.strip())
    if not names:
        raise ValidationError('--rows needs at least one row name')
    return names


def run_one(name: str, item_set: str | None, ctx: RunContext) -> int:
    """Run one row and report how many items it produced."""
    row = ctx.ladder.rows.get(name)
    if row is None:
        raise ValidationError(
            f'unknown row {name!r}; see configs/experiments/ladder.yaml'
        )
    chosen = item_set or row.item_set
    examples = examples_for(chosen, ctx.paths)
    predictions = run_row(
        row, examples, ctx.with_experiment(f'row:{name}'), item_set=chosen
    )
    print(f'{name} {chosen}: {len(predictions)} predictions')
    return len(predictions)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every requested row in order."""
    parser = build_parser(__doc__ or '')
    add_target_options(parser)
    parser.add_argument('--rows', required=True, help='comma-separated row names')
    parser.add_argument(
        '--items', default=None, help='full, subset, or external:<split>'
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'run_ladder')
        for name in row_names(args.rows):
            run_one(name, args.items, ctx)
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
