"""Phase 6: run one construction stage's Builder loop, and freeze what it produced.

    python examples/build_harness.py --stage 2 --replicate 0
    python examples/build_harness.py --stage 3 --freeze
    python examples/build_harness.py --stage 2 --fake --root /tmp/emoharness-fake

Stage 2 starts from the Stage 1 scaffold; Stage 3 starts from the first
completed Stage 2 replicate, declared before the runs start.
``--freeze`` writes ``results/freeze.json``, evaluates D_ext-test once, and
prints the git tag command to run.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.common.io import write_json_atomic
from emoharness.runner.cli import (
    add_target_options,
    build_parser,
    context_from,
    guard,
)
from emoharness.runner.construction import (
    evaluate_external_test,
    run_construction,
    write_freeze_record,
)
from emoharness.runner.context import RunContext
from emoharness.runner.paths import guard_write

FREEZE_TAG = 'harness-frozen'


def freeze(bundle_dir: str, stage: int, ctx: RunContext) -> int:
    """Record the freeze, evaluate D_ext-test once, and print the tag command."""
    from pathlib import Path

    path = write_freeze_record(Path(bundle_dir), ctx.paths, ctx.force)
    print(path)
    scores = evaluate_external_test(Path(bundle_dir), ctx)
    metrics = guard_write(ctx.paths.metrics / f'ext_test_stage{stage}.json', ctx.force)
    write_json_atomic(metrics, scores.model_dump(mode='json'))
    print(metrics)
    print(f'now run: git tag -a {FREEZE_TAG} -m "stage {stage} frozen"')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage's loop, then optionally freeze it."""
    parser = build_parser(__doc__ or '')
    add_target_options(parser)
    parser.add_argument('--stage', type=int, default=2, help='2 or 3')
    parser.add_argument(
        '--rounds', type=int, default=None, help='cap on Builder rounds this run'
    )
    parser.add_argument(
        '--freeze',
        action='store_true',
        help='write the freeze record and evaluate D_ext-test once',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, f'build_harness:stage{args.stage}', rounds=args.rounds)
        result = run_construction(args.stage, args.replicate, ctx)
        print(f'stage {args.stage} replicate {args.replicate}: {result.stopped_reason}')
        print(f'rounds: {len(result.rounds)}; final bundle: {result.final_bundle}')
        if args.freeze:
            return freeze(str(result.final_bundle), args.stage, ctx)
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
