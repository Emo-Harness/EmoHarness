"""Phase 2 gate: the four construction-versus-benchmark leakage checks (ADR-0003).

    python examples/leakage_report.py --drop
    python examples/leakage_report.py --fake --root /tmp/emoharness-fake

The visual and transcript checks read the preprocessing cache; with ``--fake``
the cache is filled first, so the pHash check runs over the generated frames.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.runner.cli import announce, build_parser, context_from, guard
from emoharness.runner.leakage import drop_flagged_examples, leakage_sweep


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check and write ``data/checksums/leakage_report.json``."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--drop',
        action='store_true',
        help='rewrite the construction manifest without any flagged clip',
    )
    parser.add_argument(
        '--fail-on-flags',
        action='store_true',
        help='exit non-zero while any flag is unresolved',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'leakage_report')
        report, written = leakage_sweep(ctx)
        announce(written)
        by_check: dict[str, int] = {}
        for flag in report.flags:
            by_check[flag.check] = by_check.get(flag.check, 0) + 1
        print(
            f'{len(report.flags)} flags over {report.checked_pairs} pairs: {by_check}'
        )
        if args.drop and report.flags:
            print(drop_flagged_examples(ctx, report))
            return 0
        return 1 if args.fail_on_flags and report.flags else 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
