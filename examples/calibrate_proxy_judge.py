"""Phase 5: measure how well the proxy judge tracks the reference judge (ADR-0007).

    python examples/calibrate_proxy_judge.py --row L3 --items external:val
    python examples/calibrate_proxy_judge.py --row L3 --items subset --fake

The Reasoning gate may only block a Builder candidate if this agreement clears
the threshold in ``configs/builder/proxy_judge.yaml``; the derived margin is
written alongside it in ``results/calibration/proxy_judge.json``.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.runner.calibration import calibrate_proxy_judge, write_calibration
from emoharness.runner.cli import build_parser, context_from, guard


def main(argv: Sequence[str] | None = None) -> int:
    """Score one sample with both judges and write the calibration report."""
    parser = build_parser(__doc__ or '')
    parser.add_argument('--row', default='L3', help='row whose responses are scored')
    parser.add_argument(
        '--items', default='external:val', help='item set to sample from'
    )
    parser.add_argument(
        '--sample', type=int, default=None, help='items to score (default: the config)'
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'calibrate_proxy_judge')
        payload = calibrate_proxy_judge(args.row, args.items, ctx, args.sample)
        print(write_calibration(payload, ctx))
        print(
            f'recognition agreement {payload["report"]["recognition_agreement"]:.3f} '
            f'(threshold {payload["min_agreement"]}), '
            f'gate margin {payload["gate_margin_points"]:.2f} points'
        )
        return 0 if payload['passes'] else 1

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
