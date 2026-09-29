"""Phase 7: prepare the official judge's inputs, print its commands, and run them.

    python examples/judge_official.py --row L5 --items full
    python examples/judge_official.py --row L5 --items full --execute
    python examples/judge_official.py --row L3 --items subset --fake

Without ``--execute`` nothing is run: the per-task response files are written and
the three official commands are printed for review, so the ledger can be checked
before any paid call. The judge's own evidence directory is passed through as an
argument and never opened here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from emoharness.runner.cli import announce, build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.judging import (
    fake_judge,
    official_commands,
    prepare_official_inputs,
    run_official,
)

REPO_ENV = 'MME_EMOTION_REPO'
VIDEO_ENV = 'MME_EMOTION_ROOT'
EVIDENCE_ENV = 'MME_EMOTION_EVIDENCE_DIR'


def _path_from(value: Path | None, env: str, fallback: Path) -> Path:
    if value is not None:
        return value
    found = os.environ.get(env)
    return Path(found) if found else fallback


def execute(args: argparse.Namespace, ctx: RunContext) -> int:
    """Build the official commands and either print or run them."""
    repo = _path_from(args.repo, REPO_ENV, ctx.paths.root / 'MME-Emotion')
    videos = _path_from(args.video_dir, VIDEO_ENV, ctx.paths.raw)
    evidence = args.evidence_dir or os.environ.get(EVIDENCE_ENV)
    commands = official_commands(
        args.row, args.items, ctx, repo, Path(evidence) if evidence else None, videos
    )
    if not args.execute:
        for command in commands:
            for command_argv in command.commands():
                print(' '.join(command_argv))
        return 0
    results = run_official(commands, ctx)
    for task, codes in results.items():
        print(f'{task}: {codes}')
    return 0 if all(all(c == 0 for c in codes) for codes in results.values()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare, print, and optionally run the official judge for one row."""
    parser = build_parser(__doc__ or '')
    parser.add_argument('--row', required=True, help='ladder row name, e.g. L5')
    parser.add_argument('--items', default='full', help='full or subset')
    parser.add_argument('--repo', type=Path, default=None, help=f'${REPO_ENV}')
    parser.add_argument('--video-dir', type=Path, default=None, help=f'${VIDEO_ENV}')
    parser.add_argument(
        '--evidence-dir',
        default=None,
        help=f'the judge evidence directory, forwarded as an argument (${EVIDENCE_ENV})',
    )
    parser.add_argument(
        '--execute', action='store_true', help='actually run the official scripts'
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, f'judge:{args.row}')
        announce(prepare_official_inputs(args.row, args.items, ctx))
        if ctx.fake:
            announce(fake_judge(args.row, args.items, ctx))
            return 0
        return execute(args, ctx)

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
