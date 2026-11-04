"""Argument parsing shared by every entry point under ``examples/``.

Each example stays thin: it builds a parser from :func:`build_parser`, turns the
arguments into a :class:`RunContext`, calls library code, and prints what it
wrote. Errors from the package are reported once, here, as an exit code.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from emoharness.common.errors import EmoHarnessError
from emoharness.common.logging import get_logger
from emoharness.runner.context import DEFAULT_EXPERIMENT, RunContext, build_context
from emoharness.runner.paths import source_root

DEFAULT_RESAMPLES: Final = 10_000

_LOGGER = get_logger('emoharness.examples')


def build_parser(description: str) -> argparse.ArgumentParser:
    """A parser carrying the three flags every entry point accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--root',
        type=Path,
        default=source_root(),
        help='project root that holds data/ and results/ (default: the checkout)',
    )
    parser.add_argument(
        '--fake',
        action='store_true',
        help='run with fakes: no GPU, no data, no API keys',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite artifacts that already exist',
    )
    return parser


def add_target_options(parser: argparse.ArgumentParser) -> None:
    """Flags for the steps that call a Target model."""
    parser.add_argument(
        '--target', type=Path, default=None, help='path to a configs/target/*.yaml'
    )
    parser.add_argument(
        '--bundle', type=Path, default=None, help='harness bundle to use for this run'
    )
    parser.add_argument(
        '--replicate',
        type=int,
        default=0,
        help='construction replicate to read or write',
    )


def context_from(
    args: argparse.Namespace,
    experiment: str = DEFAULT_EXPERIMENT,
    **overrides: Any,
) -> RunContext:
    """Build the run context from parsed arguments."""
    settings: dict[str, Any] = {
        'fake': args.fake,
        'force': args.force,
        'experiment': experiment,
        'target_config': getattr(args, 'target', None),
        'bundle': getattr(args, 'bundle', None),
        'replicate': getattr(args, 'replicate', 0),
        'resamples': getattr(args, 'resamples', DEFAULT_RESAMPLES),
    }
    return build_context(args.root, **{**settings, **overrides})


def announce(written: Iterable[Path] | Mapping[str, Path]) -> None:
    """Print every artifact a step produced, one per line."""
    paths = written.values() if isinstance(written, Mapping) else written
    for path in paths:
        print(path)


def guard(action: Callable[[], int]) -> int:
    """Run one entry point, turning a package error into a non-zero exit code."""
    try:
        return action()
    except EmoHarnessError as exc:
        _LOGGER.error('%s: %s', type(exc).__name__, exc)
        return 1
