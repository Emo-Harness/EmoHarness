"""Phase 2: build the construction manifest, assign its splits, report its labels.

    python examples/prepare_external.py
    python examples/prepare_external.py --fake --root /tmp/emoharness-fake

Corpora are read from the roots in ``configs/data/sources.yaml``; one whose root
is absent is skipped with a warning, so CREMA-D can stand in for IEMOCAP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from emoharness.common.errors import ValidationError
from emoharness.data.fakes import DEFAULT_PER_SOURCE, make_fake_external
from emoharness.runner.cli import announce, build_parser, context_from, guard
from emoharness.runner.ingest import prepare_external
from emoharness.runner.items import EXTERNAL_SPLITS

#: Split targets for --fake: the 150/150/100 ratio of ADR-0003, scaled to a
#: corpus of a few dozen clips. A target that can never be reached would starve
#: the splits below it.
FAKE_PER_KIND: Mapping[str, int] = {'build': 6, 'val': 6, 'ext_test': 4}


def parse_per_kind(raw: str | None, fake: bool) -> Mapping[str, int] | None:
    """Parse ``--per-kind build,val,ext_test``; ``--fake`` has its own default."""
    if raw is None:
        return FAKE_PER_KIND if fake else None
    parts = tuple(part.strip() for part in raw.split(','))
    if len(parts) != len(EXTERNAL_SPLITS):
        raise ValidationError('--per-kind takes three counts: build,val,ext_test')
    try:
        counts = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValidationError(f'--per-kind takes integers, got {raw!r}') from exc
    return dict(zip(EXTERNAL_SPLITS, counts, strict=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Build ``data/manifests/external.jsonl`` and ``data/splits/external_splits.json``."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--per-source',
        type=int,
        default=DEFAULT_PER_SOURCE,
        help='groups per corpus to synthesize with --fake',
    )
    parser.add_argument(
        '--per-kind',
        default=None,
        help='split targets per seen kind, as build,val,ext_test',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'prepare_external')
        roots = (
            make_fake_external(ctx.paths.fake_raw, args.per_source)
            if args.fake
            else None
        )
        per_kind = parse_per_kind(args.per_kind, args.fake)
        announce(prepare_external(ctx, roots, per_kind))
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
