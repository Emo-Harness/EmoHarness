"""Phase 3: fill the shared preprocessing cache, or verify the one that exists.

    python examples/data/preprocess.py --items all
    python examples/data/preprocess.py --items external:build --verify
    python examples/data/preprocess.py --fake --root /tmp/emoharness-fake

Every method downstream reads these entries, so the cache is what makes the
ladder's "byte-identical inputs" claim true.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.data.manifest import Example
from emoharness.runner.cli import build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.items import examples_for, load_benchmark, load_external
from emoharness.runner.perception import preprocess_all, verify_cache

ALL = 'all'
BENCHMARK = 'benchmark'
EXTERNAL = 'external'


def select(item_set: str, ctx: RunContext) -> tuple[Example, ...]:
    """The examples one ``--items`` value names."""
    if item_set == BENCHMARK:
        return load_benchmark(ctx.paths)
    if item_set == EXTERNAL:
        return load_external(ctx.paths)
    if item_set == ALL:
        return load_benchmark(ctx.paths) + load_external(ctx.paths)
    return examples_for(item_set, ctx.paths)


def main(argv: Sequence[str] | None = None) -> int:
    """Preprocess (or verify) every selected clip."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--items',
        default=ALL,
        help='all, benchmark, external, full, subset, or external:<split>',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='report entries that are missing or whose frames are gone',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'preprocess')
        examples = select(args.items, ctx)
        cache = ctx.clip_cache()
        if args.verify:
            broken = verify_cache(examples, cache, ctx.preprocess)
            for example_id in broken:
                print(f'missing: {example_id}')
            print(
                f'{len(examples) - len(broken)}/{len(examples)} cache entries verified'
            )
            return 1 if broken else 0
        clips = preprocess_all(examples, cache, ctx.preprocess, ctx.clip_preprocessor())
        print(f'{len(clips)} clips cached under {ctx.paths.preprocess_cache}')
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
