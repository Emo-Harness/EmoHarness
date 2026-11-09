"""Phase 3 prerequisite: fit the per-corpus eGeMAPS reference statistics.

    python examples/data/fit_prosody_reference.py            # over the external manifest
    python examples/data/fit_prosody_reference.py --items external:build

``describe_prosody`` turns a clip's eGeMAPS functionals into phrases such as
"high pitch" by z-scoring them against corpus means and standard deviations.
Those statistics have to exist before the shared cache is
filled, or every cached clip carries an empty prosody list. This step extracts
each construction clip's audio into the cache's audio slot (so ``preprocess``
reuses it), runs openSMILE, and writes ``configs/preprocess.yaml``'s
``prosody.reference_stats`` path (``data/derived/reference_stats.json``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from emoharness.common.hashing import sha256_file_cached
from emoharness.common.logging import get_logger
from emoharness.common.parallel import map_ordered, worker_count
from emoharness.data.manifest import Example
from emoharness.perceive.audio import OpenSmileProsody
from emoharness.perceive.descriptors import fit_reference, save_reference
from emoharness.perceive.video import extract_audio_track
from emoharness.runner.cli import build_parser, context_from, guard
from emoharness.runner.context import RunContext
from emoharness.runner.items import examples_for, load_external
from emoharness.runner.perception import resolve_input

EXTERNAL = 'external'
_MAX_WORKERS = 2
_LOG_EVERY = 100

_LOGGER = get_logger(__name__)


def select(item_set: str, ctx: RunContext) -> tuple[Example, ...]:
    """The construction clips whose statistics define 'typical' prosody."""
    if item_set == EXTERNAL:
        return load_external(ctx.paths)
    return examples_for(item_set, ctx.paths)


def fit(examples: Sequence[Example], ctx: RunContext) -> Path:
    """Extract, featurise, and write the reference statistics; return its path."""
    cache = ctx.clip_cache()
    prosody = OpenSmileProsody()
    total = len(examples)

    def features(example: Example) -> Mapping[str, float] | None:
        media = Path(example.media_path)
        wav = extract_audio_track(media, cache.audio_path(sha256_file_cached(media)))
        return None if wav is None else prosody.extract(wav)

    rows = [
        row
        for row in map_ordered(features, examples, min(worker_count(), _MAX_WORKERS))
        if row is not None
    ]
    _LOGGER.info('eGeMAPS functionals for %d/%d clips with audio', len(rows), total)
    stats = fit_reference(rows)
    target = ctx.paths.root / ctx.preprocess.prosody.reference_stats
    save_reference(target, stats)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the reference statistics the prosody descriptors z-score against."""
    parser = build_parser(__doc__ or '')
    parser.add_argument(
        '--items', default=EXTERNAL, help='external or external:<split>'
    )
    args = parser.parse_args(argv)

    def action() -> int:
        ctx = context_from(args, 'fit_prosody_reference')
        existing = resolve_input(ctx.paths, ctx.preprocess.prosody.reference_stats)
        if existing.is_file() and not ctx.force:
            print(f'{existing} already exists; pass --force to refit')
            return 0
        print(fit(select(args.items, ctx), ctx))
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
