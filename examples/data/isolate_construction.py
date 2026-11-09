"""Phase 2: isolate the construction split by source and media identity.

    python examples/data/isolate_construction.py

Reads ``data/manifests/external.jsonl`` (from ``split_construction.py``) and
writes ``data/manifests/external_v3.jsonl``, the construction manifest every MHE
run reads (``EMOHARNESS_EXTERNAL_MANIFEST=external_v3.jsonl``), plus
``data/splits/external_v3_isolation.json``. Items that share a media hash, a
media path or a recorded source with an item of another split are connected;
in each connected group only the copies in the most protected split are kept
(ext_test over val over build) and the rest are quarantined, never moved to
another split. No score is read.

The construction loop refuses a manifest whose splits overlap
(``emoharness.data.isolation.assert_isolated``). For the paper this quarantined
one item, ``mme_er_wild_test_dia190_utt4``, and left 399 build, 394 val and 129
ext_test items (the reserved set).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from emoharness.common.hashing import sha256_file
from emoharness.common.io import write_json_atomic
from emoharness.data.isolation import quarantine_overlaps
from emoharness.data.manifest import read_manifest, write_manifest
from emoharness.runner.cli import build_parser, guard
from emoharness.runner.paths import EXTERNAL_MANIFEST_FILE, ProjectPaths, guard_write

ISOLATED_MANIFEST_FILE = 'external_v3.jsonl'
ISOLATION_RECORD_FILE = 'external_v3_isolation.json'


def main(argv: Sequence[str] | None = None) -> int:
    """Write the isolated manifest and the record of what was quarantined."""
    args = build_parser(__doc__ or '').parse_args(argv)

    def action() -> int:
        paths = ProjectPaths.from_root(args.root)
        source = paths.manifests / EXTERNAL_MANIFEST_FILE
        retained, removed = quarantine_overlaps(read_manifest(source))
        target = guard_write(paths.manifests / ISOLATED_MANIFEST_FILE, args.force)
        record = guard_write(paths.splits / ISOLATION_RECORD_FILE, args.force)
        write_manifest(target, retained)
        counts = dict(Counter(item.split.value for item in retained))
        write_json_atomic(
            record,
            {
                'source': str(source),
                'source_hash': sha256_file(source),
                'manifest_hash': sha256_file(target),
                'counts': counts,
                'quarantined_ids': [item.example_id for item in removed],
                'rule': 'connected media_hash/media_path/(origin, source_id); '
                'ext_test > val > build',
                'limitation': 'recorded identities only; no new independent final '
                'test set is created',
            },
        )
        print(f'{target}: {counts}; quarantined {len(removed)} overlapping items')
        return 0

    return guard(action)


if __name__ == '__main__':
    raise SystemExit(main())
