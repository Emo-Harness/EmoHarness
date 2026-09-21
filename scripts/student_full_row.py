"""One evaluation row of the 2026-09-18 batch, over the frozen eligible item set.

    python scripts/student_full_row.py --rows L0 --items full5507
    python scripts/student_full_row.py --rows L5 --items smoke24 --bundle <final>

The item set is NOT ``full``. ``full`` means all 5,577 rows of the canonical
manifest (``src/emoharness/runner/items.py``), and 70 of those share an exact
media file with a construction item -- 27 with D_build, 40 with D_val, 3 with
the reserved ext_test -- under a different ``example_id``. Ids are disjoint;
the clips are not. Evaluating them would score the main comparison on video the
Builder had already seen, so this batch evaluates the frozen eligible set the
coordinator published instead:

    results/v3/student_specific_full_20260918/evaluation_items.jsonl   5,507 rows
    results/v3/student_specific_full_20260918/evaluation_excluded_ids.json  70 ids

The canonical manifest is never edited: the exclusion lives in a run-local file
with its own sha256, and this step refuses to run if that file is not byte-for-byte
what the execution manifest declared. The strata the report needs are 986 rows
from the pre-registered subset of 1,000 (14 of those 1,000 are media-excluded)
and 4,521 new rows -- never 1,000 and 4,577.

``smoke24`` is three items from each of the eight tasks, in frozen-set order: a
label-blind interface check, never a sample to tune against.

Predictions land under ``<EMOHARNESS_RESULTS_DIR>/raw_predictions/<row>/<items>.jsonl``
and, for harness rows, the ``HarnessOutput`` records beside them. The step also
writes ``row_provenance.json`` next to them, so the driver's marker records what
the job actually read rather than what the driver assumed it would.

A killed row loses its predictions file but not its work: every completed Target
call is in the shared content-addressed cache, so a retry re-issues only the
calls that never finished.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file
from emoharness.data.manifest import Example, read_manifest
from emoharness.runner.cli import add_target_options, build_parser, context_from, guard
from emoharness.runner.rows import run_row

BATCH_ROOT: Final = Path('results/v3/student_specific_full_20260918')
ITEM_SET: Final = 'full5507'
SMOKE_LABEL: Final = 'smoke24'
SMOKE_PER_TASK: Final = 3
TASK_COUNT: Final = 8
ITEM_SETS: Final = (ITEM_SET, SMOKE_LABEL)
EXPECTED_ITEMS: Final = 5507
EXPECTED_EXCLUDED: Final = 70


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def frozen_paths(root: Path | None = None) -> tuple[Path, Path, Path]:
    """The frozen item set, its exclusion list, and the execution manifest.

    These are batch artifacts, not per-root data: a ``--root`` pointed at a
    scratch tree still reads the one published, declared item set.
    """
    base = (root or _repo_root()) / BATCH_ROOT
    return (base / 'evaluation_items.jsonl',
            base / 'evaluation_excluded_ids.json',
            base / 'execution_manifest.json')


def load_frozen_items(
    items_file: Path,
    excluded_file: Path,
    manifest_file: Path,
    expect_sha256: str | None = None,
    expect_count: int = EXPECTED_ITEMS,
    require_manifest: bool = True,
) -> tuple[tuple[Example, ...], dict[str, Any]]:
    """Read the eligible set and prove it is the declared one before using it.

    Every check here is a launch blocker rather than a warning: scoring the
    wrong item set produces a number that looks fine and means something else.

    ``require_manifest`` is the default because an absent execution manifest,
    or a manifest without the two hash fields, means nothing is being compared
    against anything -- the checks would pass by having no reference. Only a
    test may switch it off, through ``--allow-unverified-item-set``, and it says
    so in the provenance it returns.
    """
    for path in (items_file, excluded_file):
        if not path.is_file():
            raise ValidationError(f'frozen evaluation artifact missing: {path}')
    items_sha = sha256_file(items_file)
    excluded_sha = sha256_file(excluded_file)
    declared: dict[str, Any] = {}
    if manifest_file.is_file():
        declared = json.loads(manifest_file.read_text(encoding='utf-8'))
    elif require_manifest:
        raise ValidationError(
            f'no execution manifest at {manifest_file}; a real run must verify the '
            f'frozen item set against declared hashes, not merely read it'
        )
    if require_manifest and not expect_sha256:
        missing = [name for name in ('evaluation_sha256', 'exclusions_sha256')
                   if not declared.get(name)]
        if missing:
            raise ValidationError(
                f'{manifest_file.name} declares no {", ".join(missing)}; a real run '
                f'cannot verify the frozen item set without them'
            )
    expected_sha = expect_sha256 or declared.get('evaluation_sha256')
    if expected_sha and items_sha != expected_sha:
        raise ValidationError(
            f'{items_file.name} sha256 {items_sha} does not match the declared '
            f'{expected_sha}; the frozen evaluation set has changed'
        )
    expected_excluded_sha = declared.get('exclusions_sha256')
    if expected_excluded_sha and excluded_sha != expected_excluded_sha:
        raise ValidationError(
            f'{excluded_file.name} sha256 {excluded_sha} does not match the '
            f'declared {expected_excluded_sha}'
        )
    examples = read_manifest(items_file)
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValidationError(
            f'{items_file.name} carries {len(ids) - len(set(ids))} duplicate id(s)'
        )
    if len(ids) != expect_count:
        raise ValidationError(
            f'{items_file.name} has {len(ids)} items, expected {expect_count}'
        )
    excluded = set(json.loads(excluded_file.read_text(encoding='utf-8'))['ids'])
    if len(excluded) != EXPECTED_EXCLUDED:
        raise ValidationError(
            f'{excluded_file.name} lists {len(excluded)} ids, expected '
            f'{EXPECTED_EXCLUDED}'
        )
    intersection = excluded & set(ids)
    if intersection:
        raise ValidationError(
            f'{len(intersection)} media-excluded id(s) are still in the eligible '
            f'set, e.g. {sorted(intersection)[:3]}'
        )
    tasks = {example.task.value for example in examples if example.task is not None}
    if len(tasks) != TASK_COUNT:
        raise ValidationError(
            f'the eligible set covers {len(tasks)} task(s), expected {TASK_COUNT}'
        )
    provenance = {
        'item_set_file': str(items_file),
        'item_set_sha256': items_sha,
        'excluded_ids_file': str(excluded_file),
        'excluded_ids_sha256': excluded_sha,
        'n_items_expected': len(ids),
        'n_excluded': len(excluded),
        'n_raw_canonical': declared.get('n_full_raw') or declared.get('n_raw_5577') or 5577,
        'n_old_subset_stratum': declared.get('n_subset'),
        'n_new_stratum': declared.get('n_new'),
        'tasks': sorted(tasks),
        'manifest_verified': bool(require_manifest),
    }
    return examples, provenance


def smoke_examples(examples: Sequence[Example]) -> tuple[Example, ...]:
    """Three items from each of the eight tasks, in frozen-set order."""
    chosen: dict[str, list[Example]] = {}
    for example in examples:
        if example.task is None:
            continue
        bucket = chosen.setdefault(example.task.value, [])
        if len(bucket) < SMOKE_PER_TASK:
            bucket.append(example)
    picked = tuple(x for task in sorted(chosen) for x in chosen[task])
    expected = SMOKE_PER_TASK * TASK_COUNT
    if len(chosen) != TASK_COUNT or len(picked) != expected:
        raise ValidationError(
            f'smoke set covers {len(chosen)} tasks / {len(picked)} items; '
            f'expected {TASK_COUNT} tasks / {expected} items'
        )
    return picked


def _git_state(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            out = subprocess.run(['git', *args], cwd=root, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return ''
        return out.stdout.strip() if out.returncode == 0 else ''

    return {'git_rev': git('rev-parse', 'HEAD'),
            'git_dirty': bool(git('status', '--porcelain'))}


def _serving_version(endpoint: str) -> str:
    import urllib.request
    try:
        base = endpoint.rstrip('/').removesuffix('/v1')
        with urllib.request.urlopen(f'{base}/version', timeout=5) as response:
            return str(json.load(response).get('version', ''))
    except Exception:
        return ''


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(__doc__ or '')
    add_target_options(parser)
    parser.add_argument('--rows', required=True, help='one row name, as in ladder.yaml')
    parser.add_argument('--items', default=ITEM_SET,
                        help=f'one of {", ".join(ITEM_SETS)}')
    parser.add_argument('--item-set-file', type=Path, default=None,
                        help='override the frozen eligible-set path (testing only)')
    parser.add_argument('--excluded-ids-file', type=Path, default=None)
    # Overriding the item set without being able to override the manifest that
    # declares its hash makes the override unusable: the synthetic set gets
    # checked against the real set's sha256 and can only fail.
    parser.add_argument('--execution-manifest', type=Path, default=None,
                        help='override the execution manifest the frozen item set '
                             'is verified against (testing only)')
    parser.add_argument('--expect-sha256', default=None,
                        help='require this sha256 for the frozen item set')
    parser.add_argument('--expect-count', type=int, default=EXPECTED_ITEMS)
    parser.add_argument('--allow-unverified-item-set', action='store_true',
                        help='testing only: run without an execution manifest to '
                             'verify the frozen item set against')
    args = parser.parse_args(argv)

    def action() -> int:
        name = args.rows.strip()
        if args.items not in ITEM_SETS:
            raise ValidationError(
                f'this batch evaluates {" or ".join(ITEM_SETS)} only, '
                f'got {args.items!r}; "full" would score the 70 media-overlapping rows'
            )
        ctx = context_from(args, f'student_full:{name}')
        row = ctx.ladder.rows.get(name)
        if row is None:
            raise ValidationError(f'unknown row {name!r}')
        default_items, default_excluded, manifest_file = frozen_paths()
        examples, provenance = load_frozen_items(
            args.item_set_file or default_items,
            args.excluded_ids_file or default_excluded,
            args.execution_manifest or manifest_file,
            args.expect_sha256,
            args.expect_count,
            require_manifest=not args.allow_unverified_item_set,
        )
        if args.items == SMOKE_LABEL:
            examples = smoke_examples(examples)
            provenance['n_items_expected'] = len(examples)
        expected_ids = frozenset(example.example_id for example in examples)
        target_config = ctx.target_model_config()
        endpoint = os.environ.get('VLLM_ENDPOINT', '')
        started = time.time()
        predictions = run_row(row, examples, ctx, item_set=args.items)
        written = ctx.paths.predictions(name, args.items)
        provenance.update({
            'row': name, 'items': args.items,
            'n_items_written': len(predictions),
            'seconds': round(time.time() - started, 1),
            'bundle': str(args.bundle) if args.bundle else None,
            'target_config': str(args.target) if args.target else None,
            'predictions': str(written),
            'asr_config_hash': ctx.preprocess.config_hash(),
            'provenance': {
                **_git_state(_repo_root()),
                'venv': str(Path(os.__file__).resolve().parents[2]),
                'python': platform.python_version(),
                'vllm_version': _serving_version(endpoint) if endpoint else '',
                'endpoint': endpoint,
                'target_config': str(args.target) if args.target else None,
                'model_id': target_config.model_id,
                'decoding': target_config.decoding.model_dump(mode='json'),
                'video': target_config.video.model_dump(mode='json'),
                'chat_template_kwargs': dict(target_config.chat_template_kwargs or {}),
                'call_budget': target_config.call_budget.model_dump(mode='json'),
                'call_cache_root': str(ctx.call_cache_root()),
                'bundle_hash': _bundle_hash(args.bundle),
                'scaffold_hash': _bundle_hash(_repo_root() / 'harness'),
            },
        })
        provenance_path = written.parent.parent.parent / 'row_provenance.json'
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        temp = provenance_path.with_name(f'.{provenance_path.name}.{os.getpid()}.tmp')
        temp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2,
                                   sort_keys=True, default=str) + '\n', encoding='utf-8')
        temp.replace(provenance_path)
        # The exact id set, not its size. Two rows of the same length can be
        # two different item sets, and a length check would pass on both.
        written_ids = frozenset(record.example_id for record in predictions)
        if written_ids != expected_ids:
            stray = sorted(written_ids - expected_ids)[:3]
            absent = sorted(expected_ids - written_ids)[:3]
            raise ValidationError(
                f'{name} scored a different id set than the frozen one: '
                f'{len(written_ids)} written vs {len(expected_ids)} expected, '
                f'{len(written_ids - expected_ids)} stray (e.g. {stray}), '
                f'{len(expected_ids - written_ids)} unscored (e.g. {absent})'
            )
        if len(predictions) != len(written_ids):
            raise ValidationError(
                f'{name} wrote {len(predictions)} predictions for '
                f'{len(written_ids)} distinct ids'
            )
        print(json.dumps(provenance, ensure_ascii=False, default=str))
        return 0

    return guard(action)


def _bundle_hash(path: Path | str | None) -> str | None:
    if path is None:
        return None
    from emoharness.harness.loader import load_bundle
    candidate = Path(path)
    if not candidate.is_dir():
        return None
    return load_bundle(candidate).content_hash()


if __name__ == '__main__':
    raise SystemExit(main())
