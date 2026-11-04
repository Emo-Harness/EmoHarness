"""One ladder row of the transfer run on IntentQA (test) or Social-IQ 2.0 (val).

    # fill the shared perception cache once, no Target call
    python -m experiments.transfer.row --dataset intentqa_test --perceive-only

    # one student, one row
    EMOHARNESS_RESULTS_DIR=results/transfer/qwen3vl_4b/L5_own/intentqa_test \\
    python -m experiments.transfer.row --dataset intentqa_test --student qwen3vl_4b --row L5

The same ``run_row`` path, ladder rows, perception config and bundles as the
main evaluation, so the only thing that differs is the item set. It refuses to
run if the manifest is not the one ``experiments.transfer.manifests``
recorded, if the bundle does not hash to the student's frozen one, or if the
perception config hash differs from the main evaluation's.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Final

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file
from emoharness.data.manifest import read_manifest
from emoharness.runner.cli import build_parser, context_from, guard
from emoharness.runner.rows import inputs_for, run_row
from experiments.common import (
    PERCEPTION_CONFIG_HASH,
    ROOT,
    SCAFFOLD,
    SCAFFOLD_HASH,
    STUDENTS,
)

RUN: Final = ROOT / 'results/transfer'
DATASETS: Final = ('intentqa_test', 'social_iq_val')
#: L0 native video, L1 video + the audio evidence text, L3 the shared scaffold,
#: L5 the student's own bundle.
ROWS: Final = ('L0', 'L1', 'L3', 'L5')


def load_items(dataset: str):
    record = json.loads((RUN / 'manifests/manifest_record.json').read_text())
    declared = record.get(dataset)
    if not declared:
        raise ValidationError(
            f'{dataset} is not in manifest_record.json; build it first'
        )
    path = ROOT / declared['file']
    actual = sha256_file(path)
    if actual != declared['sha256']:
        raise ValidationError(
            f'{path.name} sha256 {actual[:16]} != recorded '
            f'{declared["sha256"][:16]}; the manifest changed'
        )
    examples = read_manifest(path)
    ids = [e.example_id for e in examples]
    if len(ids) != len(set(ids)) or len(ids) != declared['n_items']:
        raise ValidationError(
            f'{path.name}: {len(ids)} rows, {len(set(ids))} distinct, '
            f'recorded {declared["n_items"]}'
        )
    return examples, {
        'manifest': declared['file'],
        'manifest_sha256': actual,
        'n_items': len(ids),
    }


def main(argv=None) -> int:
    parser = build_parser(__doc__ or '')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--student', choices=sorted(STUDENTS))
    parser.add_argument('--row', choices=ROWS, default='L5')
    parser.add_argument(
        '--perceive-only',
        action='store_true',
        help='fill the shared perception cache; no Target call',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='interface check: only the items of the first two videos, '
        'written under a separate smoke_<dataset> item set',
    )
    parser.add_argument(
        '--shard',
        default='0/1',
        help='i/n: perceive only videos whose index mod n == i, so '
        'several processes never touch the same clip',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        examples, provenance = load_items(args.dataset)
        if args.perceive_only:
            ctx = context_from(args, f'transfer:perceive:{args.dataset}')
            if ctx.preprocess.config_hash() != PERCEPTION_CONFIG_HASH:
                raise ValidationError(
                    'perception config differs from the main evaluation'
                )
            i, n = (int(x) for x in args.shard.split('/'))
            videos = sorted({e.source_id for e in examples})
            mine = {v for k, v in enumerate(videos) if k % n == i}
            examples = tuple(e for e in examples if e.source_id in mine)
            started = time.time()
            done = inputs_for(examples, ctx)
            print(
                json.dumps(
                    {
                        'dataset': args.dataset,
                        'shard': args.shard,
                        'videos': len(mine),
                        'perceived': len(done),
                        'seconds': round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
            return 0
        if not args.student:
            raise ValidationError('--student is required unless --perceive-only')
        student = STUDENTS[args.student]
        args.target = ROOT / student.target
        from emoharness.harness.loader import load_bundle

        if args.row in ('L0', 'L1'):
            args.bundle, want, bundle_rel = None, None, None
        elif args.row == 'L3':
            args.bundle, want, bundle_rel = ROOT / SCAFFOLD, SCAFFOLD_HASH, SCAFFOLD
        else:
            args.bundle, want, bundle_rel = (
                ROOT / student.bundle,
                student.bundle_hash,
                student.bundle,
            )
        got = load_bundle(args.bundle).content_hash() if args.bundle else None
        if got != want:
            raise ValidationError(
                f'{args.student} {args.row} bundle hashes to '
                f'{str(got)[:16]}, not the frozen {str(want)[:16]}'
            )
        ctx = context_from(args, f'transfer:{args.student}:{args.row}:{args.dataset}')
        if ctx.preprocess.config_hash() != PERCEPTION_CONFIG_HASH:
            raise ValidationError('perception config differs from the main evaluation')
        row_name = args.row
        row = ctx.ladder.rows[row_name]
        item_set = args.dataset
        if args.smoke:
            first = sorted({e.source_id for e in examples})[:2]
            examples = tuple(e for e in examples if e.source_id in first)
            item_set = f'smoke_{args.dataset}'
            provenance['smoke'] = True
        started = time.time()
        predictions = run_row(row, examples, ctx, item_set=item_set)
        written = ctx.paths.predictions(row_name, item_set)
        want = {e.example_id for e in examples}
        got_ids = {p.example_id for p in predictions}
        if got_ids != want or len(predictions) != len(want):
            raise ValidationError(
                f'{args.student}/{args.dataset}: wrote '
                f'{len(predictions)} predictions, {len(got_ids & want)} '
                f'of {len(want)} expected ids'
            )
        target = ctx.target_model_config()
        provenance.update(
            {
                'student': args.student,
                'dataset': args.dataset,
                'row': row_name,
                'bundle': bundle_rel,
                'bundle_hash': got,
                'target_config': student.target,
                'model_id': target.model_id,
                'decoding': target.decoding.model_dump(mode='json'),
                'predictions': str(written.relative_to(ROOT)),
                'n_written': len(predictions),
                'seconds': round(time.time() - started, 1),
                'asr_config_hash': ctx.preprocess.config_hash(),
                'endpoint': os.environ.get('VLLM_ENDPOINT', ''),
                'python': platform.python_version(),
                'host': platform.node(),
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
        )
        out = written.parent.parent.parent / 'row_provenance.json'
        tmp = out.with_name(f'.{out.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(provenance, indent=2, default=str) + '\n')
        tmp.replace(out)
        print(json.dumps(provenance, default=str), flush=True)
        return 0

    return guard(action)


if __name__ == '__main__':
    sys.exit(main())
