"""A per-item data package of every run behind the paper, for sharing or archiving.

    python -m experiments.paper.export                  # the paper's two Builders
    python -m experiments.paper.export --all-builders   # + GPT-6 Luna

Writes ``results/export/emoharness_<date>/`` and a ``.tar.gz`` beside it. No
benchmark media is included. Runs that have not happened are listed under
``missing`` in ``summary.json``.

- ``summary.json``: everything ``experiments.paper.report`` computes, and the
  missing runs.
- ``main_eval/<student>.jsonl``: one line per evaluation item: task, gold labels,
  and for Video, Audio, Shared and each Builder's MHE the predicted labels, local
  Recognition, call count and full response text.
- ``construction/<builder>/<student>/``: ``loop_result.json`` (every generation,
  candidate, gate result and promotion), ``bundle_diff.patch`` (the evolved
  harness against the Shared scaffold), ``manifest.json`` and ``cost.json``
  (API spend by role).
- ``emotion_context/<dataset>/<student>.jsonl``: one line per question: video,
  gold answer, and for Frames, Audio and each Builder's Audio + Emotion the chosen
  letter, correctness and full reply.
- ``emotion_context/contexts/<builder>/<student>/<dataset>.jsonl``: the context
  each harness wrote per video.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from emoharness.evaluation.scoreboard import Scoreboard
from experiments.common import ORDER
from experiments.emotion_context.row import DATASETS
from experiments.layout import (
    ALL_BUILDERS,
    BUILDERS,
    CONDITION_ROWS,
    PAPER_BUILDERS,
    Layout,
)
from experiments.paper import intentqa, mme, report
from experiments.paper.cases import bundle_diff


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def export_main(
    layout: Layout,
    pkg: Path,
    builders: Sequence[str],
    examples: Mapping[str, Any],
    missing: list[str],
) -> None:
    for st in ORDER:
        paths = {
            name: layout.row_predictions(st, CONDITION_ROWS[name])
            for name in mme.BASE_CONDITIONS
        }
        paths.update({mme.mhe(b): layout.mhe_predictions(b, st) for b in builders})
        rows: dict[str, dict[str, Any]] = {}
        for name, path in paths.items():
            preds = mme.read_predictions(path)
            if preds is None:
                missing.append(f'main_eval/{st}/{name}')
                continue
            board = Scoreboard.from_predictions(preds, examples)
            rows[name] = {
                p.example_id: {
                    'labels': list(p.labels),
                    'recognition': board.items[p.example_id].recognition,
                    'target_calls': p.target_calls,
                    'response_text': p.response_text,
                }
                for p in preds
            }
        write_jsonl(
            pkg / 'main_eval' / f'{st}.jsonl',
            (
                {
                    'example_id': i,
                    'task': e.task.value if e.task is not None else None,
                    'gold': list(e.labels),
                    **{name: rows.get(name, {}).get(i) for name in paths},
                }
                for i, e in examples.items()
            ),
        )


def export_construction(
    layout: Layout, pkg: Path, builders: Sequence[str], missing: list[str]
) -> None:
    scaffold = layout.checkout / 'harness'
    for b in builders:
        for st in ORDER:
            root = layout.construction_dir(b, st)
            if not (root / 'loop_result.json').is_file():
                missing.append(f'construction/{b}/{st}')
                continue
            d = pkg / 'construction' / b / st
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / 'loop_result.json', d / 'loop_result.json')
            final = root / 'final'
            (d / 'bundle_diff.patch').write_text(bundle_diff(scaffold, final))
            shutil.copy(final / 'manifest.json', d / 'manifest.json')
            cost: dict[str, dict[str, Any]] = {}
            for f in sorted(layout.build_dir(b, st).rglob('ledger*.jsonl')):
                for x in f.read_text().splitlines():
                    if not x.strip():
                        continue
                    r = json.loads(x)
                    if r.get('role') == 'target' or not str(
                        r.get('experiment', '')
                    ).startswith('build_harness'):
                        continue
                    c = cost.setdefault(
                        f'{r["role"]}:{r.get("model_id")}',
                        {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'usd': 0.0},
                    )
                    c['calls'] += 1
                    c['input_tokens'] += r.get('input_tokens') or 0
                    c['output_tokens'] += r.get('output_tokens') or 0
                    c['usd'] += r.get('cost_usd') or 0.0
            (d / 'cost.json').write_text(json.dumps(cost, indent=1))


def export_emotion_context(
    layout: Layout, pkg: Path, builders: Sequence[str], missing: list[str]
) -> None:
    for ds in DATASETS:
        items = intentqa.questions(layout, ds)
        if items is None:
            missing.append(f'emotion_context/{ds}')
            continue
        for st in ORDER:
            rows = intentqa.load_student(layout, st, ds, builders, items)
            names = {
                'none': 'none',
                'asr': 'asr',
                **{b: f'emoctx_{b}' for b in builders},
            }
            for key, name in names.items():
                if rows[key] is None:
                    missing.append(f'emotion_context/{ds}/{st}/{name}')
            write_jsonl(
                pkg / 'emotion_context' / ds / f'{st}.jsonl',
                (
                    {
                        'example_id': i,
                        'video_id': q['video_id'],
                        'question': q.get('question'),
                        'options': q['options'],
                        'answer_index': q['answer_index'],
                        **{
                            name: (
                                {
                                    'choice': r.get('choice'),
                                    'correct': r['correct'],
                                    'response_text': r.get('response_text'),
                                }
                                if (r := (rows[key] or {}).get(i))
                                else None
                            )
                            for key, name in names.items()
                        },
                    }
                    for i, q in items.items()
                    if any(i in (rows[k] or {}) for k in names)
                ),
            )
            for b in builders:
                done = (
                    layout.emotion_context_step(st, 'context', None, ds, b)
                    / 'done.json'
                )
                if not done.is_file():
                    continue
                src = layout.root / json.loads(done.read_text())['contexts']
                dst = pkg / 'emotion_context' / 'contexts' / b / st / f'{ds}.jsonl'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument(
        '--root', type=Path, default=None, help='run root (default: the checkout)'
    )
    parser.add_argument('--all-builders', action='store_true', help='add GPT-6 Luna')
    args = parser.parse_args(argv)
    layout = Layout(root=args.root.resolve()) if args.root else Layout()
    builders = ALL_BUILDERS if args.all_builders else PAPER_BUILDERS
    stamp = time.strftime('%Y-%m-%d', time.gmtime())
    pkg = layout.root / 'results/export' / f'emoharness_{stamp}'
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    examples = mme.evaluation_examples(layout)
    missing: list[str] = []
    export_main(layout, pkg, builders, examples, missing)
    export_construction(layout, pkg, builders, missing)
    export_emotion_context(layout, pkg, builders, missing)
    summary = {
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'builders': {b: BUILDERS[b].label for b in builders},
        'students': list(ORDER),
        'results': report.build(layout, builders),
        'missing': sorted(set(missing)),
    }
    (pkg / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    (pkg / 'README.md').write_text(
        (__doc__ or '').split('\n\n', 2)[2].lstrip()
        + f'\nGenerated {summary["generated_utc"]}. Missing runs: {len(summary["missing"])} '
        '(see summary.json).\n'
    )
    tar = pkg.with_suffix('.tar.gz')
    with tarfile.open(tar, 'w:gz') as t:
        t.add(pkg, arcname=pkg.name)
    print(
        json.dumps(
            {'package': str(pkg), 'tar': str(tar), 'missing': summary['missing']},
            indent=1,
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
