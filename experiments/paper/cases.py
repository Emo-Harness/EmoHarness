"""The per-item records behind the case studies (Figures 1(a), 2(b), 4 and 11).

    # the paper's four cases, resolved from the runs (report writes them too)
    python -m experiments.paper.cases --paper
    # items where a Builder's MHE harness changes Shared's answer, with the direction
    python -m experiments.paper.cases --student gemma4_e4b --builder gpt5_6_luna --list
    # one item under every condition: gold, labels, correctness, full response
    python -m experiments.paper.cases --student gemma4_e4b --item <example_id>

The paper's cases are identified by student, task, gold label and the words the
figure quotes from the transcript (``CASES``); the harness's own evidence record
holds that transcript, so the item is found in the Shared row's
``clean.outputs.jsonl``. For each case the record carries every condition's
answer and response, the Shared and MHE harnesses' evidence, hypotheses and
step outputs, and the edits each Builder's harness makes to the Shared scaffold.
Cases in the paper were selected, not sampled; ``--list`` is a starting point
for choosing one, never a summary statistic.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from emoharness.data.label_set import normalize_label
from emoharness.data.manifest import Example
from experiments.common import ORDER
from experiments.layout import ALL_BUILDERS, BUILDERS, PAPER_BUILDERS, Layout
from experiments.paper import mme

#: The paper's case studies. ``transcript`` is the quoted speech, as the figure prints it.
CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        'key': 'figure4a',
        'figures': ('1a', '4a'),
        'student': 'gemma4_e4b',
        'task': 'FG_SA',
        'gold': 'negative',
        'transcript': "It's not my form so",
    },
    {
        'key': 'figure4b',
        'figures': ('4b',),
        'student': 'gemma4_e4b',
        'task': 'ER_Lab',
        'gold': 'neutral',
        'transcript': 'I know I paid last month',
    },
    {
        'key': 'figure11a',
        'figures': ('2b', '11a'),
        'student': 'qwen3vl_4b',
        'task': 'FG_SA',
        'gold': 'negative',
        'transcript': 'And I hate to say that',
    },
    {
        'key': 'figure11b',
        'figures': ('11b',),
        'student': 'qwen3vl_8b',
        'task': 'ER_Lab',
        'gold': 'excited',
        'transcript': 'what will we do',
    },
)
#: The parts of a harness record a case figure draws on.
HARNESS_FIELDS: Final = (
    'labels',
    'evidence',
    'hypotheses',
    'verification',
    'steps',
    'fallback_used',
)


def _norm(text: str) -> str:
    text = text.replace('’', "'").replace('‘', "'").lower()
    return ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', text).split())


def outputs_path(predictions: Path) -> Path:
    """The harness records beside a harness row's predictions."""
    return predictions.with_name(predictions.name.replace('.jsonl', '.outputs.jsonl'))


def harness_records(
    path: Path, ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if ids is None or record['example_id'] in ids:
            out[record['example_id']] = record
    return out


def resolve(
    layout: Layout, case: Mapping[str, Any], examples: Mapping[str, Example]
) -> list[str]:
    """The evaluation items of the case's task and gold label whose transcript quotes it."""
    wanted = _norm(case['transcript'])
    candidates = {
        i
        for i, e in examples.items()
        if e.task is not None
        and e.task.value == case['task']
        and normalize_label(case['gold']) in {normalize_label(g) for g in e.labels}
    }
    records = harness_records(
        outputs_path(layout.row_predictions(case['student'], 'L3')), candidates
    )
    return sorted(
        i
        for i, r in records.items()
        if wanted
        in _norm(((r.get('evidence') or {}).get('audio') or {}).get('transcript') or '')
    )


def item(
    layout: Layout,
    student: str,
    example_id: str,
    builders: Sequence[str],
    examples: Mapping[str, Example] | None = None,
) -> dict[str, Any]:
    """One item under Video, Audio, Shared and each Builder's MHE, with the harness records."""
    examples = examples if examples is not None else mme.evaluation_examples(layout)
    example = examples[example_id]
    rows = mme.load_student(layout, student, builders, examples)
    harness_paths = {
        'shared': outputs_path(layout.row_predictions(student, 'L3')),
        **{
            mme.mhe(b): outputs_path(layout.mhe_predictions(b, student))
            for b in builders
        },
    }
    records = {
        name: harness_records(path, {example_id}).get(example_id)
        for name, path in harness_paths.items()
    }
    return {
        'example_id': example_id,
        'task': example.task.value if example.task else None,
        'gold': list(example.labels),
        'prompt': example.prompt,
        'media_path': example.media_path,
        'conditions': {
            name: (
                {
                    'labels': list(c.predictions[example_id].labels),
                    'recognition': c.board.items[example_id].recognition,
                    'target_calls': c.predictions[example_id].target_calls,
                    'response_text': c.predictions[example_id].response_text,
                    'harness': (
                        {k: records[name].get(k) for k in HARNESS_FIELDS}
                        if records.get(name)
                        else None
                    ),
                }
                if c is not None and example_id in c.predictions
                else None
            )
            for name, c in rows.items()
        },
    }


def bundle_diff(scaffold: Path, final: Path) -> str:
    """The evolved harness's edits to the Shared scaffold, as a unified diff."""
    out: list[str] = []
    files = sorted(
        {p.relative_to(scaffold) for p in scaffold.rglob('*') if p.is_file()}
        | {p.relative_to(final) for p in final.rglob('*') if p.is_file()}
    )
    for rel in files:
        if rel.name == 'manifest.json' or '__pycache__' in rel.parts:
            continue
        a, b = scaffold / rel, final / rel
        old = a.read_text().splitlines(keepends=True) if a.is_file() else []
        new = b.read_text().splitlines(keepends=True) if b.is_file() else []
        if old != new:
            out += difflib.unified_diff(old, new, f'scaffold/{rel}', f'final/{rel}')
    return ''.join(out)


def paper_cases(
    layout: Layout,
    builders: Sequence[str],
    examples: Mapping[str, Example] | None = None,
) -> dict[str, Any]:
    """Every case of the paper: the resolved items and the records the figures draw."""
    examples = examples if examples is not None else mme.evaluation_examples(layout)
    out: dict[str, Any] = {}
    for case in CASES:
        st = case['student']
        ids = resolve(layout, case, examples)
        edits = {}
        for b in builders:
            bundle = layout.mhe_bundle(b, st)
            edits[b] = (
                bundle_diff(layout.checkout / 'harness', bundle)
                if bundle.is_dir()
                else None
            )
        out[case['key']] = {
            **case,
            'figures': list(case['figures']),
            'builders': {b: BUILDERS[b].label for b in builders},
            'matches': ids,
            'items': [item(layout, st, i, builders, examples) for i in ids],
            'evolved_edits': edits,
        }
    return out


def main(argv: Sequence[str] | None = None, echo: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--student', choices=ORDER)
    parser.add_argument('--builder', choices=ALL_BUILDERS, default=PAPER_BUILDERS[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--paper', action='store_true', help="the paper's four cases")
    group.add_argument(
        '--list', action='store_true', help='items whose MHE answer differs from Shared'
    )
    group.add_argument('--item', help='one example id under every condition')
    args = parser.parse_args(argv)
    layout = Layout(root=args.root.resolve()) if args.root else Layout()
    if args.paper:
        echo(
            json.dumps(
                paper_cases(layout, PAPER_BUILDERS), indent=1, ensure_ascii=False
            )
        )
        return 0
    if args.student is None:
        parser.error('--list and --item need --student')
    if args.list:
        for row in changed(layout, args.student, args.builder):
            echo(json.dumps(row, ensure_ascii=False))
        return 0
    builders = tuple(dict.fromkeys((args.builder, *PAPER_BUILDERS)))
    echo(
        json.dumps(
            item(layout, args.student, args.item, builders),
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


def changed(layout: Layout, student: str, builder: str) -> list[dict[str, object]]:
    """Items whose MHE label set differs from Shared's, most consequential first."""
    examples = mme.evaluation_examples(layout)
    rows = mme.load_student(layout, student, (builder,), examples)
    shared, harness = rows['shared'], rows[mme.mhe(builder)]
    if shared is None or harness is None:
        return []
    out = []
    for i in sorted(set(shared.predictions) & set(harness.predictions)):
        before = shared.predictions[i].labels
        after = harness.predictions[i].labels
        if {normalize_label(x) for x in before} == {normalize_label(x) for x in after}:
            continue
        delta = harness.board.items[i].recognition - shared.board.items[i].recognition
        out.append(
            {
                'example_id': i,
                'task': examples[i].task.value if examples[i].task else None,
                'gold': list(examples[i].labels),
                'shared': list(before),
                'mhe': list(after),
                'delta': delta,
            }
        )
    return sorted(
        out, key=lambda r: (-abs(float(str(r['delta']))), str(r['example_id']))
    )


if __name__ == '__main__':
    sys.exit(main())
