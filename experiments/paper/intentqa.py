"""IntentQA emotion context: Table 2, Tables 5 and 6, Figures 5 and 10, Section 5.4 and App D.

Every reply is re-read with the final answer parser
(``experiments.emotion_context.row.parse_letter``); an unparsed reply counts as
wrong. Paired differences resample videos (10,000 clustered draws, the seed of
``experiments.emotion_context.report``) and Holm runs over the 4 x B tests of
each family. Table 6 splits the questions by IntentQA's official type, joined
from the transfer manifest: causal-why (CW), causal-how (CH) and temporal
(TN + TP).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from emoharness.evaluation.bootstrap import holm_adjust
from experiments.common import ORDER
from experiments.emotion_context.report import paired
from experiments.emotion_context.row import DATASETS, parse_letter
from experiments.layout import BUILDERS, INTENTQA, Layout

#: Table 6's strata over IntentQA's official question types.
STRATA: Final = {'why': ('CW',), 'how': ('CH',), 'temporal': ('TN', 'TP')}
BASE: Final = ('none', 'asr')

Rows = dict[str, dict[str, Any]]


# --- loading -------------------------------------------------------------


def questions(layout: Layout, dataset: str) -> dict[str, dict[str, Any]] | None:
    """The dataset's QA items (options, gold, video), as the emotion-context manifests wrote them."""
    record_path = layout.emotion_context / 'manifests/manifest_record.json'
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    if dataset not in record:
        return None
    path = layout.root / record[dataset]['qa']['file']
    return {
        q['example_id']: q for q in map(json.loads, path.read_text().splitlines()) if q
    }


def question_types(layout: Layout) -> dict[str, str]:
    """IntentQA's official type per question (``metadata.qtype`` of the transfer manifest)."""
    path = layout.transfer_manifests / f'{INTENTQA}.jsonl'
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            qtype = (row.get('metadata') or {}).get('qtype')
            if qtype:
                out[row['example_id']] = str(qtype)
    return out


def scored(
    layout: Layout, done: Path, items: Mapping[str, Mapping[str, Any]]
) -> Rows | None:
    """One QA step's replies, re-parsed; None if the step has not finished."""
    if not done.is_file():
        return None
    meta = json.loads(done.read_text())
    rows = [
        json.loads(x)
        for x in (layout.root / meta['predictions']).read_text().splitlines()
        if x.strip()
    ]
    for r in rows:
        choice = parse_letter(r['response_text'], items[r['example_id']]['options'])
        r['choice'] = choice
        r['parsed'] = choice is not None
        r['correct'] = int(choice == r['answer_index'])
    return {r['example_id']: r for r in rows}


def contexts(layout: Layout, done: Path) -> dict[str, dict[str, Any]] | None:
    """One context step's contexts, keyed by video; None if it has not finished."""
    if not done.is_file():
        return None
    meta = json.loads(done.read_text())
    path = layout.root / meta['contexts']
    return {
        c['video_id']: c for c in map(json.loads, path.read_text().splitlines()) if c
    }


def load_student(
    layout: Layout,
    student: str,
    dataset: str,
    builders: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Rows | None]:
    """none, asr and each Builder's emoctx for one student."""
    out: dict[str, Rows | None] = {
        cond: scored(
            layout,
            layout.emotion_context_step(student, 'qa', cond, dataset) / 'done.json',
            items,
        )
        for cond in BASE
    }
    for b in builders:
        step = layout.emotion_context_step(student, 'qa', 'emoctx', dataset, b)
        out[b] = scored(layout, step / 'done.json', items)
    return out


# --- analysis --------------------------------------------------------------


def accuracy(rows: Rows | None) -> float | None:
    return sum(r['correct'] for r in rows.values()) / len(rows) if rows else None


def contrast(
    new: Rows, base: Rows, ids: Iterable[str] | None = None
) -> dict[str, Any] | None:
    """``new - base`` over the shared questions (or ``ids``), video-clustered."""
    common = set(new) & set(base)
    shared = sorted(common if ids is None else common & set(ids))
    if not shared:
        return None
    result = paired(
        shared,
        {i: new[i]['correct'] for i in shared},
        {i: base[i]['correct'] for i in shared},
        {i: new[i]['video_id'] for i in shared},
    )
    result['correctness_changed'] = result['items_gained'] + result['items_lost']
    return result


def _holm(tests: Mapping[str, Mapping[str, Any]], family: int) -> dict[str, float]:
    """Holm over a complete family only; an incomplete one is not corrected."""
    raw = {k: float(t['p_for_holm']) for k, t in tests.items()}
    return dict(holm_adjust(raw)) if len(raw) == family else {}


def context_agreement(
    layout: Layout, student: str, dataset: str, builder: str
) -> dict[str, Any] | None:
    """How often a rebuilt harness writes the frozen bundle's context (Section 5.4)."""
    frozen = contexts(
        layout,
        layout.emotion_context_step(student, 'context', None, dataset) / 'done.json',
    )
    other = contexts(
        layout,
        layout.emotion_context_step(student, 'context', None, dataset, builder)
        / 'done.json',
    )
    if frozen is None or other is None:
        return None
    shared = sorted(set(frozen) & set(other))
    return {
        'videos': len(shared),
        'identical_context': sum(
            frozen[v]['context'] == other[v]['context'] for v in shared
        ),
        'same_label': sum(frozen[v]['label'] == other[v]['label'] for v in shared),
    }


def analyse(
    layout: Layout,
    builders: Sequence[str],
    students: Sequence[str] = ORDER,
    dataset: str = INTENTQA,
) -> dict[str, Any] | None:
    """Table 2, Table 5, Table 6 (IntentQA only), Figures 5 and 10, Section 5.4."""
    items = questions(layout, dataset)
    if items is None:
        return None
    qtypes = question_types(layout) if dataset == INTENTQA else {}
    family = len(students) * len(builders)
    out: dict[str, Any] = {
        'dataset': dataset,
        'n_questions': len(items),
        'n_videos': len({q['video_id'] for q in items.values()}),
        'holm_family': family,
        'students': {},
    }
    table5: dict[str, dict[str, Any]] = {}
    table6: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in STRATA}
    for st in students:
        rows = load_student(layout, st, dataset, builders, items)
        entry: dict[str, Any] = {
            'accuracy': {k: accuracy(v) for k, v in rows.items()},
            'unparsed': {
                k: (sum(not r['parsed'] for r in v.values()) if v else None)
                for k, v in rows.items()
            },
            'audio_minus_frames': (
                contrast(rows['asr'], rows['none'])
                if rows['asr'] and rows['none']
                else None
            ),
            'emotion_minus_audio': {},
            'by_type': {},
            'context_agreement': {},
        }
        for b in builders:
            new, base = rows[b], rows['asr']
            if not new or not base:
                continue
            test = contrast(new, base)
            if test is None:
                continue
            entry['emotion_minus_audio'][b] = test
            table5[f'{st}|{b}'] = test
            for stratum, types in STRATA.items():
                t = contrast(new, base, [i for i in new if qtypes.get(i) in types])
                if t is None:
                    continue
                entry['by_type'].setdefault(stratum, {})[b] = t
                table6[stratum][f'{st}|{b}'] = t
            if not BUILDERS[b].frozen:
                entry['context_agreement'][b] = context_agreement(
                    layout, st, dataset, b
                )
        out['students'][st] = entry
    for key, p in _holm(table5, family).items():
        st, b = key.split('|')
        out['students'][st]['emotion_minus_audio'][b]['p_holm'] = p
    for stratum, tests in table6.items():
        for key, p in _holm(tests, family).items():
            st, b = key.split('|')
            out['students'][st]['by_type'][stratum][b]['p_holm'] = p
    out['strata_sizes'] = {
        stratum: sum(1 for i in items if qtypes.get(i) in types)
        for stratum, types in STRATA.items()
    }
    out['figure5'] = figure5(out, builders)
    out['ranges'] = _ranges(out, builders)
    return out


def figure5(out: Mapping[str, Any], builders: Sequence[str]) -> dict[str, Any]:
    """Mean and individual Audio + Emotion - Audio changes, all questions and per type."""
    strata: dict[str, list[float]] = {'all': []}
    for entry in out['students'].values():
        for b in builders:
            test = entry['emotion_minus_audio'].get(b)
            if test:
                strata['all'].append(test['estimate'])
            for stratum, by_b in entry['by_type'].items():
                if b in by_b:
                    strata.setdefault(stratum, []).append(by_b[b]['estimate'])
    return {
        s: {'mean': sum(v) / len(v) if v else None, 'values': v}
        for s, v in strata.items()
    }


def _ranges(out: Mapping[str, Any], builders: Sequence[str]) -> dict[str, Any]:
    entries = list(out['students'].values())

    def span(values: Iterable[float | None]) -> list[float] | None:
        kept = [v for v in values if v is not None]
        return [min(kept), max(kept)] if kept else None

    return {
        'audio_minus_frames': span(
            (e['audio_minus_frames'] or {}).get('estimate') for e in entries
        ),
        'emotion_minus_audio': span(
            e['emotion_minus_audio'].get(b, {}).get('estimate')
            for e in entries
            for b in builders
        ),
        'correctness_changed': span(
            e['emotion_minus_audio'].get(b, {}).get('correctness_changed')
            for e in entries
            for b in builders
        ),
    }


def volume(
    layout: Layout, builders: Sequence[str], students: Sequence[str] = ORDER
) -> dict[str, int]:
    """App D: emotion contexts written and questions answered, over every dataset run."""
    n_contexts = n_answers = 0
    for st in students:
        for ds in DATASETS:
            for b in builders:
                step = layout.emotion_context_step(st, 'context', None, ds, b)
                n_contexts += _n_written(step / 'done.json')
            for cond in BASE:
                n_answers += _n_written(
                    layout.emotion_context_step(st, 'qa', cond, ds) / 'done.json'
                )
            for b in builders:
                step = layout.emotion_context_step(st, 'qa', 'emoctx', ds, b)
                n_answers += _n_written(step / 'done.json')
    return {'emotion_contexts': n_contexts, 'questions_answered': n_answers}


def _n_written(done: Path) -> int:
    if not done.is_file():
        return 0
    return int(json.loads(done.read_text()).get('n_written', 0))
