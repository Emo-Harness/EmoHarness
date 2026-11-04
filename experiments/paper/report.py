"""Every table, figure's data and in-text number of the paper, from the results tree.

    python -m experiments.paper.report                    # the paper's two Builders
    python -m experiments.paper.report --all-builders     # + GPT-6 Luna (App A, 12-test families)
    python -m experiments.paper.report --root /tmp/fake   # a --fake pipeline run

Writes, under ``results/paper/`` (``--out`` overrides):

- ``paper.md`` and ``paper.tex``: Tables 1-6 and the in-text numbers, each
  labelled with where the paper reports it;
- ``paper.json``: everything those are computed from;
- ``figures/figure<N>.json``: the data behind every figure with data: 1(a),
  1(c), 2(b), 3, 4, 5-10 and 11 (the case figures hold each condition's
  answer, the harness records and the evolved edits; ``experiments.paper.cases``).

A part whose runs are missing is reported as an em dash, so the report can be
re-run as runs land. It reads every run through ``experiments.layout``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.common import NAMES, ORDER
from experiments.emotion_context.row import DATASETS
from experiments.layout import ALL_BUILDERS, BUILDERS, INTENTQA, PAPER_BUILDERS, Layout
from experiments.paper import cases, construction, intentqa, mme

DASH: Final = '—'
#: Case panels of figures whose other panels are not cases (Figure 1: a, b, c; Figure 2: a, b).
SINGLE_CASE_PANELS: Final = ('1a', '2b')
Table = tuple[str, str, list[str], list[list[str]]]


# --- formatting --------------------------------------------------------------


def pct(x: float | None, digits: int = 2) -> str:
    return DASH if x is None else f'{100 * x:.{digits}f}'


def pp(x: float | None, digits: int = 2) -> str:
    return DASH if x is None else f'{100 * x:+.{digits}f}'


def interval(test: Mapping[str, Any] | None, holm: bool = False) -> str:
    if not test:
        return DASH
    lo, hi = test.get('ci95') or (None, None)
    text = f'{pp(test["estimate"])} [{pp(lo)}, {pp(hi)}]'
    if holm:
        text += f' p_Holm={test["p_holm"]:.3f}' if 'p_holm' in test else ' p_Holm=—'
    return text


def span(
    values: Sequence[float] | None, scale: float = 100, signed: bool = True
) -> str:
    if not values:
        return DASH
    fmt = (lambda v: f'{scale * v:+.2f}') if signed else (lambda v: f'{scale * v:.2f}')
    return f'{fmt(values[0])} to {fmt(values[1])}'


def label(builder: str) -> str:
    return BUILDERS[builder].label


# --- tables --------------------------------------------------------------------


def table1(m: Mapping[str, Any], builders: Sequence[str]) -> Table:
    head = ['Student', 'Condition', *(t.value for t in mme.TASKS)]
    head += ['Seen', 'Held-out', 'Macro', '95% CI']
    rows = []
    names = [
        ('video', 'Video'),
        ('audio', 'Audio'),
        ('shared', 'Shared'),
        *((mme.mhe(b), f'MHE {label(b)}') for b in builders),
    ]
    for st in m['students']:
        t1 = m['students'][st]['table1']
        for key, name in names:
            r = t1.get(key)
            if r is None:
                rows.append([NAMES[st], name, *([DASH] * (len(head) - 2))])
                continue
            ci = r['ci95']
            rows.append(
                [
                    NAMES[st],
                    name,
                    *(pct(r['per_task'][t.value], 1) for t in mme.TASKS),
                    pct(r['seen'], 1),
                    pct(r['held_out'], 1),
                    pct(r['macro']),
                    f'{pct(ci[0], 1)}-{pct(ci[1], 1)}' if ci else DASH,
                ]
            )
    return (
        'Table 1: Recognition (%) per task and task-macro averages on the evaluation '
        'items; 95% CI is the task-stratified bootstrap interval of the macro.',
        'tab:main',
        head,
        rows,
    )


def table2(q: Mapping[str, Any] | None, builders: Sequence[str]) -> Table:
    head = [
        'Student',
        'Frames',
        'Audio',
        *(f'+ Emotion MHE {label(b)}' for b in builders),
    ]
    rows = []
    for st in ORDER:
        acc = ((q or {}).get('students', {}).get(st) or {}).get('accuracy', {})
        rows.append(
            [NAMES[st], pct(acc.get('none')), pct(acc.get('asr'))]
            + [pct(acc.get(b)) for b in builders]
        )
    return (
        'Table 2: IntentQA test accuracy (%): Frames, Audio, and Audio plus each '
        "Builder's MHE emotion context.",
        'tab:qa',
        head,
        rows,
    )


def table3(
    c: Mapping[str, Any], m: Mapping[str, Any], builders: Sequence[str]
) -> Table:
    head = ['Builder', 'Student', 'Generations', 'Scored', 'Screened', 'Prom.']
    head += ['D_v start', 'D_v final', 'Seen eval.', 'USD']
    rows = []
    for b in builders:
        for st in ORDER:
            r = c['runs'][b].get(st)
            gain = m['students'].get(st, {}).get('mhe_minus_shared', {}).get(b)
            seen = pp(gain['seen_flat']) if gain else DASH
            if r is None:
                rows.append([label(b), NAMES[st], *([DASH] * 6), seen, DASH])
                continue
            rows.append(
                [
                    label(b),
                    NAMES[st],
                    str(r['generations']),
                    str(r['scored']),
                    str(r['screened_out']),
                    str(r['promotions']),
                    pct(r['dval_start']),
                    pct(r['dval_final']),
                    seen,
                    f'{r["usd"]:.2f}',
                ]
            )
    return (
        'Table 3: MHE construction records. D_v: validation Recognition (flat item '
        'mean); Seen eval.: MHE - Shared as a flat mean over the seen-type evaluation '
        'items; USD: Builder, reviewer and proxy-judge spend.',
        'tab:construction',
        head,
        rows,
    )


def table4(m: Mapping[str, Any], builders: Sequence[str]) -> list[Table]:
    family = m['holm_family']
    a = [
        [NAMES[st], label(b), interval(e['mhe_minus_shared'].get(b), holm=True)]
        for st, e in m['students'].items()
        for b in builders
    ]
    b_rows = [
        [NAMES[st], name, interval(e['contrasts'].get(key))]
        for st, e in m['students'].items()
        for key, name in (
            ('audio_minus_video', 'Audio - Video'),
            ('shared_minus_audio', 'Shared - Audio'),
        )
    ]
    return [
        (
            f'Table 4(a): MHE - Shared (pp), task-stratified paired bootstrap, Holm over '
            f'{family} tests.',
            'tab:contrasts_mhe',
            ['Student', 'Builder', 'Delta [95% CI] p_Holm'],
            a,
        ),
        (
            'Table 4(b): Audio - Video and Shared - Audio (pp), pointwise intervals.',
            'tab:contrasts_evidence',
            ['Student', 'Contrast', 'Delta [95% CI]'],
            b_rows,
        ),
    ]


def table5(q: Mapping[str, Any] | None, builders: Sequence[str]) -> Table:
    family = (q or {}).get('holm_family', len(ORDER) * len(builders))
    rows = [
        [
            NAMES[st],
            label(b),
            interval(
                ((q or {}).get('students', {}).get(st) or {})
                .get('emotion_minus_audio', {})
                .get(b),
                holm=True,
            ),
        ]
        for st in ORDER
        for b in builders
    ]
    return (
        f'Table 5: Audio + Emotion - Audio on IntentQA (pp), video-clustered paired '
        f'bootstrap, Holm over {family} tests.',
        'tab:qa_contrasts',
        ['Student', 'Builder', 'Delta [95% CI] p_Holm'],
        rows,
    )


def table6(q: Mapping[str, Any] | None, builders: Sequence[str]) -> Table:
    sizes = (q or {}).get('strata_sizes', {})
    head = ['Stratum', *(f'{NAMES[st]} {label(b)}' for st in ORDER for b in builders)]
    rows = []
    for stratum in intentqa.STRATA:
        rows.append(
            [f'{stratum} (n={sizes.get(stratum, DASH)})']
            + [
                interval(
                    ((q or {}).get('students', {}).get(st) or {})
                    .get('by_type', {})
                    .get(stratum, {})
                    .get(b),
                    holm=True,
                )
                for st in ORDER
                for b in builders
            ]
        )
    return (
        'Table 6: Audio + Emotion - Audio by IntentQA question type (pp); Holm within '
        'each stratum.',
        'tab:qa_types',
        head,
        rows,
    )


# --- in-text numbers -----------------------------------------------------------


def numbers(doc: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(where in the paper, value) for every number the text quotes."""
    m, c, q, d = doc['mme'], doc['construction'], doc['intentqa'], doc['data']
    r, t = m['ranges'], c['totals']
    qr = (q or {}).get('ranges', {})
    out = [
        (
            '§4.1 evaluation items / media',
            f'{d.get("evaluation_items")} / {d.get("evaluation_media")}',
        ),
        (
            '§4.1 seen / held-out evaluation items',
            f'{d.get("evaluation_seen")} / {d.get("evaluation_held_out")}',
        ),
        (
            '§4.1 construction / validation / reserved items',
            json.dumps(d.get('construction_splits')),
        ),
        (
            'App A items excluded for media overlap',
            str(d.get('excluded_media_overlap', DASH)),
        ),
        ('Abstract, §5.1 Audio - Video (pp)', span(r['audio_minus_video'])),
        ('§5.1 Shared - Audio (pp)', span(r['shared_minus_audio'])),
        ('Abstract, §1 MHE - Shared (pp)', span(r['mhe_minus_shared'])),
        ('Abstract, §5.1 MHE - Video (pp)', span(r['mhe_minus_video'])),
        ('§5.1 MHE - Audio (pp)', span(r['mhe_minus_audio'])),
        ('§5.2 MHE - Shared on seen types (pp)', span(r['mhe_minus_shared_seen'])),
        (
            '§5.2 MHE - Shared on held-out types (pp)',
            span(r['mhe_minus_shared_held_out']),
        ),
        ('§5.1, App C set-F1 Shared - Audio (pp)', span(r['f1_shared_minus_audio'])),
        ('App C set-F1 Audio - Video (pp)', span(r['f1_audio_minus_video'])),
        ('App C set-F1 MHE - Shared (pp)', span(r['f1_mhe_minus_shared'])),
        ('App C set-F1 MHE - Audio (pp)', span(r['f1_mhe_minus_audio'])),
        (
            'App C labels per FG-ER / ML-ER item, Shared / Audio',
            f'{span(r["mean_labels_multilabel"]["shared"], 1, False)} / '
            f'{span(r["mean_labels_multilabel"]["audio"], 1, False)}',
        ),
        (
            'App D MME-Emotion student calls by condition',
            json.dumps(r['student_calls']),
        ),
        (
            'App D MME-Emotion student calls, total',
            str(sum(r['student_calls'].values())),
        ),
    ]
    for st, e in m['students'].items():
        for b, g in e['mhe_minus_shared'].items():
            ls = g['label_sets']
            out.append(
                (
                    f'§5.3, §6 label-set changes vs Shared, {NAMES[st]} {label(b)}',
                    f'{ls["changed"]} of {ls["n_items"]} ({pct(ls["changed_fraction"], 1)}%): '
                    f'{ls["improved"]} improve, {ls["worsened"]} worsen, '
                    f'{ls["same_score"]} same',
                )
            )
    for b, tb in t['by_builder'].items():
        out.append(
            (
                f'§5.1, Table 3 {label(b)}: promotions / screened out / scored',
                f'{tb["promotions"]} / {tb["screened_out"]} / {tb["scored"]}',
            )
        )
        out.append(
            (
                f'§5.1 {label(b)}: largest validation gain (pp)',
                pp(tb['max_validation_gain']),
            )
        )
        out.append(
            (f'App D {label(b)}: USD per construction run', span(tb['usd'], 1, False))
        )
    out += [
        (
            '§6 promotions resting on a single validation item',
            f'{t["single_item_promotions"]} of {t["promotions"]}',
        ),
        (
            '§6 candidates gaining >= 8 validation items / blocked by the regression gate',
            f'{t["large_gain_candidates"]} / {t["large_gain_blocked_by_regression"]}',
        ),
        ('§6 ... of which per student', json.dumps(t['large_gain_by_student'])),
        (
            '§6 construction items whose correctness a candidate changes, mean per student',
            json.dumps(
                {
                    k: round(v, 1) if v is not None else None
                    for k, v in t['mean_build_items_changed_by_student'].items()
                }
            ),
        ),
        ('Table 3 generations per run', json.dumps(t['generations'])),
        (
            'App D student calls issued per construction run (cache hits are not re-issued)',
            span(t['student_calls'], 1, False),
        ),
        (
            'App D construction student calls / input tokens / output tokens',
            f'{t["student_calls_total"]} / {t["student_input_tokens_total"]} / '
            f'{t["student_output_tokens_total"]}',
        ),
        ('§5.4 Audio - Frames on IntentQA (pp)', span(qr.get('audio_minus_frames'))),
        (
            '§5.4 Audio + Emotion - Audio on IntentQA (pp)',
            span(qr.get('emotion_minus_audio')),
        ),
        (
            '§5.4 answers whose correctness a context changes',
            span(qr.get('correctness_changed'), 1, False),
        ),
    ]
    for st, e in ((q or {}).get('students') or {}).items():
        for b, agree in e['context_agreement'].items():
            if agree:
                out.append(
                    (
                        f"§5.4 {label(b)} context identical to the frozen bundle's, {NAMES[st]}",
                        f'{agree["identical_context"]} of {agree["videos"]} videos '
                        f'(same label: {agree["same_label"]})',
                    )
                )
    v = doc['volume']
    out.append(
        (
            'App D emotion contexts written / questions answered (all datasets)',
            f'{v["emotion_contexts"]} / {v["questions_answered"]}',
        )
    )
    family = m['holm_family']
    for where, tests in (
        (
            'Tables 1, 4: smallest Holm-adjusted p, MHE - Shared',
            [t for e in m['students'].values() for t in e['mhe_minus_shared'].values()],
        ),
        (
            '§5.4, Table 5: smallest Holm-adjusted p, Audio + Emotion - Audio',
            [
                t
                for e in ((q or {}).get('students') or {}).values()
                for t in e['emotion_minus_audio'].values()
            ],
        ),
        (
            'Table 6: smallest Holm-adjusted p within a stratum',
            [
                t
                for e in ((q or {}).get('students') or {}).values()
                for by_b in e['by_type'].values()
                for t in by_b.values()
            ],
        ),
    ):
        ps = [t['p_holm'] for t in tests if t and 'p_holm' in t]
        out.append((f'{where} ({family} tests)', f'{min(ps):.3f}' if ps else DASH))
    for st, e in m['students'].items():
        f1 = {k: v['macro'] for k, v in e['f1'].items()}
        if 'shared' in f1 and 'audio' in f1:
            out.append(
                (
                    f'§5.1 set-F1 Shared - Audio, {NAMES[st]} (pp)',
                    pp(f1['shared'] - f1['audio']),
                )
            )
        for b in doc['builders']:
            if mme.mhe(b) in f1 and 'audio' in f1:
                out.append(
                    (
                        f'App C set-F1 MHE - Audio, {NAMES[st]} {label(b)} (pp)',
                        pp(f1[mme.mhe(b)] - f1['audio']),
                    )
                )
        for b, g in e['mhe_minus_shared'].items():
            out.append(
                (
                    f'§5.2 MHE - Shared seen / held-out, {NAMES[st]} {label(b)} (pp)',
                    f'{pp(g["seen"])} / {pp(g["held_out"])}',
                )
            )
        held = {k: v['held_out'] for k, v in e['table1'].items()}
        out.append(
            (
                f'§5.2 held-out Recognition, {NAMES[st]} (%)',
                ', '.join(f'{k} {pct(v)}' for k, v in held.items()),
            )
        )
    return out


# --- figures -------------------------------------------------------------------


def figures(doc: Mapping[str, Any], builders: Sequence[str]) -> dict[str, Any]:
    m, c, q = doc['mme'], doc['construction'], doc['intentqa']
    students = m['students']

    def table1_col(st: str, key: str, col: str) -> float | None:
        row = students[st]['table1'].get(key)
        return row[col] if row else None

    keys = ['video', 'audio', 'shared', *(mme.mhe(b) for b in builders)]
    return {
        'figure1c': {
            st: {
                'audio_minus_video': e['contrasts'].get('audio_minus_video'),
                'mhe_minus_shared': e['mhe_minus_shared'],
            }
            for st, e in students.items()
        },
        'figure3': {
            st: {
                k: {
                    'seen': table1_col(st, k, 'seen'),
                    'held_out': table1_col(st, k, 'held_out'),
                }
                for k in keys
            }
            for st in students
        },
        'figure5': (q or {}).get('figure5'),
        'figure6': {
            b: {st: (r or {}).get('trajectory') for st, r in c['runs'][b].items()}
            for b in builders
        },
        'figure7': {
            b: {
                st: {
                    'validation_gain': (
                        r['dval_final'] - r['dval_start']
                        if r and r['dval_start'] is not None
                        else None
                    ),
                    'evaluation_gain': (
                        students.get(st, {}).get('mhe_minus_shared', {}).get(b) or {}
                    ).get('estimate'),
                    'evaluation_ci95': (
                        students.get(st, {}).get('mhe_minus_shared', {}).get(b) or {}
                    ).get('ci95'),
                }
                for st, r in c['runs'][b].items()
            }
            for b in builders
        },
        'figure8': {
            b: {
                st: (e['mhe_minus_shared'].get(b) or {}).get('per_task')
                for st, e in students.items()
            }
            for b in builders
        },
        'figure9': {
            st: {'contrasts': e['contrasts'], 'mhe_minus_shared': e['mhe_minus_shared']}
            for st, e in students.items()
        },
        'figure10': {
            st: e['emotion_minus_audio']
            for st, e in ((q or {}).get('students') or {}).items()
        },
        **_case_figures(doc.get('cases') or {}),
    }


def _case_figures(by_case: Mapping[str, Any]) -> dict[str, Any]:
    """Figures 1(a), 2(b), 4 and 11 from the resolved case records."""
    out: dict[str, Any] = {}
    for case in by_case.values():
        for fig in case['figures']:
            if fig in SINGLE_CASE_PANELS:
                out[f'figure{fig}'] = (
                    case  # one panel of a figure that is not all cases
                )
            else:
                out.setdefault(f'figure{fig[:-1]}', {})[fig[-1]] = case
    return out


# --- assembly ----------------------------------------------------------------


def build(layout: Layout, builders: Sequence[str]) -> dict[str, Any]:
    examples = mme.evaluation_examples(layout)
    m = mme.analyse(layout, builders, examples=examples)
    return {
        'builders': list(builders),
        'data': mme.data_summary(layout, examples),
        'mme': m,
        'construction': construction.analyse(layout, builders),
        'intentqa': intentqa.analyse(layout, builders),
        'supplementary': {
            ds: intentqa.analyse(layout, builders, dataset=ds)
            for ds in DATASETS
            if ds != INTENTQA
        },
        'volume': intentqa.volume(layout, builders),
        'cases': cases.paper_cases(layout, builders, examples),
    }


def tables(doc: Mapping[str, Any], builders: Sequence[str]) -> list[Table]:
    m, q = doc['mme'], doc['intentqa']
    return [
        table1(m, builders),
        table2(q, builders),
        table3(doc['construction'], m, builders),
        *table4(m, builders),
        table5(q, builders),
        table6(q, builders),
    ]


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        '| ' + ' | '.join(head) + ' |',
        '|' + '|'.join('---' for _ in head) + '|',
        *('| ' + ' | '.join(r) + ' |' for r in rows),
    ]


def tex_table(
    caption: str, name: str, head: list[str], rows: list[list[str]]
) -> list[str]:
    def esc(text: str) -> str:
        for old, new in (
            ('%', r'\%'),
            ('_', r'\_'),
            ('&', r'\&'),
            ('—', '---'),
            ('>=', r'$\geq$'),
        ):
            text = text.replace(old, new)
        return text

    return [
        r'\begin{table*}[t]',
        r'\centering\small',
        rf'\caption{{{esc(caption)}}}',
        rf'\label{{{name}}}',
        r'\begin{tabular}{' + 'l' * 2 + 'r' * (len(head) - 2) + '}',
        r'\toprule',
        ' & '.join(esc(h) for h in head) + r' \\',
        r'\midrule',
        *(' & '.join(esc(c) for c in r) + r' \\' for r in rows),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table*}',
        '',
    ]


def render(doc: Mapping[str, Any], builders: Sequence[str]) -> tuple[str, str]:
    labels = ', '.join(label(b) for b in builders)
    md = [
        f'# EmoHarness: paper results ({labels})',
        '',
        f'Generated by `python -m experiments.paper.report`; {DASH} = not run yet.',
        '',
    ]
    tex = ['% Generated by python -m experiments.paper.report', '']
    for caption, name, head, rows in tables(doc, builders):
        md += [f'**{caption}**', '', *md_table(head, rows), '']
        tex += tex_table(caption, name, head, rows)
    md += [
        '## Numbers quoted in the text',
        '',
        *md_table(['Where', 'Value'], [list(x) for x in numbers(doc)]),
        '',
    ]
    return '\n'.join(md), '\n'.join(tex)


def write(doc: Mapping[str, Any], builders: Sequence[str], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    md, tex = render(doc, builders)
    (out / 'paper.md').write_text(md)
    (out / 'paper.tex').write_text(tex)
    (out / 'paper.json').write_text(json.dumps(doc, indent=1, default=str))
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    for name, data in figures(doc, builders).items():
        (fig_dir / f'{name}.json').write_text(json.dumps(data, indent=1, default=str))
    return md


def main(argv: Sequence[str] | None = None, echo: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument(
        '--root', type=Path, default=None, help='run root (default: the checkout)'
    )
    parser.add_argument(
        '--out', type=Path, default=None, help='default: <root>/results/paper'
    )
    parser.add_argument(
        '--all-builders',
        action='store_true',
        help='add GPT-6 Luna, the third configuration of Appendix A (12-test families)',
    )
    args = parser.parse_args(argv)
    layout = Layout(root=args.root.resolve()) if args.root else Layout()
    builders = ALL_BUILDERS if args.all_builders else PAPER_BUILDERS
    doc = build(layout, builders)
    echo(write(doc, builders, args.out or layout.paper))
    return 0


if __name__ == '__main__':
    sys.exit(main())
