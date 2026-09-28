#!/usr/bin/env python3
"""Score the emotion-context experiment and write report.json / report.md.

    .venv/bin/python scripts/emoctx_report.py

Accuracy is exact match of the parsed option letter against the gold index;
an unparsed reply counts as wrong and is reported per arm. The three
comparisons per student are paired on identical questions:

    asr - none      what the raw transcript + prosody text adds
    emoctx - asr    what the harness's emotion analysis adds on top of it
    emoctx - none   the total effect of the harness context over the bare model

Intervals and p-values resample videos (10,000 clustered bootstrap draws, the
same seed as the transfer report), and Holm runs per dataset over its 12
comparisons with the finite-resample floor 2/(B+1).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from emoharness.evaluation.bootstrap import _two_sided_p_value, holm_adjust

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emoctx_row import parse_letter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'results/newdata/emotion_context_20260925'
ORDER = ('qwen3vl_8b', 'qwen3vl_4b', 'qwen3_5_4b', 'gemma4_e4b')
NAMES = {
    'qwen3vl_8b': 'Qwen3-VL-8B',
    'qwen3vl_4b': 'Qwen3-VL-4B',
    'qwen3_5_4b': 'Qwen3.5-4B',
    'gemma4_e4b': 'Gemma-4-E4B',
}
DATASETS = {
    'intentqa_test': ('IntentQA test', 5),
    'social_iq_val': ('Social-IQ 2.0 val', 4),
}
CONDITIONS = ('none', 'asr', 'emoctx')
COMPARISONS = (
    ('asr', 'none', 'transcript + prosody over frames alone'),
    ('emoctx', 'asr', 'harness emotion analysis over the same raw perception'),
    ('emoctx', 'none', 'total effect of the harness context'),
)
RESAMPLES = 10000
SEED = 20260923
FLOOR = 2 / (RESAMPLES + 1)


def load(student: str, cond: str, ds: str, options: dict):
    """One arm, every reply re-read with the final parser so all students share it."""
    done = RUN / student / f'qa_{cond}' / ds / 'done.json'
    if not done.is_file():
        return None
    meta = json.loads(done.read_text())
    rows = [
        json.loads(x)
        for x in (ROOT / meta['predictions']).read_text().splitlines()
        if x
    ]
    for r in rows:
        r['choice'] = parse_letter(r['response_text'], options[r['example_id']])
        r['parsed'] = r['choice'] is not None
        r['correct'] = int(r['choice'] == r['answer_index'])
    return {r['example_id']: r for r in rows}, meta


def clustered(ids, values, video_of, rng):
    by_video = defaultdict(list)
    for i in ids:
        by_video[video_of[i]].append(values[i])
    videos = sorted(by_video)
    sums = np.array([sum(by_video[v]) for v in videos])
    counts = np.array([len(by_video[v]) for v in videos])
    idx = rng.integers(0, len(videos), size=(RESAMPLES, len(videos)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def paired(ids, hi: dict, lo: dict, video_of: dict) -> dict:
    """Paired difference hi - lo over ``ids``, video-clustered CI and two-sided p."""
    delta = {i: hi[i] - lo[i] for i in ids}
    boot = clustered(ids, delta, video_of, np.random.default_rng(SEED))
    est = sum(delta.values()) / len(ids)
    p = _two_sided_p_value(boot, est)
    return {
        'estimate': est,
        'n_items': len(ids),
        'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        'p_raw': p,
        'p_for_holm': max(p, FLOOR) if est != 0 else 1.0,
        'items_gained': sum(1 for v in delta.values() if v > 0),
        'items_lost': sum(1 for v in delta.values() if v < 0),
    }


def contexts_summary(student: str, ds: str):
    done = RUN / student / 'context' / ds / 'done.json'
    if not done.is_file():
        return None
    meta = json.loads(done.read_text())
    rows = [
        json.loads(x) for x in (ROOT / meta['contexts']).read_text().splitlines() if x
    ]
    return {
        'n_videos': len(rows),
        'n_fallback': sum(r['fallback_used'] for r in rows),
        'labels': dict(Counter(r['label'] for r in rows).most_common()),
        'mean_chars': round(sum(len(r['context']) for r in rows) / len(rows)),
        'bundle_hash': meta.get('bundle_hash'),
        'seconds': meta.get('seconds'),
    }


def main() -> int:
    record = json.loads((RUN / 'manifests/manifest_record.json').read_text())
    out = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scoring': 'parsed option letter == gold index; unparsed replies count as wrong',
        'ci': f'{RESAMPLES} video-clustered bootstrap resamples, seed {SEED}',
        'holm': 'per dataset over 12 comparisons, p floored at 2/(B+1)',
        'emotion_prompt': record['intentqa_test']['emotion_prompt'],
        'datasets': {},
    }
    md = [
        '# Does the harness emotion context help the bare model?',
        '',
        "Each student's own harness writes one question-independent emotion context per "
        'video (its native MME-Emotion task, 7 basic emotions). The same bare student then '
        'answers the lettered multiple-choice question from the same cached frames, with no '
        'text (**none**), with the transcript + prosody text (**asr**), or with that text plus '
        'the harness emotion context (**emoctx**).',
        '',
    ]
    for ds, (label, k) in DATASETS.items():
        qa = [
            json.loads(x)
            for x in (ROOT / record[ds]['qa']['file']).read_text().splitlines()
            if x
        ]
        ids = sorted(q['example_id'] for q in qa)
        video_of = {q['example_id']: q['video_id'] for q in qa}
        options = {q['example_id']: q['options'] for q in qa}
        section = {
            'label': label,
            'chance': 1 / k,
            'n_items': len(ids),
            'n_videos': record[ds]['n_videos'],
            'arms': {},
            'comparisons': {},
            'contexts': {},
        }
        md += [
            f'## {label}',
            '',
            f'{len(ids)} questions on {record[ds]["n_videos"]} videos; '
            f'chance {100 / k:.0f}%.',
            '',
            '| student | none | asr | emoctx | unparsed (n/a/e) |',
            '|---|---|---|---|---|',
        ]
        scores, parsed, raw_p = {}, {}, {}
        for st in ORDER:
            section['contexts'][st] = contexts_summary(st, ds)
            cells, unparsed = [], []
            for cond in CONDITIONS:
                got = load(st, cond, ds, options)
                if got is None:
                    cells.append('pending')
                    unparsed.append('–')
                    continue
                rows, meta = got
                if set(rows) != set(ids):
                    raise SystemExit(
                        f'{st}/{cond}/{ds}: {len(rows)} ids, manifest has {len(ids)}'
                    )
                sc = {i: rows[i]['correct'] for i in ids}
                scores[(st, cond)] = sc
                parsed[(st, cond)] = {i for i in ids if rows[i]['parsed']}
                boot = clustered(ids, sc, video_of, np.random.default_rng(SEED))
                acc = sum(sc.values()) / len(ids)
                n_un = sum(not rows[i]['parsed'] for i in ids)
                section['arms'][f'{st}/{cond}'] = {
                    'accuracy': acc,
                    'ci95': [
                        float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5)),
                    ],
                    'n_unparsed': n_un,
                    'n_errors': sum(rows[i]['error'] is not None for i in ids),
                    'n_truncated': sum(bool(rows[i].get('truncated')) for i in ids),
                    'qa_max_tokens': meta.get('qa_max_tokens'),
                    'topup': meta.get('topup'),
                    'mean_input_tokens': sum(rows[i]['input_tokens'] for i in ids)
                    / len(ids),
                    'seconds': meta.get('seconds'),
                }
                cells.append(f'{100 * acc:.2f}%')
                unparsed.append(str(n_un))
            md.append(
                f'| {NAMES[st]} | ' + ' | '.join(cells) + f' | {"/".join(unparsed)} |'
            )
        md += [
            '',
            '| student | comparison | Δ (pp) | 95% CI | p | Holm p |',
            '|---|---|---|---|---|---|',
        ]
        for st in ORDER:
            for hi, lo, meaning in COMPARISONS:
                if (st, hi) not in scores or (st, lo) not in scores:
                    continue
                key = f'{st} {hi} - {lo}'
                section['comparisons'][key] = {
                    'meaning': meaning,
                    **paired(ids, scores[(st, hi)], scores[(st, lo)], video_of),
                }
                raw_p[key] = section['comparisons'][key]['p_for_holm']
        for key, adj in (holm_adjust(raw_p) if raw_p else {}).items():
            section['comparisons'][key]['p_holm'] = adj
        for key, c in section['comparisons'].items():
            st, hi, _, lo = key.split(' ')
            lo_ci, hi_ci = c['ci95']
            p_text = f'<{FLOOR:.4f}' if c['p_raw'] < FLOOR else f'{c["p_raw"]:.4f}'
            md.append(
                f'| {NAMES[st]} | {hi}−{lo} | {100 * c["estimate"]:+.2f} | '
                f'[{100 * lo_ci:+.2f}, {100 * hi_ci:+.2f}] | {p_text} | '
                f'{c.get("p_holm", float("nan")):.4f} |'
            )
        # Sensitivity: only questions every arm of the student answered with a letter,
        # so a refusal or truncation in one arm cannot drive the difference.
        md += [
            '',
            'Sensitivity, questions answered in all three arms (no Holm):',
            '',
            '| student | n | none | asr | emoctx | emoctx−asr [95% CI] | emoctx−none [95% CI] |',
            '|---|---|---|---|---|---|---|',
        ]
        section['answered_in_all_arms'] = {}
        for st in ORDER:
            if not all((st, c) in scores for c in CONDITIONS):
                continue
            common = sorted(set.intersection(*(parsed[(st, c)] for c in CONDITIONS)))
            accs = {
                c: sum(scores[(st, c)][i] for i in common) / len(common)
                for c in CONDITIONS
            }
            cmp = {
                f'{hi} - {lo}': paired(
                    common, scores[(st, hi)], scores[(st, lo)], video_of
                )
                for hi, lo, _ in COMPARISONS
            }
            section['answered_in_all_arms'][st] = {
                'n_items': len(common),
                'accuracy': accs,
                'comparisons': cmp,
            }
            cells = [f'{100 * accs[c]:.2f}%' for c in CONDITIONS]
            diffs = [
                f'{100 * cmp[k]["estimate"]:+.2f} [{100 * cmp[k]["ci95"][0]:+.2f}, '
                f'{100 * cmp[k]["ci95"][1]:+.2f}]'
                for k in ('emoctx - asr', 'emoctx - none')
            ]
            md.append(
                f'| {NAMES[st]} | {len(common)} | ' + ' | '.join(cells + diffs) + ' |'
            )
        md += [
            '',
            '| student | contexts | harness fallbacks | predicted emotions |',
            '|---|---|---|---|',
        ]
        for st in ORDER:
            c = section['contexts'][st]
            if c:
                labels = ', '.join(f'{k} {v}' for k, v in c['labels'].items())
                md.append(
                    f'| {NAMES[st]} | {c["n_videos"]} | {c["n_fallback"]} | {labels} |'
                )
        md.append('')
        out['datasets'][ds] = section
    (RUN / 'report.json').write_text(json.dumps(out, indent=2))
    (RUN / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
