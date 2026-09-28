#!/usr/bin/env python3
"""Score the new-dataset transfer evaluation and write report.json / report.md.

    .venv/bin/python scripts/newdata_report.py

Scoring is the harness's own local Recognition (``Scoreboard.from_predictions``),
which for a single-answer item is exact match of the chosen option against the
gold option after the harness's label normalisation. No judge model is called.

Rows mirror the 2026-09-19 ladder: L0 native video, L1 video + audio evidence
text, L3 the shared scaffold, L5 each student's own bundle. The comparisons are
the same four the published page reports -- L1-L0, L3-L1, L5-L3, L5-L0 -- on
exactly the same items, so every difference is paired.

Several questions share one video in both datasets, so every interval and
p-value resamples VIDEOS, not questions: resampling questions would treat the
answers to one clip as independent draws and overstate the precision. Holm is
applied per dataset over its 16 comparisons (4 students x 4), as the published
evaluation did over its 16, with the finite-resample floor 2/(B+1) so a
bootstrap that happened to draw no opposite-sign resample is never read as p=0.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from emoharness.data.manifest import read_manifest
from emoharness.evaluation.bootstrap import _two_sided_p_value, holm_adjust
from emoharness.evaluation.scoreboard import Scoreboard
from emoharness.runner.rows import PredictionRecord

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'results/newdata/intentqa_socialiq_20260923'
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
SLOTS = (('L0', 'L0'), ('L1', 'L1'), ('L3', 'L3_scaffold'), ('L5', 'L5_own'))
COMPARISONS = (
    ('L1', 'L0', 'perception text over bare video'),
    ('L3', 'L1', 'shared scaffold over the same information'),
    ('L5', 'L3', "the student's own bundle over the scaffold"),
    ('L5', 'L0', 'total effect over the bare model'),
)
RESAMPLES = 10000
SEED = 20260923
FLOOR = 2 / (RESAMPLES + 1)


def load_scores(student: str, slot: str, dataset: str, examples: dict):
    done = RUN / student / slot / dataset / 'done.json'
    if not done.is_file():
        return None, None
    meta = json.loads(done.read_text())
    preds = [
        PredictionRecord.model_validate_json(line)
        for line in (ROOT / meta['predictions']).read_text().splitlines()
        if line.strip()
    ]
    board = Scoreboard.from_predictions(preds, examples)
    scores = {p.example_id: board.items[p.example_id].recognition for p in preds}
    extra = {
        'no_parsable_answer': sum(1 for p in preds if not p.labels),
        'mean_target_calls': sum(p.target_calls for p in preds) / len(preds),
        'bundle_hash': meta.get('bundle_hash'),
        'seconds': meta.get('seconds'),
    }
    return scores, extra


def video_arrays(ids: list[str], values: dict[str, float], examples: dict):
    by_video: dict[str, list[float]] = defaultdict(list)
    for i in ids:
        by_video[examples[i].source_id].append(values[i])
    videos = sorted(by_video)
    sums = np.array([sum(by_video[v]) for v in videos])
    counts = np.array([len(by_video[v]) for v in videos])
    return sums, counts


def clustered(sums: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    idx = rng.integers(0, len(sums), size=(RESAMPLES, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def main() -> int:
    record = json.loads((RUN / 'manifests/manifest_record.json').read_text())
    out: dict = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scoring': 'local Recognition, exact option match; no judge',
        'ci': f'{RESAMPLES} video-clustered bootstrap resamples, seed {SEED}',
        'holm': 'per dataset over 16 comparisons, p floored at 2/(B+1)',
        'rendering': record.get('rendering'),
        'datasets': {},
    }
    md = [
        '# New-dataset transfer of the 2026-09-19 per-student harnesses',
        '',
        'Same items, same prompt rendering, same videos in every row. Accuracy and '
        'paired differences with 95% CIs resampled over videos; Holm per dataset '
        'over 16 comparisons.',
        '',
    ]
    for ds, (label, k) in DATASETS.items():
        examples = {e.example_id: e for e in read_manifest(ROOT / record[ds]['file'])}
        ids = sorted(examples)
        section = {
            'label': label,
            'chance': 1 / k,
            'n_items': len(ids),
            'n_videos': record[ds]['n_videos'],
            'n_excluded': record[ds]['n_excluded'],
            'rows': {},
            'comparisons': {},
        }
        md += [
            f'## {label}',
            '',
            f'{len(ids)} items, {record[ds]["n_videos"]} videos, '
            f'{record[ds]["n_excluded"]} excluded; chance {100 / k:.0f}%.',
            '',
            '| student | L0 | L1 | L3 | L5 |',
            '|---|---|---|---|---|',
        ]
        scores: dict = {}
        raw_p: dict[str, float] = {}
        for st in ORDER:
            cells = []
            for row, slot in SLOTS:
                sc, extra = load_scores(st, slot, ds, examples)
                if sc is None:
                    cells.append('pending')
                    continue
                if set(sc) != set(ids):
                    raise SystemExit(
                        f'{st}/{slot}/{ds} covers {len(sc)} ids, not the '
                        f'{len(ids)} in the manifest'
                    )
                scores[(st, row)] = sc
                sums, counts = video_arrays(ids, sc, examples)
                boot = clustered(sums, counts, np.random.default_rng(SEED))
                acc = sum(sc.values()) / len(ids)
                section['rows'][f'{st}/{row}'] = {
                    'accuracy': acc,
                    'ci95': [
                        float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5)),
                    ],
                    **extra,
                }
                cells.append(f'{100 * acc:.2f}%')
            md.append(f'| {NAMES[st]} | ' + ' | '.join(cells) + ' |')
        md += [
            '',
            '| student | comparison | Δ (pp) | 95% CI | p | Holm p |',
            '|---|---|---|---|---|---|',
        ]
        for st in ORDER:
            for hi, lo, meaning in COMPARISONS:
                if (st, hi) not in scores or (st, lo) not in scores:
                    continue
                delta = {i: scores[(st, hi)][i] - scores[(st, lo)][i] for i in ids}
                sums, counts = video_arrays(ids, delta, examples)
                boot = clustered(sums, counts, np.random.default_rng(SEED))
                est = sum(delta.values()) / len(ids)
                p = _two_sided_p_value(boot, est)
                key = f'{st} {hi} - {lo}'
                raw_p[key] = max(p, FLOOR) if est != 0 else 1.0
                section['comparisons'][key] = {
                    'meaning': meaning,
                    'estimate': est,
                    'ci95': [
                        float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5)),
                    ],
                    'p_raw': p,
                    'p_for_holm': raw_p[key],
                    'items_changed': sum(1 for v in delta.values() if v != 0),
                }
        holm = holm_adjust(raw_p) if raw_p else {}
        for key, adj in holm.items():
            section['comparisons'][key]['p_holm'] = adj
        for st in ORDER:
            for hi, lo, _ in COMPARISONS:
                c = section['comparisons'].get(f'{st} {hi} - {lo}')
                if c is None:
                    continue
                lo_ci, hi_ci = c['ci95']
                md.append(
                    f'| {NAMES[st]} | {hi}−{lo} | {100 * c["estimate"]:+.2f} | '
                    f'[{100 * lo_ci:+.2f}, {100 * hi_ci:+.2f}] | {c["p_raw"]:.4f} | '
                    f'{c.get("p_holm", float("nan")):.4f} |'
                )
        md.append('')
        out['datasets'][ds] = section
    (RUN / 'report.json').write_text(json.dumps(out, indent=2))
    (RUN / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
