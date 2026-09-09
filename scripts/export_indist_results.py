"""Package the in-distribution effect check for a results page hosted elsewhere.

    .venv/bin/python scripts/export_indist_results.py            # -> results/export/

Writes one directory (and a .tar.gz of it) with everything a static
before/after page needs and nothing it must not carry: no benchmark media,
no frames, no keys. Numbers are recomputed from the raw predictions so the
page's provenance is the same as docs/run-indist.md's.

  summary.json            headline numbers, per-task scores, paired bootstrap
                          CIs for every delta, construction trajectory per
                          replicate, gate outcomes, cost, metadata
  items.jsonl             one line per subset item: task, gold, label set, and
                          each row's prediction, score, and response text
  harness_outputs/        L3 and L4 per-item harness records (evidence, steps)
  construction/           per replicate: loop_result, each round's proposal,
                          candidates and scores, gate reasons, accepted patch;
                          the accepted patch as a unified diff
  metrics/                analyze.py's tables as written
  leakage_report.json, subset_1000.json, docs/run-indist.md, README.md
"""

from __future__ import annotations

import difflib
import json
import shutil
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import read_manifest
from emoharness.data.tasks import SEEN_TASKS, Task
from emoharness.evaluation.bootstrap import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'export' / 'emoharness_indist_2026-09-08'
ROWS = ('L0', 'L3', 'L4', 'L6')
ROW_LABELS = {
    'L0': 'Qwen3-VL-8B Direct (official prompt, native video)',
    'L3': 'Stage 1 scaffold (human-written harness, 4 calls)',
    'L4': 'Stage 2 harness (Luna Builder, replicate 0, 1 accepted patch)',
    'L6': 'GPT 5.6 Luna Direct (frames + audio text)',
}
DELTAS = (('L0', 'L3'), ('L3', 'L4'), ('L0', 'L4'), ('L0', 'L6'))
TASKS = [t.value for t in Task]
SEEN = {t.value for t in SEEN_TASKS}
RESAMPLES = 5000
PAID = ('builder', 'judge_proxy', 'teacher_direct')


def load_rows() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    examples = {
        e.example_id: e
        for e in read_manifest(ROOT / 'data/manifests/mme_emotion.jsonl')
    }
    rows: dict[str, dict[str, Any]] = {}
    for row in ROWS:
        rows[row] = {}
        for line in open(ROOT / f'results/raw_predictions/{row}/subset.jsonl'):
            r = json.loads(line)
            e = examples[r['example_id']]
            ls = LabelSet.from_prompt(e.prompt)
            rows[row][r['example_id']] = {
                'labels': r['labels'],
                'recognition': ls.recognition(r['labels'], e.labels),
                'parse_failures': r.get('parse_failures', 0),
                'target_calls': r.get('target_calls', 1),
                'response_text': r.get('response_text', ''),
                'latency_s': r.get('latency_s'),
            }
    return examples, rows


def macro(
    scores: dict[str, float], task_of: dict[str, str], only: set[str] | None = None
) -> dict[str, float]:
    per = defaultdict(list)
    for k, v in scores.items():
        if only is None or task_of[k] in only:
            per[task_of[k]].append(v)
    by_task = {t: sum(v) / len(v) for t, v in per.items()}
    return by_task


def group_means(by_task: dict[str, float]) -> dict[str, float]:
    seen = [v for t, v in by_task.items() if t in SEEN]
    held = [v for t, v in by_task.items() if t not in SEEN]
    return {
        'seen': sum(seen) / len(seen),
        'held_out': sum(held) / len(held),
        'overall': sum(by_task.values()) / len(by_task),
    }


def construction_summary() -> list[dict[str, Any]]:
    out = []
    for rep_dir in sorted((ROOT / 'results/construction/stage2').glob('[0-9]')):
        lr = json.load(open(rep_dir / 'loop_result.json'))
        rounds = []
        incumbent = None
        for rd in sorted(rep_dir.glob('round_*')):
            cand = json.load(open(rd / 'candidates.json'))
            acc = json.load(open(rd / 'accepted.json'))
            pkt = json.load(open(rd / 'packet.json'))
            gates = {_norm(g['name']): g for g in cand['gates']}
            evaluated = []
            for c in cand['evaluated']:
                s = c['scores']
                g = gates.get(_norm(c['name']))
                for reason in g['reasons'] if g else ():
                    if 'does not exceed incumbent' in reason:
                        incumbent = float(reason.split('incumbent ')[1].split(' ')[0])
                evaluated.append(
                    {
                        'name': c.get('patch_name', c['name']),
                        'targeted_cluster': c['targeted_cluster'],
                        'val_recognition': s['recognition'],
                        'reasoning': s['reasoning'],
                        'regression_rate': s['regression_rate'],
                        'mean_calls': s['mean_calls'],
                        'p95_latency_s': s['p95_latency_s'],
                        'accepted': bool(acc.get('accepted'))
                        and _norm(acc['accepted']) == _norm(c['name']),
                        'gate_reasons': g['reasons'] if g else [],
                    }
                )
            rounds.append(
                {
                    'round': int(rd.name.split('_')[1]),
                    'build_recognition_incumbent': pkt['build_recognition'],
                    'candidates': evaluated,
                    'accepted': acc.get('accepted'),
                    'accepted_rationale': acc.get('rationale'),
                }
            )
        out.append(
            {
                'replicate': int(rep_dir.name),
                'stopped_reason': lr['stopped_reason'],
                'rounds_run': len(lr['rounds']),
                'accepted_per_round': [r['accepted'] for r in lr['rounds']],
                'val_incumbent_start': 0.4772,
                'rounds': rounds,
                'final_manifest': json.load(open(rep_dir / 'final/manifest.json')),
            }
        )
    return out


def _norm(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name).strip('.')


def accepted_patch_diff() -> str:
    base = ROOT / 'harness/prompts/verify.md'
    final = ROOT / 'results/construction/stage2/0/final/prompts/verify.md'
    return ''.join(
        difflib.unified_diff(
            base.read_text().splitlines(True),
            final.read_text().splitlines(True),
            fromfile='harness/prompts/verify.md (Stage 1 scaffold)',
            tofile='stage2/0/final/prompts/verify.md (after the accepted patch)',
        )
    )


def cost_summary() -> dict[str, Any]:
    by_role: dict[str, dict[str, float]] = defaultdict(
        lambda: {'calls': 0, 'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
    )
    for ledger in sorted((ROOT / 'results').glob('ledger*.jsonl')):
        for line in open(ledger):
            r = json.loads(line)
            b = by_role[r['role']]
            b['calls'] += 1
            b['cost_usd'] += r['cost_usd']
            b['input_tokens'] += r['input_tokens']
            b['output_tokens'] += r['output_tokens']
    return {
        'by_role': by_role,
        'luna_total_usd': sum(v['cost_usd'] for k, v in by_role.items() if k in PAID),
        'student_calls_local': by_role.get('target', {}).get('calls', 0),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'harness_outputs').mkdir(parents=True)
    (OUT / 'construction').mkdir()
    (OUT / 'metrics').mkdir()
    examples, rows = load_rows()
    ids = sorted(rows['L0'])
    task_of = {i: examples[i].task.value for i in ids}

    # per-row scores
    row_scores = {}
    for row in ROWS:
        rec = {i: rows[row][i]['recognition'] for i in ids}
        by_task = macro(rec, task_of)
        row_scores[row] = {
            'label': ROW_LABELS[row],
            'by_task': by_task,
            **group_means(by_task),
            'parse_failures': sum(rows[row][i]['parse_failures'] for i in ids),
            'mean_calls': sum(rows[row][i]['target_calls'] for i in ids) / len(ids),
        }

    # paired bootstrap deltas, overall / seen / held-out / per task
    deltas = []
    for a, b in DELTAS:
        ra = {i: rows[a][i]['recognition'] for i in ids}
        rb = {i: rows[b][i]['recognition'] for i in ids}
        entry: dict[str, Any] = {'from': a, 'to': b}
        for scope, keep in (
            ('overall', None),
            ('seen', SEEN),
            ('held_out', set(TASKS) - SEEN),
        ):
            sub = [i for i in ids if keep is None or task_of[i] in keep]
            res = paired_bootstrap(
                {i: ra[i] for i in sub}, {i: rb[i] for i in sub}, task_of, RESAMPLES
            )
            entry[scope] = {
                'estimate': res.estimate,
                'ci_low': res.ci_low,
                'ci_high': res.ci_high,
                'p_value': res.p_value,
            }
        entry['by_task'] = {}
        for t in TASKS:
            sub = [i for i in ids if task_of[i] == t]
            res = paired_bootstrap(
                {i: ra[i] for i in sub}, {i: rb[i] for i in sub}, task_of, RESAMPLES
            )
            entry['by_task'][t] = {
                'estimate': res.estimate,
                'ci_low': res.ci_low,
                'ci_high': res.ci_high,
            }
        entry['items_better'] = sum(1 for i in ids if rb[i] > ra[i])
        entry['items_worse'] = sum(1 for i in ids if rb[i] < ra[i])
        deltas.append(entry)

    leakage = json.load(open(ROOT / 'data/checksums/leakage_report.json'))
    subset = json.load(open(ROOT / 'data/splits/subset_1000.json'))
    l6, l0, l4 = (
        row_scores['L6']['overall'],
        row_scores['L0']['overall'],
        row_scores['L4']['overall'],
    )
    l3 = row_scores['L3']['overall']
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'protocol': {
            'name': 'in-distribution effect check (original README 10/10/80 split)',
            'construction_data': 'MME-Emotion seen tasks (ER_Lab, ER_Wild, SA): build 400 / val 394 / ext_test 129, split by source id',
            'evaluation': '1,000-item subset of the remaining 5,577 test items, 125 per task (seed 20260907)',
            'metric': 'Recognition computed locally: exact label match after the official five-synonym normalisation; C/K on FG_ER and ML_ER. No official GPT-4o judge.',
            'student': 'Qwen/Qwen3-VL-8B-Instruct via vLLM 0.11.2, temperature 0, up to 32 frames per clip, sources capped at 640 px wide',
            'builder': 'gpt-5.6-luna, reasoning_effort high, up to 4 candidates per round, 6 rounds max, early stop after 2 rounds without acceptance',
            'gates': 'Recognition must strictly improve on D_val; regression of already-solved D_val items <= 5%; proxy Reasoning (Luna) within 2 points; mean calls <= 4.5 (amended from 4.0, see docs)',
            'teacher_row': 'gpt-5.6-luna answering the official prompt with the cached frames and audio text',
            'notes_doc': 'docs/run-indist.md',
        },
        'headline': {
            'harness_gain_L3_minus_L0': l3 - l0,
            'builder_gain_L4_minus_L3': l4 - l3,
            'teacher_gap_L6_minus_L0': l6 - l0,
            'gap_closed_by_scaffold': (l3 - l0) / (l6 - l0),
            'gap_closed_by_L4': (l4 - l0) / (l6 - l0),
        },
        'rows': row_scores,
        'deltas': deltas,
        'tasks': {
            t: {
                'group': 'seen' if t in SEEN else 'held_out',
                'n': sum(1 for i in ids if task_of[i] == t),
            }
            for t in TASKS
        },
        'construction': construction_summary(),
        'cost': cost_summary(),
        'leakage': {
            'checked_pairs': leakage.get('checked_pairs'),
            'flags': leakage.get('flags'),
            'note': 'Under this protocol construction and test items come from the same benchmark; identical-media pairs are MME reusing a recording under two tasks. Excluding flagged subset items moves no row by more than 0.2 points.',
        },
        'subset': {
            'size': len(subset['ids']),
            'seed': subset.get('seed'),
            'size_per_task': subset.get('size_per_task'),
        },
    }
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)

    with open(OUT / 'items.jsonl', 'w') as fh:
        for i in ids:
            e = examples[i]
            fh.write(
                json.dumps(
                    {
                        'example_id': i,
                        'task': task_of[i],
                        'group': 'seen' if task_of[i] in SEEN else 'held_out',
                        'gold': list(e.labels),
                        'label_set': list(e.label_set),
                        'prompt': e.prompt,
                        'duration_seconds': e.duration_seconds,
                        'rows': {row: rows[row][i] for row in ROWS},
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )

    for row in ('L3', 'L4'):
        shutil.copyfile(
            ROOT / f'results/raw_predictions/{row}/subset.outputs.jsonl',
            OUT / 'harness_outputs' / f'{row}.jsonl',
        )
    for rep_dir in sorted((ROOT / 'results/construction/stage2').glob('[0-9]')):
        dest = OUT / 'construction' / f'replicate_{rep_dir.name}'
        shutil.copytree(rep_dir, dest, ignore=shutil.ignore_patterns('candidates'))
    (OUT / 'construction' / 'accepted_patch_replicate0.diff').write_text(
        accepted_patch_diff()
    )
    shutil.copytree(ROOT / 'harness', OUT / 'construction' / 'stage1_scaffold')
    for f in (ROOT / 'results/metrics').glob('*'):
        shutil.copyfile(f, OUT / 'metrics' / f.name)
    shutil.copyfile(
        ROOT / 'data/checksums/leakage_report.json', OUT / 'leakage_report.json'
    )
    shutil.copyfile(ROOT / 'data/splits/subset_1000.json', OUT / 'subset_1000.json')
    (OUT / 'docs').mkdir()
    shutil.copyfile(ROOT / 'docs/run-indist.md', OUT / 'docs' / 'run-indist.md')
    (OUT / 'README.md').write_text(_readme(summary))
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=OUT.name)
    print(OUT)
    print(archive)


def _readme(summary: dict[str, Any]) -> str:
    rows = summary['rows']
    h = summary['headline']
    lines = [
        '# EmoHarness in-distribution effect check, 2026-09-08/09',
        '',
        'Student Qwen3-VL-8B-Instruct, Builder GPT 5.6 Luna, MME-Emotion. Local Recognition on a',
        '1,000-item subset (125 per task). No benchmark media is included in this package.',
        '',
        '| Row | seen | held-out | overall |',
        '|---|---|---|---|',
    ]
    for row, s in rows.items():
        lines.append(
            f'| {row} {s["label"]} | {100 * s["seen"]:.1f} | {100 * s["held_out"]:.1f} | {100 * s["overall"]:.1f} |'
        )
    lines += [
        '',
        f'Harness gain L3 - L0: {100 * h["harness_gain_L3_minus_L0"]:+.1f}. Builder gain L4 - L3: {100 * h["builder_gain_L4_minus_L3"]:+.1f}'
        ' (net 2 items of 1,000; inside the paired bootstrap interval; read as no measurable improvement).',
        f'Teacher gap L6 - L0: {100 * h["teacher_gap_L6_minus_L0"]:+.1f}; the scaffold closes {100 * h["gap_closed_by_scaffold"]:.0f}% of it.',
        '',
        '## Files',
        '',
        '- `summary.json`: everything above plus paired-bootstrap CIs for each delta (overall, seen, held-out, per task),',
        '  the Stage-2 construction trajectory of all three replicates (every candidate with its D_val score, regression',
        '  rate, reasoning score, and gate reasons), cost by role, leakage counts, protocol notes.',
        "- `items.jsonl`: one line per item with task, gold labels, label set, prompt, and each row's predicted labels,",
        '  Recognition, parse failures, call count, latency, and full response text (for before/after examples).',
        "- `harness_outputs/L3.jsonl`, `L4.jsonl`: the harness's per-item evidence record, hypotheses, verification,",
        '  five steps, and stage traces.',
        '- `construction/replicate_{0,1,2}/`: loop_result.json, per round packet.json/packet.txt (what the Builder saw),',
        '  proposal.json (what it answered), candidates.json (scores and gates), accepted.json, final/ bundle.',
        '- `construction/accepted_patch_replicate0.diff`: the one accepted patch, as a diff of prompts/verify.md.',
        '- `construction/stage1_scaffold/`: the human-written harness bundle every replicate started from.',
        '- `metrics/`: analyze.py tables (ladder.json/md, per_task.json, multilabel.json, cost.json for machine 1 only).',
        '- `leakage_report.json`, `subset_1000.json`, `docs/run-indist.md` (protocol, deviations, amendments, results).',
        '',
        '## Caveats to state on the page',
        '',
        "- In-distribution protocol (build/val from the benchmark's own seen tasks), not the preregistered one; not",
        '  leaderboard-comparable; Recognition is computed locally, no official judge.',
        "- 95% intervals on a row's overall score are about +/-2.7 points; per-task numbers (125 items) are descriptive.",
        '- The one post-hoc change: the mean-calls gate ceiling 4.0 -> 4.5 (a retry-counting defect; it changed no',
        '  accept/reject outcome). Other gates are as preregistered.',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
