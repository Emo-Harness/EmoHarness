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
                          verifier report, candidates and scores, gate reasons,
                          cost, accepted patch; the accepted patches as diffs
  metrics/                analyze.py's tables as written
  previous_runs           (inside summary.json) the same row scores recomputed
                          from results/archive/<run>/ for a v1-vs-v2 panel
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
RUN_TAG = 'v2_2026-09-11'
OUT = ROOT / 'results' / 'export' / f'emoharness_indist_{RUN_TAG}'
RESULTS = ROOT / 'results'
#: Earlier runs whose row scores are recomputed for a side-by-side panel.
PREVIOUS_RUNS = {
    'v1_omni_describer': {
        'dir': ROOT / 'results/archive/v1-omni-describer',
        'label': 'v1, 2026-09-08/09: with the Qwen2.5-Omni-7B audio describer, no verifier',
    },
}
ROWS = ('L0', 'L3', 'L4', 'L6')
#: Rows outside the ladder, measured for a named secondary question. Never a
#: Stage-2 result: the ladder's L4 is the first completed replicate (ADR-0006 §6).
SIDE_ROWS = {
    'L4_replicate2': (
        'Replicate 2 of 3 (not the declared run): the only Stage-2 bundle in this '
        'run with an accepted patch, scored on the same subset for reference'
    ),
}
ROW_LABELS = {
    'L0': 'Qwen3-VL-8B Direct (official prompt, native video)',
    'L3': 'Stage 1 scaffold (human-written harness, 4 calls)',
    'L4': 'Stage 2 harness (Luna Builder + Luna verifier, replicate 0)',
    'L6': 'GPT 5.6 Luna Direct (frames + audio text)',
}
DELTAS = (('L0', 'L3'), ('L3', 'L4'), ('L0', 'L4'), ('L0', 'L6'))
TASKS = [t.value for t in Task]
SEEN = {t.value for t in SEEN_TASKS}
RESAMPLES = 5000
PAID = ('builder', 'verifier', 'judge_proxy', 'teacher_direct')


def load_rows(
    results: Path = RESULTS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    examples = {
        e.example_id: e
        for e in read_manifest(ROOT / 'data/manifests/mme_emotion.jsonl')
    }
    rows: dict[str, dict[str, Any]] = {}
    for row in ROWS:
        rows[row] = {}
        with open(
            results / f'raw_predictions/{row}/subset.jsonl', encoding='utf-8'
        ) as fh:
            lines = fh.readlines()
        for line in lines:
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
    return {t: sum(v) / len(v) for t, v in per.items()}


def group_means(by_task: dict[str, float]) -> dict[str, float]:
    seen = [v for t, v in by_task.items() if t in SEEN]
    held = [v for t, v in by_task.items() if t not in SEEN]
    return {
        'seen': sum(seen) / len(seen),
        'held_out': sum(held) / len(held),
        'overall': sum(by_task.values()) / len(by_task),
    }


def _read_json(path: Path) -> Any:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _load(path: Path, default: Any = None) -> Any:
    return _read_json(path) if path.is_file() else default


def construction_summary(results: Path = RESULTS) -> list[dict[str, Any]]:
    out = []
    for rep_dir in sorted((results / 'construction/stage2').glob('[0-9]')):
        lr = _read_json(rep_dir / 'loop_result.json')
        records = {r['round_index']: r for r in lr['rounds']}
        rounds = []
        for rd in sorted(rep_dir.glob('round_*')):
            index = int(rd.name.split('_')[1])
            record = records.get(index, {})
            cand = _read_json(rd / 'candidates.json')
            acc = _read_json(rd / 'accepted.json')
            pkt = _read_json(rd / 'packet.json')
            verification = _load(rd / 'verification.json')
            gates = {_norm(g['name']): g for g in cand['gates']}
            evaluated = []
            for c in cand['evaluated']:
                s = c['scores']
                g = gates.get(_norm(c['name']))
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
            proposed = len(_load(rd / 'proposal.json', {}).get('candidates', []))
            rounds.append(
                {
                    'round': index,
                    'build_recognition_incumbent': pkt['build_recognition'],
                    'val_recognition_incumbent': record.get('incumbent_recognition'),
                    'proposed': proposed,
                    'rejected': cand.get('rejected', []),
                    'vetoed': cand.get('vetoed', []),
                    'verifier': (
                        {
                            'model_id': verification['model_id'],
                            'summary': verification['report'].get('summary', ''),
                            'verdicts': verification['report'].get('verdicts', []),
                        }
                        if verification
                        else None
                    ),
                    'candidates': evaluated,
                    'accepted': acc.get('accepted'),
                    'accepted_rationale': acc.get('rationale'),
                    'cost': record.get('cost') or _load(rd / 'cost.json'),
                }
            )
        first = lr['rounds'][0] if lr['rounds'] else {}
        out.append(
            {
                'replicate': int(rep_dir.name),
                'stopped_reason': lr['stopped_reason'],
                'rounds_run': len(lr['rounds']),
                'accepted_per_round': [r['accepted'] for r in lr['rounds']],
                'vetoed_per_round': [r.get('vetoed', []) for r in lr['rounds']],
                'val_incumbent_start': (
                    first.get('incumbent_recognition')
                    if first.get('incumbent_recognition') is not None
                    else _incumbent_from_gates(rounds)
                ),
                'val_recognition_final': _final_val(lr),
                'baseline_cost': lr.get('baseline_cost'),
                'total_cost': lr.get('total_cost'),
                'rounds': rounds,
                'final_manifest': _read_json(rep_dir / 'final/manifest.json'),
            }
        )
    return out


def _incumbent_from_gates(rounds: list[dict[str, Any]]) -> float | None:
    """v1 records carry no incumbent score; the first gate reason names it."""
    for rd in rounds:
        for c in rd['candidates']:
            for reason in c['gate_reasons']:
                if 'does not exceed incumbent' in reason:
                    return float(reason.split('incumbent ')[1].split(' ')[0])
    return None


def _final_val(loop_result: dict[str, Any]) -> float | None:
    """D_val Recognition of the last accepted candidate, else the starting incumbent."""
    value = None
    for record in loop_result['rounds']:
        if value is None:
            value = record.get('incumbent_recognition')
        if record.get('accepted'):
            for scores in record['candidates']:
                if scores['name'] == _norm(record['accepted']):
                    value = scores['recognition']
    return value


def _norm(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name).strip('.')


def accepted_patch_diff(replicate: int = 0) -> str:
    """Every file the replicate's final bundle changed against the scaffold, as one diff."""
    base_root = ROOT / 'harness'
    final_root = RESULTS / f'construction/stage2/{replicate}/final'
    if not final_root.is_dir():
        return f'(replicate {replicate} has no final bundle)\n'
    chunks = []
    for base in sorted(base_root.rglob('*')):
        if not base.is_file() or '__pycache__' in base.parts:
            continue
        rel = base.relative_to(base_root).as_posix()
        final = final_root / rel
        if not final.is_file() or rel == 'manifest.json':
            continue
        diff = ''.join(
            difflib.unified_diff(
                base.read_text().splitlines(True),
                final.read_text().splitlines(True),
                fromfile=f'harness/{rel} (Stage 1 scaffold)',
                tofile=f'stage2/{replicate}/final/{rel} (after the accepted patches)',
            )
        )
        if diff:
            chunks.append(diff)
    return ''.join(chunks) if chunks else '(no file differs: nothing was accepted)\n'


def cost_summary(results: Path = RESULTS) -> dict[str, Any]:
    by_role: dict[str, dict[str, float]] = defaultdict(
        lambda: {'calls': 0, 'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
    )
    for ledger in sorted(results.glob('ledger*.jsonl')):
        with open(ledger, encoding='utf-8') as fh:
            entries = [json.loads(line) for line in fh if line.strip()]
        for r in entries:
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


def row_scores_for(
    rows: dict[str, dict[str, Any]], ids: list[str], task_of: dict[str, str]
) -> dict[str, dict[str, Any]]:
    scores = {}
    for row in ROWS:
        rec = {i: rows[row][i]['recognition'] for i in ids}
        by_task = macro(rec, task_of)
        scores[row] = {
            'label': ROW_LABELS[row],
            'by_task': by_task,
            **group_means(by_task),
            'parse_failures': sum(rows[row][i]['parse_failures'] for i in ids),
            'mean_calls': sum(rows[row][i]['target_calls'] for i in ids) / len(ids),
        }
    return scores


def side_rows(ids: list[str], task_of: dict[str, str]) -> dict[str, Any]:
    """Named non-ladder rows with their delta against L3, when they were run."""
    _, base = load_rows()
    out = {}
    for row, label in SIDE_ROWS.items():
        path = RESULTS / f'raw_predictions/{row}/subset.jsonl'
        if not path.is_file():
            continue
        scores = {}
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        examples = {
            e.example_id: e
            for e in read_manifest(ROOT / 'data/manifests/mme_emotion.jsonl')
        }
        for line in lines:
            r = json.loads(line)
            e = examples[r['example_id']]
            ls = LabelSet.from_prompt(e.prompt)
            scores[r['example_id']] = ls.recognition(r['labels'], e.labels)
        shared = [i for i in ids if i in scores]
        by_task = macro({i: scores[i] for i in shared}, task_of)
        against = {i: base['L3'][i]['recognition'] for i in shared}
        res = paired_bootstrap(
            against, {i: scores[i] for i in shared}, task_of, RESAMPLES
        )
        out[row] = {
            'label': label,
            'by_task': by_task,
            **group_means(by_task),
            'vs_L3': {
                'estimate': res.estimate,
                'ci_low': res.ci_low,
                'ci_high': res.ci_high,
                'p_value': res.p_value,
                'items_better': sum(1 for i in shared if scores[i] > against[i]),
                'items_worse': sum(1 for i in shared if scores[i] < against[i]),
            },
        }
    return out


def previous_runs(ids: list[str], task_of: dict[str, str]) -> dict[str, Any]:
    """The same row scores, construction trajectory, and cost for each archived run."""
    out = {}
    for name, run in PREVIOUS_RUNS.items():
        if not (run['dir'] / 'raw_predictions').is_dir():
            continue
        _, rows = load_rows(run['dir'])
        same_ids = [i for i in ids if all(i in rows[r] for r in ROWS)]
        out[name] = {
            'label': run['label'],
            'rows': row_scores_for(rows, same_ids, task_of),
            'items': len(same_ids),
            'construction': construction_summary(run['dir']),
            'cost': cost_summary(run['dir']),
        }
    return out


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'harness_outputs').mkdir(parents=True)
    (OUT / 'construction').mkdir()
    (OUT / 'metrics').mkdir()
    examples, rows = load_rows()
    ids = sorted(rows['L0'])
    task_of = {i: examples[i].task.value for i in ids}
    row_scores = row_scores_for(rows, ids, task_of)

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

    leakage = _read_json(ROOT / 'data/checksums/leakage_report.json')
    subset = _read_json(ROOT / 'data/splits/subset_1000.json')
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
            'perception': 'deterministic tools only: ffmpeg frames (2 fps, 4-32), faster-whisper large-v3 transcript + language, openSMILE eGeMAPS prosody phrases; no audio-language model (v1 had a Qwen2.5-Omni-7B describer, removed 2026-09-11)',
            'builder': 'gpt-5.6-luna, reasoning_effort high, up to 4 candidates per round, 6 rounds max, early stop after 2 rounds without acceptance',
            'verifier': "gpt-5.6-luna (same model, separate prompt, reasoning_effort high) reviews each round's valid candidates as rationale + unified diff and vetoes overfitting, label forcing, contract breaks, evidence-discipline violations, out-of-scope edits, YAML damage; vetoed candidates are never evaluated",
            'cost_per_round': 'every round records calls, tokens, and dollars by role (builder, verifier, judge_proxy, target) in cost.json',
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
        'side_rows': side_rows(ids, task_of),
        'previous_runs': previous_runs(ids, task_of),
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
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    with open(OUT / 'items.jsonl', 'w', encoding='utf-8') as fh:
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
    for rep_dir in sorted((RESULTS / 'construction/stage2').glob('[0-9]')):
        dest = OUT / 'construction' / f'replicate_{rep_dir.name}'
        shutil.copytree(rep_dir, dest, ignore=shutil.ignore_patterns('candidates'))
        (
            OUT / 'construction' / f'accepted_patches_replicate{rep_dir.name}.diff'
        ).write_text(accepted_patch_diff(int(rep_dir.name)))
    shutil.copytree(ROOT / 'harness', OUT / 'construction' / 'stage1_scaffold')
    for f in (RESULTS / 'metrics').glob('*'):
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


def _side_row_lines(summary: dict[str, Any]) -> list[str]:
    """The headline sentence for each non-ladder row, if any were run."""
    lines = []
    for row, s in summary.get('side_rows', {}).items():
        d = s['vs_L3']
        lines += [
            f'{row}: {100 * s["overall"]:.1f} overall, {100 * d["estimate"]:+.1f} against L3'
            f' [{100 * d["ci_low"]:+.1f}, {100 * d["ci_high"]:+.1f}], p={d["p_value"]:.2f},'
            f' {d["items_better"]} items better / {d["items_worse"]} worse.',
            f'  {s["label"]}.',
            '',
        ]
    return lines


def _readme(summary: dict[str, Any]) -> str:
    rows = summary['rows']
    h = summary['headline']
    construction = summary['construction']
    accepted_total = sum(
        sum(1 for a in rep['accepted_per_round'] if a) for rep in construction
    )
    vetoed_total = sum(
        sum(len(v) for v in rep['vetoed_per_round']) for rep in construction
    )
    lines = [
        f'# EmoHarness in-distribution effect check, {RUN_TAG}',
        '',
        'Student Qwen3-VL-8B-Instruct, Builder and verifier GPT 5.6 Luna, MME-Emotion. Local',
        'Recognition on a 1,000-item subset (125 per task). Perception is deterministic tools only',
        '(no audio-language model). No benchmark media is included in this package.',
        '',
        '| Row | seen | held-out | overall |',
        '|---|---|---|---|',
    ]
    for row, s in rows.items():
        lines.append(
            f'| {row} {s["label"]} | {100 * s["seen"]:.1f} | {100 * s["held_out"]:.1f} | {100 * s["overall"]:.1f} |'
        )
    builder_delta = next(
        d for d in summary['deltas'] if (d['from'], d['to']) == ('L3', 'L4')
    )
    lines += [
        '',
        f'Harness gain L3 - L0: {100 * h["harness_gain_L3_minus_L0"]:+.1f}. Builder gain L4 - L3: {100 * h["builder_gain_L4_minus_L3"]:+.1f}'
        f' ({builder_delta["items_better"]} items better, {builder_delta["items_worse"]} worse of {summary["subset"]["size"]};'
        f' 95% CI [{100 * builder_delta["overall"]["ci_low"]:+.1f}, {100 * builder_delta["overall"]["ci_high"]:+.1f}]).',
        f'Teacher gap L6 - L0: {100 * h["teacher_gap_L6_minus_L0"]:+.1f}; the scaffold closes {100 * h["gap_closed_by_scaffold"]:.0f}% of it.',
        f'Construction: {len(construction)} replicates, {accepted_total} accepted patch(es), {vetoed_total} candidates vetoed by the verifier;',
        f'Luna spend ${summary["cost"]["luna_total_usd"]:.2f}.',
        '',
        *_side_row_lines(summary),
        '## Files',
        '',
        '- `summary.json`: everything above plus paired-bootstrap CIs for each delta (overall, seen, held-out, per task),',
        '  the Stage-2 construction trajectory of all three replicates (every candidate with its D_val score, regression',
        '  rate, reasoning score, and gate reasons), cost by role, leakage counts, protocol notes.',
        "- `items.jsonl`: one line per item with task, gold labels, label set, prompt, and each row's predicted labels,",
        '  Recognition, parse failures, call count, latency, and full response text (for before/after examples).',
        "- `harness_outputs/L3.jsonl`, `L4.jsonl`: the harness's per-item evidence record, hypotheses, verification,",
        '  five steps, and stage traces.',
        '- `construction/replicate_{0,1,2}/`: loop_result.json (rounds with incumbent score, vetoed names, cost by role),',
        '  per round packet.json/packet.txt (what the Builder saw), proposal.json (what it answered), review.txt and',
        '  verification.json (what the verifier saw and ruled), candidates.json (scores, gates, rejected, vetoed),',
        '  accepted.json, cost.json, final/ bundle.',
        '- `construction/accepted_patches_replicate{N}.diff`: every accepted patch of a replicate as a diff against the scaffold.',
        '- `summary.json` -> `previous_runs.v1_omni_describer`: the v1 rows, trajectory, and cost recomputed from the archive.',
        '- `summary.json` -> `side_rows`: non-ladder rows with their delta against L3 (here: replicate 2, the only',
        '  bundle in this run with an accepted patch). Label these clearly on the page; the Stage-2 number is L4.',
        '- `construction/stage1_scaffold/`: the human-written harness bundle every replicate started from.',
        '- `metrics/`: analyze.py tables (ladder.json/md, per_task.json, multilabel.json, cost.json for machine 1 only).',
        '- `leakage_report.json`, `subset_1000.json`, `docs/run-indist.md` (protocol, deviations, amendments, results).',
        '',
        '## Caveats to state on the page',
        '',
        "- In-distribution protocol (build/val from the benchmark's own seen tasks), not the preregistered one; not",
        '  leaderboard-comparable; Recognition is computed locally, no official judge.',
        "- 95% intervals on a row's overall score are about +/-2.7 points; per-task numbers (125 items) are descriptive.",
        '- The one post-hoc change (made during v1, kept): the mean-calls gate ceiling 4.0 -> 4.5 (a retry-counting',
        '  defect). Other gates are as preregistered.',
        '- v2 differs from v1 in the perception layer (no audio-language model) and in the verifier step, so the two',
        '  runs are compared on the same subset but not with identical inputs to the harness.',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
