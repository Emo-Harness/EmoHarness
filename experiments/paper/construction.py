"""MHE construction records: Table 3, Figures 6 and 7, and the numbers of Sections 5.1, 6 and App D.

Read from each run's ``loop_result.json`` (every generation: the incumbent's
validation Recognition, the candidates scored on D_v, the gate results and the
promotion; and the feedback returned to the Builder, whose ``status`` says
whether a proposal was screened out by the static checks or the reviewer and
whose ``build`` records its effect on the construction split), its
``run_config.json`` (the validation-set size) and its ledger (API spend and
student calls).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from emoharness.builder.loop import safe_name
from experiments.common import ORDER
from experiments.layout import Layout

#: Ledger roles billed to a construction run (App D: Builder, reviewer, proxy judge).
PAID_ROLES: Final = ('builder', 'verifier', 'judge_proxy')
STUDENT_ROLE: Final = 'target'
CONSTRUCTION_EXPERIMENT: Final = 'build_harness'
SCREENED: Final = ('rejected', 'vetoed')
#: Section 6: "candidates that gained at least eight validation items".
LARGE_GAIN_ITEMS: Final = 8


def load_run(layout: Layout, builder: str, student: str) -> dict[str, Any] | None:
    """One construction's loop result and run config; None if it has not run."""
    root = layout.construction_dir(builder, student)
    loop = root / 'loop_result.json'
    if not loop.is_file():
        return None
    config = root / 'run_config.json'
    return {
        'loop': json.loads(loop.read_text()),
        'config': json.loads(config.read_text()) if config.is_file() else {},
        'ledger': ledger_totals(layout.build_dir(builder, student)),
    }


def ledger_totals(build_dir: Path) -> dict[str, Any]:
    """API spend and student calls of the construction (not its later evaluation)."""
    usd = 0.0
    calls = tokens_in = tokens_out = 0
    for path in sorted(build_dir.rglob('ledger*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get('experiment', '')).startswith(CONSTRUCTION_EXPERIMENT):
                continue
            if row.get('role') in PAID_ROLES:
                usd += row.get('cost_usd') or 0.0
            elif row.get('role') == STUDENT_ROLE:
                calls += 1
                tokens_in += row.get('input_tokens') or 0
                tokens_out += row.get('output_tokens') or 0
    return {
        'usd': usd,
        'student_calls': calls,
        'student_input_tokens': tokens_in,
        'student_output_tokens': tokens_out,
    }


def _is(accepted: str | None, name: str) -> bool:
    """Does the promoted patch name (Builder's spelling) denote this candidate?"""
    return accepted is not None and name in (accepted, safe_name(accepted))


def summarize(run: Mapping[str, Any]) -> dict[str, Any]:
    """Table 3's row and Figure 6's trajectory for one construction."""
    loop = run['loop']
    rounds = loop['rounds']
    val_size = run['config'].get('val_size')
    start = rounds[0]['incumbent_recognition'] if rounds else None
    incumbent = start
    trajectory = []
    scored = screened = 0
    promotions: list[dict[str, Any]] = []
    candidates_out: list[dict[str, Any]] = []
    build_effects: list[int] = []
    for k, rnd in enumerate(rounds):
        before = incumbent
        accepted = rnd.get('accepted')
        reasons = {g['name']: g.get('reasons', []) for g in rnd.get('gate_results', [])}
        points = []
        for cand in rnd.get('candidates', []):
            scored += 1
            promoted = _is(accepted, cand['name'])
            gain = (
                cand['recognition'] - before
                if before is not None and cand.get('recognition') is not None
                else None
            )
            items = round(gain * val_size) if gain is not None and val_size else None
            record = {
                'generation': k + 1,
                'name': cand['name'],
                'recognition': cand['recognition'],
                'gain': gain,
                'gain_items': items,
                'regression_rate': cand.get('regression_rate'),
                'mean_calls': cand.get('mean_calls'),
                'reasoning': cand.get('reasoning'),
                'status': 'promoted' if promoted else 'rejected_by_gates',
                'gate_reasons': reasons.get(cand['name'], []),
            }
            points.append(record)
            candidates_out.append(record)
            if promoted:
                promotions.append(record)
                incumbent = cand['recognition']
        n_screened = 0
        for entry in rnd.get('feedback', []):
            if entry.get('status') in SCREENED:
                n_screened += 1
            build = entry.get('build')
            if entry.get('status') == 'evaluated' and build:
                build_effects.append(
                    int(build.get('improved_count', 0))
                    + int(build.get('regressed_count', 0))
                )
        screened += n_screened
        trajectory.append(
            {
                'generation': k + 1,
                'incumbent_before': before,
                'incumbent_after': incumbent,
                'candidates': points,
                'screened_out': n_screened,
            }
        )
    large = [
        c
        for c in candidates_out
        if c['gain_items'] is not None and c['gain_items'] >= LARGE_GAIN_ITEMS
    ]
    return {
        'generations': len(rounds),
        'stopped_reason': loop.get('stopped_reason'),
        'val_size': val_size,
        'build_size': run['config'].get('build_size'),
        'scored': scored,
        'screened_out': screened,
        'promotions': len(promotions),
        'promotion_gain_items': [p['gain_items'] for p in promotions],
        'dval_start': start,
        'dval_final': incumbent,
        'large_gain_candidates': len(large),
        'large_gain_blocked_by_regression': sum(
            any(str(r).startswith('regression rate') for r in c['gate_reasons'])
            for c in large
        ),
        'mean_build_items_changed': (
            sum(build_effects) / len(build_effects) if build_effects else None
        ),
        **run['ledger'],
        'trajectory': trajectory,
    }


def analyse(
    layout: Layout, builders: Sequence[str], students: Sequence[str] = ORDER
) -> dict[str, Any]:
    """Every construction record, per Builder and student, plus the totals the text quotes."""
    runs: dict[str, dict[str, Any]] = {}
    for b in builders:
        runs[b] = {}
        for st in students:
            run = load_run(layout, b, st)
            runs[b][st] = summarize(run) if run is not None else None
    return {'runs': runs, 'totals': totals(runs)}


def totals(
    runs: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
) -> dict[str, Any]:
    """Sections 5.1, 6 and App D: promotions, single-item promotions, gate blocks, cost."""
    done = [(b, st, r) for b, by_st in runs.items() for st, r in by_st.items() if r]

    def span(values: Iterable[float | None]) -> list[float] | None:
        kept = [v for v in values if v is not None]
        return [min(kept), max(kept)] if kept else None

    gains = [g for _, _, r in done for g in r['promotion_gain_items'] if g is not None]
    return {
        'by_builder': {
            b: {
                'runs': sum(1 for x in by_st.values() if x),
                'promotions': sum(x['promotions'] for x in by_st.values() if x),
                'screened_out': sum(x['screened_out'] for x in by_st.values() if x),
                'scored': sum(x['scored'] for x in by_st.values() if x),
                'usd': span(x['usd'] for x in by_st.values() if x),
                'max_validation_gain': max(
                    (
                        x['dval_final'] - x['dval_start']
                        for x in by_st.values()
                        if x and x['dval_start'] is not None
                    ),
                    default=None,
                ),
            }
            for b, by_st in runs.items()
        },
        'promotions': len(gains),
        'single_item_promotions': sum(1 for g in gains if g == 1),
        'large_gain_candidates': sum(r['large_gain_candidates'] for _, _, r in done),
        'large_gain_blocked_by_regression': sum(
            r['large_gain_blocked_by_regression'] for _, _, r in done
        ),
        'large_gain_by_student': {
            st: sum(r['large_gain_candidates'] for _, s, r in done if s == st)
            for st in {s for _, s, _ in done}
        },
        'mean_build_items_changed_by_student': {
            st: _mean([r['mean_build_items_changed'] for _, s, r in done if s == st])
            for st in {s for _, s, _ in done}
        },
        'student_calls': span(r['student_calls'] for _, _, r in done),
        'student_calls_total': sum(r['student_calls'] for _, _, r in done),
        'student_input_tokens_total': sum(
            r['student_input_tokens'] for _, _, r in done
        ),
        'student_output_tokens_total': sum(
            r['student_output_tokens'] for _, _, r in done
        ),
        'generations': sorted({r['generations'] for _, _, r in done}),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None
