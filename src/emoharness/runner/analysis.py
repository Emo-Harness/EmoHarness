"""Every table in the paper, generated from raw predictions by one function.

``analyze`` writes ``ladder.json`` and ``ladder.md``, ``per_task.json``,
``hypotheses.json``, ``multilabel.json``, and ``cost.json`` under
``results/metrics/``. Nothing is computed by hand and nothing is smoothed: a row
that has not been judged officially is scored locally and every table says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.errors import ValidationError
from emoharness.common.io import write_json_atomic
from emoharness.common.ledger import read_entries, summarize, summarize_by
from emoharness.common.logging import get_logger
from emoharness.common.stats import mean
from emoharness.data.splits import read_subset
from emoharness.data.tasks import Group, Task
from emoharness.evaluation.scoreboard import OFFICIAL, Scoreboard
from emoharness.runner.context import RunContext
from emoharness.runner.hypotheses import hypotheses_report
from emoharness.runner.paths import guard_write
from emoharness.runner.results import RowData, RowIndex, load_all

LADDER_JSON: Final = 'ladder.json'
LADDER_MD: Final = 'ladder.md'
PER_TASK_JSON: Final = 'per_task.json'
HYPOTHESES_JSON: Final = 'hypotheses.json'
MULTILABEL_JSON: Final = 'multilabel.json'
COST_JSON: Final = 'cost.json'
PER_THOUSAND: Final = 1000.0
LOCAL_NOTE: Final = (
    'Rows whose source is "local" have no official judge output: their Recognition '
    'is computed locally from the emitted labels and their Rea-S, CoT-S, and average '
    'step count are unavailable.'
)
COST_NOTE: Final = (
    'USD / 1,000 is the nominal per-item price of every Target call at list prices, '
    'ignoring cache hits; cost.json reports what was actually billed, which is lower '
    'when rows share cached calls.'
)

_LOGGER = get_logger(__name__)


class LadderEntry(BaseModel):
    """One row of the main table (ADR-0009): scores, budget, and where they came from."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    row: str
    item_set: str
    source: str
    count: int = Field(ge=0)
    seen_macro: float | None = None
    held_out_macro: float | None = None
    overall_macro: float | None = None
    rec_ci_low: float | None = None
    rec_ci_high: float | None = None
    reasoning: float | None = None
    cot: float | None = None
    avg_steps: float | None = None
    mean_calls: float = Field(ge=0.0)
    cost_per_1000_usd: float = Field(ge=0.0)
    parse_failure_rate: float = Field(ge=0.0)

    @property
    def key(self) -> str:
        return f'{self.row}@{self.item_set}'


def ladder_entry(data: RowData, resamples: int) -> LadderEntry:
    """Turn one row's predictions and judge output into a main-table row."""
    board = data.scoreboard()
    interval = board.ci(resamples)
    judged = data.source == OFFICIAL
    scores = board.method_scores(data.row, data.item_set)
    return LadderEntry(
        row=data.row,
        item_set=data.item_set,
        source=data.source,
        count=len(data.predictions),
        seen_macro=board.macro(Group.SEEN),
        held_out_macro=board.macro(Group.HELD_OUT),
        overall_macro=board.macro(),
        rec_ci_low=interval[0] if interval else None,
        rec_ci_high=interval[1] if interval else None,
        reasoning=scores.reasoning if judged else None,
        cot=scores.cot if judged else None,
        avg_steps=scores.avg_steps if judged else None,
        mean_calls=mean(float(p.target_calls) for p in data.predictions),
        cost_per_1000_usd=mean(p.cost_usd for p in data.predictions) * PER_THOUSAND,
        parse_failure_rate=mean(float(p.parse_failures) for p in data.predictions),
    )


def _cell(value: float | None, digits: int = 3) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def _ci_cell(entry: LadderEntry) -> str:
    if entry.rec_ci_low is None or entry.rec_ci_high is None:
        return '-'
    return f'[{entry.rec_ci_low:.3f}, {entry.rec_ci_high:.3f}]'


def ladder_markdown(entries: Sequence[LadderEntry]) -> str:
    """The main table as markdown, in the order the rows were analyzed."""
    header = (
        '| Row | Items | Source | N | Seen | Held-out | Overall | Rec-S 95% CI | '
        'Rea-S | CoT-S | Avg steps | Mean calls | USD / 1,000 |'
    )
    divider = '| --- ' * 13 + '|'
    lines = [
        '| '
        + ' | '.join(
            (
                entry.row,
                entry.item_set,
                entry.source,
                str(entry.count),
                _cell(entry.seen_macro),
                _cell(entry.held_out_macro),
                _cell(entry.overall_macro),
                _ci_cell(entry),
                _cell(entry.reasoning),
                _cell(entry.cot),
                _cell(entry.avg_steps, 2),
                _cell(entry.mean_calls, 2),
                _cell(entry.cost_per_1000_usd, 2),
            )
        )
        + ' |'
        for entry in entries
    ]
    return '\n'.join(
        ['# Ladder', '', header, divider, *lines, '', LOCAL_NOTE, '', COST_NOTE, '']
    )


def per_task_report(rows: Sequence[RowData]) -> Mapping[str, Any]:
    """Per-task Recognition and item count for every row (ADR-0013 §8)."""
    report: dict[str, Any] = {}
    for data in rows:
        board = data.scoreboard()
        counts = _task_counts(board)
        report[f'{data.row}@{data.item_set}'] = {
            'source': data.source,
            'tasks': {
                task.value: {'recognition': value, 'count': counts.get(task, 0)}
                for task, value in sorted(board.by_task().items())
            },
        }
    return report


def _task_counts(board: Scoreboard) -> Mapping[Task, int]:
    counts: dict[Task, int] = {}
    for item in board.items.values():
        if item.task is not None:
            counts[item.task] = counts.get(item.task, 0) + 1
    return counts


def multilabel_report(rows: Sequence[RowData]) -> Mapping[str, Any]:
    """Precision, recall, and F1 on the multi-label tasks (ADR-0013 §7)."""
    report: dict[str, Any] = {}
    for data in rows:
        board = data.scoreboard()
        counts = _task_counts(board)
        per_task = {
            task.value: {
                'precision': prf.precision,
                'recall': prf.recall,
                'f1': prf.f1,
                'count': counts.get(task, 0),
            }
            for task, prf in sorted(board.multilabel().items())
        }
        if per_task:
            report[f'{data.row}@{data.item_set}'] = per_task
    return report


def cost_report(ctx: RunContext, entries: Sequence[LadderEntry]) -> Mapping[str, Any]:
    """The ledger, grouped every way the budget table needs (ADR-0013 §10)."""
    ledger = read_entries(ctx.ledger)
    grouped = {
        f'by_{key}': {
            name: summary.model_dump(mode='json')
            for name, summary in summarize_by(ledger, key).items()
        }
        for key in ('role', 'model_id', 'experiment')
    }
    return {
        'total': summarize(ledger).model_dump(mode='json'),
        **grouped,
        'per_row': {
            entry.key: {
                'count': entry.count,
                'mean_calls': entry.mean_calls,
                'cost_per_1000_usd': entry.cost_per_1000_usd,
                'total_cost_usd': entry.cost_per_1000_usd * entry.count / PER_THOUSAND,
            }
            for entry in entries
        },
    }


def _subset_ids(ctx: RunContext) -> frozenset[str]:
    if not ctx.paths.subset.is_file():
        return frozenset()
    return frozenset(read_subset(ctx.paths.subset).ids)


def analyze(
    ctx: RunContext, wanted: Sequence[tuple[str, str]] | None = None
) -> Mapping[str, Path]:
    """Build every table from the artifacts on disk and write them out."""
    rows = load_all(ctx, wanted)
    if not rows:
        raise ValidationError('no raw predictions found; run a ladder row first')
    entries = tuple(ladder_entry(data, ctx.resamples) for data in rows)
    index = RowIndex(rows, _subset_ids(ctx))
    _LOGGER.info('analyzing %d row/item-set pairs', len(rows))
    documents: Mapping[str, Any] = {
        LADDER_JSON: {
            'note': LOCAL_NOTE,
            'cost_note': COST_NOTE,
            'rows': [entry.model_dump(mode='json') for entry in entries],
        },
        PER_TASK_JSON: per_task_report(rows),
        HYPOTHESES_JSON: hypotheses_report(index, ctx.resamples),
        MULTILABEL_JSON: multilabel_report(rows),
        COST_JSON: cost_report(ctx, entries),
    }
    written = {
        name: _write_json(ctx, name, document) for name, document in documents.items()
    }
    markdown = guard_write(ctx.paths.metrics / LADDER_MD, ctx.force)
    markdown.write_text(ladder_markdown(entries), encoding='utf-8')
    return {**written, LADDER_MD: markdown}


def _write_json(ctx: RunContext, name: str, document: Any) -> Path:
    path = guard_write(ctx.paths.metrics / name, ctx.force)
    write_json_atomic(path, document)
    return path
