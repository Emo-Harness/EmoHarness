"""Reading a finished run back: predictions, judge scores, and the macros over them.

One row on one item set is a :class:`RowData`. Its per-item Recognition comes
from the official judge when that row has been judged, and from local scoring
otherwise; every table records which, because the two are not interchangeable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from emoharness.data.manifest import Example
from emoharness.evaluation.schema import PredictionRecord, StepScores
from emoharness.evaluation.scoreboard import LOCAL, OFFICIAL, Scoreboard, Source
from emoharness.runner.context import RunContext
from emoharness.runner.items import FULL, SUBSET, by_id, examples_for
from emoharness.runner.judging import collect_step_scores, read_predictions
from emoharness.runner.paths import ProjectPaths

OUTPUTS_SUFFIX: Final = '.outputs.jsonl'
EXTERNAL_SLUG: Final = 'external_'


def unslug(name: str) -> str:
    """Turn a predictions file stem back into the item-set name it came from."""
    if name.startswith(EXTERNAL_SLUG):
        return 'external:' + name[len(EXTERNAL_SLUG) :]
    return name


def discover_rows(paths: ProjectPaths) -> tuple[tuple[str, str], ...]:
    """Every ``(row, item_set)`` pair that has raw predictions on disk."""
    found = [
        (path.parent.name, unslug(path.stem))
        for path in sorted(paths.raw_predictions.glob('*/*.jsonl'))
        if not path.name.endswith(OUTPUTS_SUFFIX)
    ]
    return tuple(found)


@dataclass(frozen=True)
class RowData:
    """One row's predictions, its judge scores if any, and its gold-side joins.

    Nothing is scored here: :meth:`scoreboard` decides who scored the row and
    hands back the one object every table is built from.
    """

    row: str
    item_set: str
    predictions: tuple[PredictionRecord, ...]
    step_scores: tuple[StepScores, ...]
    examples: Mapping[str, Example]

    @property
    def source(self) -> Source:
        """Where Recognition comes from: the official judge, or local scoring."""
        return OFFICIAL if self.step_scores else LOCAL

    def scoreboard(self) -> Scoreboard:
        """Per-item scores: the judge's when this row was judged, local otherwise."""
        if self.step_scores:
            return Scoreboard.from_step_scores(
                self.step_scores, self.examples, self.predictions
            )
        return Scoreboard.from_predictions(self.predictions, self.examples)

    def restricted(self, ids: frozenset[str]) -> RowData:
        """The same row seen through a smaller item set, e.g. a full run's subset."""
        return RowData(
            row=self.row,
            item_set=self.item_set,
            predictions=tuple(p for p in self.predictions if p.example_id in ids),
            step_scores=tuple(s for s in self.step_scores if s.example_id in ids),
            examples={k: v for k, v in self.examples.items() if k in ids},
        )


def load_row_data(row: str, item_set: str, ctx: RunContext) -> RowData:
    """Load one row's predictions and attach any judge output; scoring comes later."""
    return RowData(
        row=row,
        item_set=item_set,
        predictions=read_predictions(row, item_set, ctx.paths),
        step_scores=collect_step_scores(row, item_set, ctx),
        examples=by_id(examples_for(item_set, ctx.paths)),
    )


def load_all(
    ctx: RunContext, wanted: Iterable[tuple[str, str]] | None = None
) -> tuple[RowData, ...]:
    """Every row on disk, or only the requested ``(row, item_set)`` pairs."""
    pairs = tuple(wanted) if wanted is not None else discover_rows(ctx.paths)
    return tuple(load_row_data(row, item_set, ctx) for row, item_set in pairs)


class RowIndex:
    """Finds a row's scores for an item set, narrowing a full run when asked."""

    def __init__(self, rows: Iterable[RowData], subset_ids: frozenset[str]) -> None:
        self._rows = {(data.row, data.item_set): data for data in rows}
        self._subset_ids = subset_ids

    @property
    def rows(self) -> tuple[RowData, ...]:
        return tuple(self._rows.values())

    def get(self, row: str, item_set: str) -> RowData | None:
        """The row's data for ``item_set``, reading a subset off a full run if needed."""
        exact = self._rows.get((row, item_set))
        if exact is not None:
            return exact
        if item_set == SUBSET and self._subset_ids:
            full = self._rows.get((row, FULL))
            if full is not None:
                return full.restricted(self._subset_ids)
        return None
