"""The Scoreboard: one row's per-item scores, and every aggregate the paper reports.

A scoreboard is built once from whatever that row produced — raw predictions,
harness outputs, or the official judge's per-step ratings — and then answers
every question about it: the flat mean the Builder optimizes, the
per-task macros and their bootstrap intervals, the official
Rec/Rea/CoT aggregates, and the multi-label breakdown. Nothing downstream
re-derives a score from labels.

One empty-set rule, everywhere: an aggregate with no items behind it is None,
never 0.0. A zero means "scored, and wrong"; None means "nothing was scored".
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import COT_ALPHA, SUBSET_SEED
from emoharness.common.errors import ValidationError
from emoharness.common.logging import get_logger
from emoharness.common.stats import mean, mean_or_none
from emoharness.data.label_set import PRF, LabelSet
from emoharness.data.manifest import Example
from emoharness.data.tasks import TASK_GROUP, Group, Kind, Task
from emoharness.evaluation.bootstrap import BootstrapResult, paired_bootstrap
from emoharness.evaluation.schema import MethodScores, PredictionRecord, StepScores
from emoharness.harness.schema import HarnessOutput

Source = Literal['local', 'official']

LOCAL: Final[Source] = 'local'
OFFICIAL: Final[Source] = 'official'
#: The stratum an item without a task or a kind resamples inside (see `compare`).
POOLED: Final = 'pooled'

_LOGGER = get_logger(__name__)


class ItemScore(BaseModel):
    """What one item scored, whoever scored it.

    ``reasoning`` and ``num_steps`` are only filled by the official judge, and
    ``prf`` only for multi-label items, where the analysis plan asks for it.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str = Field(min_length=1)
    task: Task | None
    kind: Kind | None
    recognition: float = Field(ge=0.0, le=1.0)
    reasoning: float | None = Field(default=None, ge=0.0, le=1.0)
    num_steps: int | None = Field(default=None, ge=1)
    prf: PRF | None = None


class Scoreboard(BaseModel):
    """Every item's score for one row on one item set, plus the aggregates over them."""

    model_config = ConfigDict(frozen=True)

    items: Mapping[str, ItemScore]
    source: Source

    @classmethod
    def from_predictions(
        cls,
        predictions: Iterable[PredictionRecord],
        examples: Mapping[str, Example],
    ) -> Scoreboard:
        """Score raw predictions locally against their examples' gold labels."""
        scored = tuple(
            _local_score(_require(examples, p.example_id), p.labels)
            for p in predictions
        )
        return cls(items=_by_id(scored), source=LOCAL)

    @classmethod
    def from_outputs(
        cls, outputs: Iterable[HarnessOutput], examples: Mapping[str, Example]
    ) -> Scoreboard:
        """Score harness outputs locally, without a `PredictionRecord` detour."""
        scored = tuple(
            _local_score(_require(examples, o.example_id), o.labels) for o in outputs
        )
        return cls(items=_by_id(scored), source=LOCAL)

    @classmethod
    def from_step_scores(
        cls,
        step_scores: Iterable[StepScores],
        examples: Mapping[str, Example],
        predictions: Iterable[PredictionRecord] = (),
    ) -> Scoreboard:
        """Take Recognition and Reasoning from the official judge's per-step ratings.

        ``predictions`` is optional and only supplies the multi-label PRF, which
        the judge does not report; a judged item with no prediction still scores.
        """
        by_example = {p.example_id: p for p in predictions}
        scored = tuple(
            _official_score(
                score,
                examples.get(score.example_id),
                by_example.get(score.example_id),
            )
            for score in step_scores
        )
        return cls(items=_by_id(scored), source=OFFICIAL)

    @property
    def count(self) -> int:
        """How many items were scored."""
        return len(self.items)

    @property
    def ids(self) -> frozenset[str]:
        """The scored example ids."""
        return frozenset(self.items)

    def mean(self) -> float | None:
        """Flat mean Recognition over every item; the Builder's objective."""
        return mean_or_none(item.recognition for item in self.items.values())

    def by_task(self) -> Mapping[Task, float]:
        """Mean Recognition per benchmark task; items without a task are skipped."""
        groups: dict[Task, list[float]] = {}
        for item in self.items.values():
            if item.task is not None:
                groups.setdefault(item.task, []).append(item.recognition)
        return {task: mean(values) for task, values in groups.items()}

    def by_kind(self) -> Mapping[Kind, float]:
        """Mean Recognition per seen capability kind; items without a kind are skipped."""
        groups: dict[Kind, list[float]] = {}
        for item in self.items.values():
            if item.kind is not None:
                groups.setdefault(item.kind, []).append(item.recognition)
        return {kind: mean(values) for kind, values in groups.items()}

    def macro(self, group: Group | None = None) -> float | None:
        """Macro average of the per-task means, over one capability group or all tasks."""
        by_task = self.by_task()
        return mean_or_none(
            value
            for task, value in by_task.items()
            if group is None or TASK_GROUP.get(task) is group
        )

    def restrict(self, ids: Iterable[str]) -> Scoreboard:
        """The same scores seen through a smaller item set, e.g. a full run's subset."""
        wanted = frozenset(ids)
        kept = {i: s for i, s in self.items.items() if i in wanted}
        return Scoreboard(items=kept, source=self.source)

    def in_group(self, group: Group) -> Scoreboard:
        """Only the items whose task belongs to one capability group."""
        return self.restrict(
            i
            for i, s in self.items.items()
            if s.task is not None and TASK_GROUP.get(s.task) is group
        )

    def ci(self, resamples: int, seed: int = SUBSET_SEED) -> tuple[float, float] | None:
        """95% bootstrap interval for the macro, resampled within task."""
        scored = {i: s for i, s in self.items.items() if s.task is not None}
        if not scored:
            return None
        result = paired_bootstrap(
            dict.fromkeys(scored, 0.0),
            {i: s.recognition for i, s in scored.items()},
            {i: _stratum(s) for i, s in scored.items()},
            n_resamples=resamples,
            seed=seed,
            macro=True,
        )
        return result.ci_low, result.ci_high

    def compare(
        self, other: Scoreboard, resamples: int, seed: int = SUBSET_SEED
    ) -> BootstrapResult | None:
        """Paired bootstrap of ``other - self`` over the items both scored.

        Items are paired by id and resampled within their stratum — the task when
        there is one, else the capability kind, else one pooled stratum — and each
        resample's statistic is the macro over strata.
        """
        shared = tuple(i for i in self.items if i in other.items)
        if not shared:
            return None
        return paired_bootstrap(
            {i: self.items[i].recognition for i in shared},
            {i: other.items[i].recognition for i in shared},
            {i: _stratum(self.items[i]) for i in shared},
            n_resamples=resamples,
            seed=seed,
            macro=True,
        )

    def method_scores(
        self, row: str, item_set: str, alpha: float = COT_ALPHA
    ) -> MethodScores:
        """Rec/Rea/CoT/avg-steps exactly as the official `cal_metrics.py` computes them.

        Local scoreboards have no Reasoning component, so `reasoning`, `cot`, and
        `avg_steps` come back as 0.0 there; only an official scoreboard fills them.
        """
        scores = tuple(self.items.values())
        recognition = mean(item.recognition for item in scores)
        reasoning = mean(
            item.reasoning for item in scores if item.reasoning is not None
        )
        return MethodScores(
            row=row,
            item_set=item_set,
            count=len(scores),
            recognition=recognition,
            reasoning=reasoning,
            cot=alpha * recognition + (1 - alpha) * reasoning,
            avg_steps=mean(
                float(item.num_steps) for item in scores if item.num_steps is not None
            ),
        )

    def multilabel(self) -> Mapping[Task, PRF]:
        """Mean precision, recall, and F1 per task over the multi-label items.

        Only multi-label items carry a `prf`, so this is exactly the multi-label
        breakdown; a task with no such item is absent rather than zero.
        """
        groups: dict[Task, list[PRF]] = {}
        for item in self.items.values():
            if item.task is not None and item.prf is not None:
                groups.setdefault(item.task, []).append(item.prf)
        return {task: _mean_prf(values) for task, values in groups.items()}


def _local_score(example: Example, labels: tuple[str, ...]) -> ItemScore:
    """One item's local Recognition, and PRF when its prompt allows several labels."""
    label_set = LabelSet.from_prompt(example.prompt)
    multi_label = label_set.is_multi_label
    return ItemScore(
        example_id=example.example_id,
        task=example.task,
        kind=example.kind,
        recognition=label_set.recognition(labels, example.labels),
        prf=label_set.prf(labels, example.labels) if multi_label else None,
    )


def _official_score(
    score: StepScores, example: Example | None, prediction: PredictionRecord | None
) -> ItemScore:
    """One item's judged scores; the example supplies its task, kind, and gold labels."""
    if example is None:
        _LOGGER.warning(
            'no example for judged item %s; it joins no macro', score.example_id
        )
    return ItemScore(
        example_id=score.example_id,
        task=example.task if example is not None else None,
        kind=example.kind if example is not None else None,
        recognition=score.recognition,
        reasoning=score.reasoning,
        num_steps=score.num_steps,
        prf=_prf_for(example, prediction),
    )


def _prf_for(
    example: Example | None, prediction: PredictionRecord | None
) -> PRF | None:
    if example is None or prediction is None:
        return None
    label_set = LabelSet.from_prompt(example.prompt)
    if not label_set.is_multi_label:
        return None
    return label_set.prf(prediction.labels, example.labels)


def _require(examples: Mapping[str, Example], example_id: str) -> Example:
    example = examples.get(example_id)
    if example is None:
        raise ValidationError(f'no example for prediction {example_id}')
    return example


def _by_id(scores: Iterable[ItemScore]) -> Mapping[str, ItemScore]:
    return {score.example_id: score for score in scores}


def _stratum(item: ItemScore) -> Hashable:
    """The group an item resamples inside: its task, else its kind, else one pool."""
    if item.task is not None:
        return item.task
    if item.kind is not None:
        return item.kind
    return POOLED


def _mean_prf(values: Iterable[PRF]) -> PRF:
    scores = tuple(values)
    return PRF(
        precision=mean(item.precision for item in scores),
        recall=mean(item.recall for item in scores),
        f1=mean(item.f1 for item in scores),
    )
