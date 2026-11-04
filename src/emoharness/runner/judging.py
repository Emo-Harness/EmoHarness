"""Driving the official judge, and standing in for it in ``--fake`` runs.

The official scripts are executed as subprocesses, never reimplemented: this
module writes the per-task response files they read, builds their argument
lists, and parses their output back into ``StepScores``. The judge's own
evidence directory is only ever named as a command-line argument; nothing here
opens it, and a test in ``tests/data_integrity`` asserts that.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from emoharness.common.constants import JUDGE_STEP_MODEL, STEP_PREFIX
from emoharness.common.errors import ValidationError
from emoharness.common.io import iter_jsonl, read_json, write_json_atomic
from emoharness.common.ledger import LedgerEntry, record
from emoharness.common.logging import get_logger
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.data.tasks import Task
from emoharness.evaluation.judge import StepScorer, judge_input_for
from emoharness.evaluation.official import (
    OfficialJudgeCommand,
    judge_cost_estimate,
    run_official_pipeline,
    score_string,
    step_scores_from_eval_json,
    to_response_rows,
    write_response_json,
)
from emoharness.evaluation.schema import PredictionRecord, StepScores
from emoharness.runner.context import RunContext
from emoharness.runner.items import by_id, examples_for
from emoharness.runner.paths import ProjectPaths, guard_write

RESPONSES_FILE: Final = 'responses.json'
INDEX_FILE: Final = 'index.json'
STEPS_FILE: Final = 'steps.json'
EVAL_FILE: Final = 'eval.json'
METRICS_FILE: Final = 'metrics.txt'
Runner = Callable[[Sequence[str]], int]
#: Per item: labels right, labels asked for, steps the response carried.
ScoreSpec = tuple[int, int, int]

_LOGGER = get_logger(__name__)


def read_predictions(
    row: str, item_set: str, paths: ProjectPaths
) -> tuple[PredictionRecord, ...]:
    """Raw predictions for one row on one item set."""
    path = paths.predictions(row, item_set)
    if not path.is_file():
        raise ValidationError(f'no predictions at {path}; run the row first')
    return tuple(PredictionRecord.model_validate(obj) for obj in iter_jsonl(path))


def group_by_task(
    predictions: Iterable[PredictionRecord],
) -> Mapping[Task, tuple[PredictionRecord, ...]]:
    """Predictions grouped by benchmark task; items without a task are dropped."""
    groups: dict[Task, list[PredictionRecord]] = {}
    for prediction in predictions:
        if prediction.task is not None:
            groups.setdefault(prediction.task, []).append(prediction)
    return {task: tuple(rows) for task, rows in sorted(groups.items())}


def task_dir(row: str, item_set: str, task: Task, paths: ProjectPaths) -> Path:
    """Where one task's judge inputs and outputs live."""
    return paths.judge_run(row, item_set) / task.value


def prepare_official_inputs(
    row: str, item_set: str, ctx: RunContext
) -> Mapping[Task, Path]:
    """Write the per-task response files the official step extractor reads.

    Alongside each one an ``index.json`` records which example id every official
    ``video_id`` came from, so the judge's output can be joined back to the
    manifest without re-deriving the id.
    """
    predictions = read_predictions(row, item_set, ctx.paths)
    lookup = by_id(examples_for(item_set, ctx.paths))
    written: dict[Task, Path] = {}
    for task, rows in group_by_task(predictions).items():
        directory = task_dir(row, item_set, task, ctx.paths)
        response_rows = to_response_rows(rows, lookup)
        path = guard_write(directory / RESPONSES_FILE, ctx.force)
        write_response_json(path, response_rows)
        index = {
            str(response['video_id']): prediction.example_id
            for response, prediction in zip(response_rows, rows, strict=True)
        }
        write_json_atomic(guard_write(directory / INDEX_FILE, ctx.force), index)
        written[task] = path
    if not written:
        raise ValidationError(f'{row}/{item_set}: no benchmark items to judge')
    return written


def official_commands(
    row: str,
    item_set: str,
    ctx: RunContext,
    repo_path: Path,
    audio_json_dir: Path | None,
    video_dir: Path,
) -> tuple[OfficialJudgeCommand, ...]:
    """One command triple per task, ready to run against a MME-Emotion checkout.

    ``audio_json_dir`` is the judge's own evidence directory. Its per-task file
    name is forwarded as an argument and never opened here.
    """
    predictions = read_predictions(row, item_set, ctx.paths)
    return tuple(
        OfficialJudgeCommand(
            repo_path=str(repo_path),
            task=task,
            response_json=str(
                task_dir(row, item_set, task, ctx.paths) / RESPONSES_FILE
            ),
            steps_json=str(task_dir(row, item_set, task, ctx.paths) / STEPS_FILE),
            eval_json=str(task_dir(row, item_set, task, ctx.paths) / EVAL_FILE),
            metrics_txt=str(task_dir(row, item_set, task, ctx.paths) / METRICS_FILE),
            video_dir=str(video_dir),
            audio_json=str(audio_json_dir / f'{task.value}.json')
            if audio_json_dir is not None
            else None,
            model_name=row,
        )
        for task in group_by_task(predictions)
    )


def _record_judge_cost(ctx: RunContext, task: Task, n_items: int) -> None:
    record(
        ctx.ledger,
        LedgerEntry(
            role='judge_official',
            model_id=JUDGE_STEP_MODEL,
            experiment=f'{ctx.experiment}:{task.value}',
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            cost_usd=judge_cost_estimate(n_items),
        ),
    )


def run_official(
    commands: Sequence[OfficialJudgeCommand],
    ctx: RunContext,
    runner: Runner = subprocess.call,
) -> Mapping[str, tuple[int, ...]]:
    """Execute each task's three official scripts in order, recording the cost."""
    results: dict[str, tuple[int, ...]] = {}
    for command in commands:
        rows = read_json(Path(command.response_json))
        _record_judge_cost(ctx, command.task, len(rows))
        codes = run_official_pipeline(command, runner)
        results[command.task.value] = codes
        if any(code != 0 for code in codes):
            _LOGGER.error('official judge failed for %s: %s', command.task.value, codes)
    return results


def count_steps(response_text: str) -> int:
    """How many numbered steps a response carries, as the judge would count them."""
    prefix = f'{STEP_PREFIX} '
    return max(
        1, sum(1 for line in response_text.splitlines() if line.startswith(prefix))
    )


def score_specs(
    predictions: Iterable[PredictionRecord], examples: Mapping[str, Example]
) -> Mapping[str, ScoreSpec]:
    """What a stand-in judge needs per item; a real judge reads the frames instead.

    Predictions whose example is not in ``examples`` are skipped, so a caller may
    pass a superset of the items it means to score.
    """
    specs: dict[str, ScoreSpec] = {}
    for prediction in predictions:
        example = examples.get(prediction.example_id)
        if example is None:
            continue
        label_set = LabelSet.from_prompt(example.prompt)
        specs[prediction.example_id] = (
            label_set.overlap(prediction.labels, example.labels),
            len(example.labels),
            count_steps(prediction.response_text),
        )
    return specs


def _score_pairs(scores: StepScores, total: int) -> tuple[tuple[int, int], ...]:
    """The judge's own C/K pairs: reasoning steps out of one, the last out of K."""
    head = tuple((round(value), 1) for value in scores.scores[:-1])
    return (*head, (round(scores.recognition * total), total))


def _eval_row(
    scorer: StepScorer, prediction: PredictionRecord, example: Example, video_id: str
) -> Any:
    """One eval-json row: the official video id and a ``<score>`` block for it."""
    scores = scorer.score(
        judge_input_for(
            prediction.example_id, (), example.labels, prediction.response_text
        )
    )
    return {
        'video_id': video_id,
        'score': score_string(_score_pairs(scores, len(example.labels))),
    }


def fake_judge(row: str, item_set: str, ctx: RunContext) -> Mapping[Task, Path]:
    """Write eval-json-shaped scores so ``--fake`` runs reach the analysis stage.

    The scores come from the context's step scorer, so the stand-in judge and the
    proxy judge answer from one table rather than two spellings of it. The
    video-id index is written too when it is missing, so this works whether or
    not :func:`prepare_official_inputs` ran first.
    """
    predictions = read_predictions(row, item_set, ctx.paths)
    lookup = by_id(examples_for(item_set, ctx.paths))
    scorer = ctx.step_scorer(ctx.step_score_table(score_specs(predictions, lookup)))
    written: dict[Task, Path] = {}
    for task, rows in group_by_task(predictions).items():
        directory = task_dir(row, item_set, task, ctx.paths)
        pairs = tuple(zip(to_response_rows(rows, lookup), rows, strict=True))
        payload = [
            _eval_row(
                scorer, prediction, lookup[prediction.example_id], str(sent['video_id'])
            )
            for sent, prediction in pairs
        ]
        index_path = directory / INDEX_FILE
        if not index_path.is_file():
            write_json_atomic(
                index_path,
                {str(s['video_id']): p.example_id for s, p in pairs},
            )
        path = guard_write(directory / EVAL_FILE, ctx.force)
        write_json_atomic(path, payload)
        _record_judge_cost(ctx, task, len(payload))
        written[task] = path
    return written


def collect_step_scores(
    row: str, item_set: str, ctx: RunContext
) -> tuple[StepScores, ...]:
    """Read every task's judge output back, keyed by example id rather than video id."""
    run_dir = ctx.paths.judge_run(row, item_set)
    collected: list[StepScores] = []
    for task in Task:
        directory = run_dir / task.value
        eval_json, index_json = directory / EVAL_FILE, directory / INDEX_FILE
        if not eval_json.is_file():
            continue
        index = read_json(index_json) if index_json.is_file() else {}
        collected.extend(
            score.model_copy(
                update={'example_id': index.get(score.example_id, score.example_id)}
            )
            for score in step_scores_from_eval_json(eval_json)
        )
    return tuple(collected)
