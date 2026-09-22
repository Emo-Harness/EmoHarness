"""Wrappers around the official MME-Emotion scoring scripts (docs/benchmark-facts.md).

This module is the only place in the harness allowed to reference the official
`audio_clue/*.json` files, and only by forwarding a path through `--audio_json`
to `eval_cot_gpt4o.py`; nothing here opens or parses that file's content.

Script CLI shapes and the GPT-4o/GPT-4.1 prompts were confirmed by reading the
official `QwenAudio/MME-Emotion` repository (MIT license) directly: each task
has its own `extract_step/<dir>/code/extract_step.py`,
`eval_cot/<dir>/code/eval_cot_gpt4o.py`, and `eval_cot/<dir>/metrics/cal_metrics.py`,
where `<dir>` is the task name with underscores replaced by hyphens (e.g.
`Task.ML_ER` -> `ML-ER`). The official annotation `video_id` is the task's
underscore-style name joined with the file's basename (e.g. `ER_Wild/xxx.avi`).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import (
    COT_ALPHA,
    JUDGE_STEP_MODEL,
    LABEL_SEPARATOR,
    STEP_PREFIX,
)
from emoharness.common.errors import ValidationError
from emoharness.common.io import read_json, write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.data.manifest import Example
from emoharness.data.tasks import Task
from emoharness.evaluation.schema import PredictionRecord, StepScores

_logger = get_logger(__name__)

_SCORE_BLOCK_RE = re.compile(r'<score>(.*?)</score>', re.DOTALL)
_SCORE_PAIR_RE = re.compile(r': (\d+)/(\d+)')


def to_response_rows(
    predictions: Iterable[PredictionRecord], examples: Mapping[str, Example]
) -> tuple[Mapping[str, str], ...]:
    """Rows in the shape `extract_step.py` reads: video_id, ground_truth, model_response."""
    rows: list[Mapping[str, str]] = []
    for prediction in predictions:
        example = examples.get(prediction.example_id)
        if example is None:
            raise ValidationError(f'no example for prediction {prediction.example_id}')
        rows.append(
            {
                'video_id': _video_id(example),
                'ground_truth': ','.join(example.labels),
                'model_response': prediction.response_text,
            }
        )
    return tuple(rows)


def _video_id(example: Example) -> str:
    """The official `video_id` string: `metadata['video_id']`, else path-from-task-root."""
    metadata_value = example.metadata.get('video_id')
    if metadata_value is not None:
        return str(metadata_value)
    path = PurePosixPath(example.media_path)
    if example.task is not None and example.task.value in path.parts:
        index = path.parts.index(example.task.value)
        return str(PurePosixPath(*path.parts[index:]))
    return path.name


def write_response_json(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Write `to_response_rows` output as the JSON array `extract_step.py` expects."""
    write_json_atomic(path, list(rows))


def parse_score_string(text: str) -> tuple[float, ...] | None:
    """Parse an official `<score>...</score>` block; None when no block is found.

    Replicates the official `cal_metrics.py` regex, which finds every `: N/D`
    pair inside the block in order; a `0` denominator (never seen in practice,
    but not excluded by the judge prompt) scores 0.0 rather than raising.
    """
    block = _SCORE_BLOCK_RE.search(text)
    if block is None:
        return None
    pairs = _SCORE_PAIR_RE.findall(block.group(1))
    if not pairs:
        return None
    return tuple(
        (int(numerator) / int(denominator)) if int(denominator) != 0 else 0.0
        for numerator, denominator in pairs
    )


def score_string(pairs: Iterable[tuple[int, int]]) -> str:
    """One item's `<score>...</score>` block, in the shape `cal_metrics.py` reads.

    Each pair is one step's `C/K`: every step but the last is rated out of one,
    and the last carries the official C/K. `parse_score_string` reads back what
    this writes, so a block built here round-trips through the official regex.
    """
    body = LABEL_SEPARATOR.join(
        f'{STEP_PREFIX} {number}: {correct}/{max(1, total)}'
        for number, (correct, total) in enumerate(pairs, start=1)
    )
    return f'<score>{body}</score>'


def step_scores_from_eval_json(path: Path) -> tuple[StepScores, ...]:
    """Read `eval_cot_gpt4o.py`'s output file into `StepScores`, keyed by `video_id`.

    Rows whose `score` field has no parseable `<score>` block are skipped and
    logged, matching the official `cal_metrics.py` (`if not scores: continue`).
    """
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValidationError(f'{path}: expected a JSON array of eval rows')
    results: list[StepScores] = []
    for row in rows:
        video_id = row.get('video_id')
        scores = parse_score_string(str(row.get('score', '')))
        if video_id is None or scores is None:
            _logger.warning('skipping unparsable eval row for video_id=%s', video_id)
            continue
        results.append(StepScores(example_id=str(video_id), scores=scores))
    return tuple(results)


def judge_cost_estimate(
    n_items: int,
    tokens_per_item: float = 9_000.0,
    price_per_million: float = 3.0,
) -> float:
    """Rough judge cost in USD.

    Defaults are the midpoint of docs/benchmark-facts.md's estimate (about
    6-12k input tokens and $0.02-0.035 per item); replace both with
    ledger-measured values from the first 100 items before trusting a full-run
    estimate (ADR-0009's cost model and cut line).
    """
    if n_items < 0:
        raise ValidationError('judge_cost_estimate requires a non-negative item count')
    return n_items * tokens_per_item / 1_000_000 * price_per_million


def _script_dir(task: Task) -> str:
    """Official repo folder name for a task, e.g. `Task.ML_ER` ('ML_ER') -> 'ML-ER'."""
    return task.value.replace('_', '-')


class OfficialJudgeCommand(BaseModel):
    """Argument lists for the three official scripts, given a repo checkout and file paths.

    This is the only place allowed to reference the official `audio_clue/`
    files, and only by forwarding `audio_json` to `eval_cot_gpt4o.py`'s
    `--audio_json` flag; the content is never opened here. Credentials are
    deliberately not part of this model or the commands it builds: the runner
    (or its environment) is responsible for the official scripts' own
    `--api_key`/`--base_url` flags, so no secret is ever embedded in an object
    that might be logged or serialized for reproducibility records.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', protected_namespaces=())

    repo_path: str = Field(
        min_length=1, description='checkout of QwenAudio/MME-Emotion'
    )
    task: Task
    response_json: str = Field(min_length=1, description='to_response_rows() output')
    steps_json: str = Field(min_length=1, description='extract_step.py output')
    eval_json: str = Field(min_length=1, description='eval_cot_gpt4o.py output')
    metrics_txt: str = Field(min_length=1, description='cal_metrics.py output')
    video_dir: str = Field(
        min_length=1,
        description='directory holding this task videos, named by video_id',
    )
    audio_json: str | None = Field(
        default=None, description='official audio_clue/*.json path, forwarded only'
    )
    model_name: str = Field(
        default='EmoHarness', description="cal_metrics.py's --model_name"
    )

    def extract_step_command(self) -> tuple[str, ...]:
        script = f'{self.repo_path}/extract_step/{_script_dir(self.task)}/code/extract_step.py'
        return (
            'python',
            script,
            '--input_json',
            self.response_json,
            '--output_json',
            self.steps_json,
            '--model',
            JUDGE_STEP_MODEL,
        )

    def eval_cot_command(self) -> tuple[str, ...]:
        script = (
            f'{self.repo_path}/eval_cot/{_script_dir(self.task)}/code/eval_cot_gpt4o.py'
        )
        args = [
            'python',
            script,
            '--response_json',
            self.steps_json,
            '--output_json',
            self.eval_json,
            '--video_dir',
            self.video_dir,
        ]
        if self.audio_json is not None:
            args += ['--audio_json', self.audio_json]
        return tuple(args)

    def cal_metrics_command(self) -> tuple[str, ...]:
        script = (
            f'{self.repo_path}/eval_cot/{_script_dir(self.task)}/metrics/cal_metrics.py'
        )
        return (
            'python',
            script,
            '--input_json',
            self.eval_json,
            '--output_txt',
            self.metrics_txt,
            '--model_name',
            self.model_name,
            '--alpha',
            str(COT_ALPHA),
        )

    def commands(self) -> tuple[tuple[str, ...], ...]:
        """The three commands in run order."""
        return (
            self.extract_step_command(),
            self.eval_cot_command(),
            self.cal_metrics_command(),
        )


def run_official_pipeline(
    cmd: OfficialJudgeCommand,
    runner: Callable[[Sequence[str]], int] = subprocess.call,
) -> tuple[int, ...]:
    """Run the three official scripts in order via `runner`; stop at the first failure.

    `runner` defaults to `subprocess.call` but tests should inject a fake that
    records its calls instead of touching the network or a real checkout.
    """
    codes: list[int] = []
    for command in cmd.commands():
        code = runner(command)
        codes.append(code)
        if code != 0:
            break
    return tuple(codes)
