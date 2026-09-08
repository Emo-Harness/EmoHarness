from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from emoharness.common.constants import COT_ALPHA, JUDGE_STEP_MODEL
from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.data.tasks import Kind, Split, Task
from emoharness.evaluation.official import (
    OfficialJudgeCommand,
    judge_cost_estimate,
    parse_score_string,
    run_official_pipeline,
    step_scores_from_eval_json,
    to_response_rows,
    write_response_json,
)
from emoharness.evaluation.schema import PredictionRecord


def test_to_response_rows_uses_metadata_video_id_when_present(tmp_path: Path) -> None:
    media = tmp_path / 'clip.mp4'
    media.write_bytes(b'\x00')
    example = Example(
        example_id='ex1',
        source_id='s1',
        origin='mme_emotion',
        task=Task.ER_WILD,
        split=Split.TEST,
        media_path=str(media),
        prompt='As an emotion recognition expert, identify the most prominent emotion '
        'displayed in the video. Candidate labels: anger, sadness. Please think step '
        'by step. Enclose your final answer within <answer></answer> tags.',
        label_set=('anger', 'sadness'),
        labels=('anger', 'disgust'),
        metadata={'video_id': 'ER_Wild/test_Anger_0001.avi'},
    )
    prediction = PredictionRecord(
        example_id='ex1',
        row='L0',
        task=Task.ER_WILD,
        labels=('anger',),
        response_text='<answer>anger</answer>',
        target_calls=1,
    )
    rows = to_response_rows((prediction,), {'ex1': example})
    assert rows == (
        {
            'video_id': 'ER_Wild/test_Anger_0001.avi',
            'ground_truth': 'anger,disgust',
            'model_response': '<answer>anger</answer>',
        },
    )


def test_to_response_rows_falls_back_to_media_path_relative_to_task_root(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / 'data' / 'raw' / 'mme_emotion' / 'ML_ER'
    media_dir.mkdir(parents=True)
    media = media_dir / '03419.mp4'
    media.write_bytes(b'\x00')
    example = Example(
        example_id='ex2',
        source_id='s2',
        origin='mme_emotion',
        task=Task.ML_ER,
        split=Split.TEST,
        media_path=str(media),
        prompt='identify multiple emotions '
        'displayed in the video. Candidate labels: anger, disgust. Please think '
        'step by step. Enclose your final answer within <answer></answer> tags.',
        label_set=('anger', 'disgust'),
        labels=('anger', 'disgust'),
    )
    prediction = PredictionRecord(
        example_id='ex2', row='L0', labels=('anger',), response_text='x', target_calls=1
    )
    rows = to_response_rows((prediction,), {'ex2': example})
    assert rows[0]['video_id'] == 'ML_ER/03419.mp4'


def test_to_response_rows_falls_back_to_basename_with_no_task_or_metadata(
    tmp_path: Path,
) -> None:
    media = tmp_path / 'unrelated_dir' / 'clip.mp4'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'\x00')
    example = Example(
        example_id='ex3',
        source_id='s3',
        origin='meld',
        kind=Kind.SENTIMENT,
        split=Split.BUILD,
        media_path=str(media),
        prompt='identify the most prominent sentiment displayed in the video. '
        'Candidate labels: positive, negative. Please think step by step. '
        'Enclose your final answer within <answer></answer> tags.',
        label_set=('positive', 'negative'),
        labels=('negative',),
    )
    prediction = PredictionRecord(
        example_id='ex3',
        row='L0',
        labels=('negative',),
        response_text='x',
        target_calls=1,
    )
    rows = to_response_rows((prediction,), {'ex3': example})
    assert rows[0]['video_id'] == 'clip.mp4'


def test_to_response_rows_raises_for_unknown_example() -> None:
    prediction = PredictionRecord(
        example_id='missing', row='L0', labels=('a',), response_text='x', target_calls=1
    )
    with pytest.raises(ValidationError):
        to_response_rows((prediction,), {})


def test_write_response_json_round_trips(tmp_path: Path) -> None:
    rows = ({'video_id': 'a.mp4', 'ground_truth': 'anger', 'model_response': 'anger'},)
    path = tmp_path / 'response.json'
    write_response_json(path, rows)
    from emoharness.common.io import read_json

    assert read_json(path) == list(rows)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('<score>Step 1: 1/1, Step 2: 0/1, Step N: 2/3</score>', (1.0, 0.0, 2 / 3)),
        ('<score>Step 1: 0/0</score>', (0.0,)),
        ('no score tag here', None),
        ('<score>nothing parseable</score>', None),
        ('prefix <score>Step 1: 1/1</score> suffix', (1.0,)),
    ],
)
def test_parse_score_string(text: str, expected: tuple[float, ...] | None) -> None:
    assert parse_score_string(text) == expected


def test_step_scores_from_eval_json(tmp_path: Path) -> None:
    path = tmp_path / 'eval.json'
    from emoharness.common.io import write_json_atomic

    write_json_atomic(
        path,
        [
            {'video_id': 'a.mp4', 'score': '<score>Step 1: 1/1, Step 2: 0/1</score>'},
            {'video_id': 'b.mp4', 'score': 'garbage, no tag'},
            {'video_id': 'c.mp4', 'score': '<score>Step 1: 1/1</score>'},
        ],
    )
    scores = step_scores_from_eval_json(path)
    assert {s.example_id for s in scores} == {'a.mp4', 'c.mp4'}
    by_id = {s.example_id: s for s in scores}
    assert by_id['a.mp4'].scores == (1.0, 0.0)


def test_step_scores_from_eval_json_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    from emoharness.common.io import write_json_atomic

    write_json_atomic(path, {'not': 'a list'})
    with pytest.raises(ValidationError):
        step_scores_from_eval_json(path)


def test_judge_cost_estimate_scales_linearly() -> None:
    assert judge_cost_estimate(0) == 0.0
    single = judge_cost_estimate(1, tokens_per_item=1_000_000, price_per_million=2.0)
    assert single == pytest.approx(2.0)
    assert judge_cost_estimate(100) == pytest.approx(100 * judge_cost_estimate(1))
    with pytest.raises(ValidationError):
        judge_cost_estimate(-1)


class _RecordingRunner:
    def __init__(self, exit_codes: Sequence[int]) -> None:
        self._exit_codes = list(exit_codes)
        self.calls: list[Sequence[str]] = []

    def __call__(self, command: Sequence[str]) -> int:
        self.calls.append(command)
        return self._exit_codes[len(self.calls) - 1]


def _command(tmp_path: Path, audio_json: str | None = None) -> OfficialJudgeCommand:
    return OfficialJudgeCommand(
        repo_path='/repo',
        task=Task.ML_ER,
        response_json=str(tmp_path / 'response.json'),
        steps_json=str(tmp_path / 'steps.json'),
        eval_json=str(tmp_path / 'eval.json'),
        metrics_txt=str(tmp_path / 'metrics.txt'),
        video_dir=str(tmp_path / 'videos'),
        audio_json=audio_json,
    )


def test_official_judge_command_uses_hyphenated_script_dirs(tmp_path: Path) -> None:
    cmd = _command(tmp_path)
    extract = cmd.extract_step_command()
    assert extract[:2] == ('python', '/repo/extract_step/ML-ER/code/extract_step.py')
    assert '--input_json' in extract and cmd.response_json in extract
    assert JUDGE_STEP_MODEL in extract

    eval_cmd = cmd.eval_cot_command()
    assert eval_cmd[:2] == ('python', '/repo/eval_cot/ML-ER/code/eval_cot_gpt4o.py')
    assert '--audio_json' not in eval_cmd

    metrics_cmd = cmd.cal_metrics_command()
    assert metrics_cmd[:2] == ('python', '/repo/eval_cot/ML-ER/metrics/cal_metrics.py')
    assert str(COT_ALPHA) in metrics_cmd


def test_official_judge_command_forwards_audio_json_without_reading_it(
    tmp_path: Path,
) -> None:
    missing_audio_path = str(tmp_path / 'does_not_exist_audio_clue.json')
    cmd = _command(tmp_path, audio_json=missing_audio_path)
    eval_cmd = cmd.eval_cot_command()
    assert '--audio_json' in eval_cmd
    assert eval_cmd[eval_cmd.index('--audio_json') + 1] == missing_audio_path

    # Never opened: building/running the pipeline must not raise even though
    # the audio_clue path does not exist on disk.
    runner = _RecordingRunner([0, 0, 0])
    codes = run_official_pipeline(cmd, runner=runner)
    assert codes == (0, 0, 0)
    assert len(runner.calls) == 3


def test_run_official_pipeline_stops_at_first_failure(tmp_path: Path) -> None:
    cmd = _command(tmp_path)
    runner = _RecordingRunner([1, 0, 0])
    codes = run_official_pipeline(cmd, runner=runner)
    assert codes == (1,)
    assert len(runner.calls) == 1
