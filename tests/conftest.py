"""Shared fixtures. Media are synthetic and generated per test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import (
    render_prompt_for_kind,
    render_prompt_for_task,
)
from emoharness.data.tasks import Kind, Split, Task

_ER_LABELS = ('anger', 'happiness', 'sadness', 'neutral')
_SA_LABELS = ('positive', 'negative', 'neutral')


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark_example(tmp_path: Path) -> Example:
    media = tmp_path / 'clip_er.mp4'
    media.write_bytes(b'\x00' * 64)
    return Example(
        example_id='mme_er_wild_000001',
        source_id='dia_1',
        origin='mme_emotion',
        task=Task.ER_WILD,
        split=Split.TEST,
        media_path=str(media),
        prompt=render_prompt_for_task(Task.ER_WILD, _ER_LABELS),
        label_set=_ER_LABELS,
        labels=('sadness',),
    )


@pytest.fixture
def multilabel_example(tmp_path: Path) -> Example:
    media = tmp_path / 'clip_ml.mp4'
    media.write_bytes(b'\x00' * 64)
    labels = ('anger', 'disgust', 'fear', 'sadness', 'happiness')
    return Example(
        example_id='mme_ml_er_000001',
        source_id='03419',
        origin='mme_emotion',
        task=Task.ML_ER,
        split=Split.TEST,
        media_path=str(media),
        prompt=render_prompt_for_task(Task.ML_ER, labels),
        label_set=labels,
        labels=('anger', 'disgust'),
    )


@pytest.fixture
def external_example(tmp_path: Path) -> Example:
    media = tmp_path / 'clip_sa.mp4'
    media.write_bytes(b'\x00' * 64)
    return Example(
        example_id='meld_train_dia0_utt0',
        source_id='meld_dia0',
        origin='meld',
        kind=Kind.SENTIMENT,
        split=Split.BUILD,
        media_path=str(media),
        prompt=render_prompt_for_kind(Kind.SENTIMENT, _SA_LABELS),
        label_set=_SA_LABELS,
        labels=('negative',),
        language='en',
    )
