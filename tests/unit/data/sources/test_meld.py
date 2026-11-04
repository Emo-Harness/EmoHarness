from __future__ import annotations

import csv
from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.data.sources import meld
from emoharness.data.tasks import Kind, Split

_FIELDS = [
    'Sr No.',
    'Utterance',
    'Speaker',
    'Emotion',
    'Sentiment',
    'Dialogue_ID',
    'Utterance_ID',
]
_ROWS = [
    {
        'Sr No.': '1',
        'Utterance': 'Oh my God, he’s lost it.',
        'Speaker': 'Chandler',
        'Emotion': 'surprise',
        'Sentiment': 'negative',
        'Dialogue_ID': '0',
        'Utterance_ID': '0',
    },
    {
        'Sr No.': '2',
        'Utterance': 'I am fine.',
        'Speaker': 'Monica',
        'Emotion': 'neutral',
        'Sentiment': 'neutral',
        'Dialogue_ID': '0',
        'Utterance_ID': '1',
    },
]


def _write_split(root: Path, split: str) -> None:
    path = root / f'{split}_sent_emo.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(_ROWS)


def test_load_emotion_reads_the_emotion_column(tmp_path: Path) -> None:
    _write_split(tmp_path, 'train')
    examples = meld.load_emotion(tmp_path, ('train',))
    assert len(examples) == 2
    first = examples[0]
    assert first.example_id == 'meld_train_dia0_utt0'
    assert first.source_id == 'meld_train_dia0'
    assert first.origin == 'meld'
    assert first.kind is Kind.WILD_ER
    assert first.split is Split.BUILD
    assert first.labels == ('surprise',)
    assert first.label_set == meld.EMOTION_LABELS
    assert first.media_path == str(tmp_path / 'train_splits' / 'dia0_utt0.mp4')
    assert first.language == 'en'


def test_load_sentiment_reads_the_sentiment_column(tmp_path: Path) -> None:
    _write_split(tmp_path, 'train')
    examples = meld.load_sentiment(tmp_path, ('train',))
    assert examples[0].example_id == 'meld_sentiment_train_dia0_utt0'
    assert examples[0].origin == 'meld_sentiment'
    assert examples[0].kind is Kind.SENTIMENT
    assert examples[0].labels == ('negative',)
    assert examples[0].label_set == ('positive', 'negative', 'neutral')


def test_emotion_and_sentiment_ids_never_collide(tmp_path: Path) -> None:
    _write_split(tmp_path, 'train')
    emotion_ids = {e.example_id for e in meld.load_emotion(tmp_path, ('train',))}
    sentiment_ids = {e.example_id for e in meld.load_sentiment(tmp_path, ('train',))}
    assert emotion_ids.isdisjoint(sentiment_ids)


def test_multiple_splits_are_concatenated(tmp_path: Path) -> None:
    _write_split(tmp_path, 'train')
    _write_split(tmp_path, 'dev')
    examples = meld.load_emotion(tmp_path, ('train', 'dev'))
    assert len(examples) == 4
    assert {e.source_id for e in examples} == {'meld_train_dia0', 'meld_dev_dia0'}


def test_unknown_emotion_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / 'train_sent_emo.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        bad_row = dict(_ROWS[0])
        bad_row['Emotion'] = 'confused'
        writer.writerow(bad_row)
    with pytest.raises(ValidationError, match='not one of'):
        meld.load_emotion(tmp_path, ('train',))


def test_missing_split_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='cannot read CSV'):
        meld.load_emotion(tmp_path, ('train',))
