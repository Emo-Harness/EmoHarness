"""MELD adapter: dialogue/utterance clips with emotion and sentiment labels.

Assumes the standard MELD.Raw layout: ``train_sent_emo.csv`` and
``dev_sent_emo.csv`` at the corpus root, each with (at least) the columns
``Sr No., Utterance, Speaker, Emotion, Sentiment, Dialogue_ID, Utterance_ID``,
and clips at ``<root>/<split>_splits/dia<Dialogue_ID>_utt<Utterance_ID>.mp4``
(no separating underscore, matching the official release's file names -- the
benchmark's own repackaging uses a different, underscored convention, handled
separately by ``emoharness.data.benchmark.derive_source_id``).

Two origins are produced from the same rows: 'meld' (Kind.WILD_ER, the
``Emotion`` column, 7 classes) and 'meld_sentiment' (Kind.SENTIMENT, the
``Sentiment`` column, 3 classes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import render_prompt_for_kind
from emoharness.data.sources.base import (
    SENTIMENT_LABELS,
    parse_int,
    read_csv_rows,
    require_canonical,
    require_column,
)
from emoharness.data.tasks import Kind, Split

EMOTION_LABELS: Final[tuple[str, ...]] = (
    'anger',
    'disgust',
    'fear',
    'joy',
    'neutral',
    'sadness',
    'surprise',
)


def load_emotion(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
    """Load MELD rows as 7-class in-the-wild emotion recognition examples."""
    return tuple(
        _to_example(
            root,
            split,
            row,
            origin='meld',
            kind=Kind.WILD_ER,
            label_set=EMOTION_LABELS,
            label_column='Emotion',
        )
        for split in split_names
        for row in read_csv_rows(root / f'{split}_sent_emo.csv')
    )


def load_sentiment(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
    """Load MELD rows as 3-class sentiment examples."""
    return tuple(
        _to_example(
            root,
            split,
            row,
            origin='meld_sentiment',
            kind=Kind.SENTIMENT,
            label_set=SENTIMENT_LABELS,
            label_column='Sentiment',
        )
        for split in split_names
        for row in read_csv_rows(root / f'{split}_sent_emo.csv')
    )


def _to_example(
    root: Path,
    split: str,
    row: dict[str, str],
    *,
    origin: str,
    kind: Kind,
    label_set: tuple[str, ...],
    label_column: str,
) -> Example:
    context = f'{origin} {split} row'
    dialogue_id = parse_int(
        require_column(row, 'Dialogue_ID', context=context), context=context
    )
    utterance_id = parse_int(
        require_column(row, 'Utterance_ID', context=context), context=context
    )
    raw_label = require_column(row, label_column, context=context)
    label = require_canonical(raw_label, label_set, context=context)
    media = root / f'{split}_splits' / f'dia{dialogue_id}_utt{utterance_id}.mp4'
    return Example(
        example_id=f'{origin}_{split}_dia{dialogue_id}_utt{utterance_id}',
        source_id=f'meld_{split}_dia{dialogue_id}',
        origin=origin,
        kind=kind,
        split=Split.BUILD,
        media_path=str(media),
        prompt=render_prompt_for_kind(kind, label_set),
        label_set=label_set,
        labels=(label,),
        language='en',
    )
