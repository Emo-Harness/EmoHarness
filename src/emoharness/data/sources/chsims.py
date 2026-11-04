"""CH-SIMS adapter: Chinese multimodal sentiment clips collapsed to 3 classes.

Assumes ``label.csv`` at the corpus root with columns ``video_id, clip_id,
label`` (a continuous score in [-1, 1]), and clips at
``<root>/Raw/<video_id>/<clip_id>.mp4``. Collapsed the same way as MOSEI:
score > 0 positive, score < 0 negative, score == 0 neutral.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import render_prompt_for_kind
from emoharness.data.sources.base import (
    SENTIMENT_LABELS,
    collapse_sentiment,
    parse_float,
    read_csv_rows,
    require_column,
)
from emoharness.data.tasks import Kind, Split

_LABELS_FILE: Final = 'label.csv'


def load(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:  # noqa: ARG001
    """Load every row of ``label.csv`` as a sentiment example.

    ``split_names`` is accepted for ``SourceAdapter`` conformity; this format
    carries no split column, so every row is loaded regardless of its value.
    """
    return tuple(_to_example(root, row) for row in read_csv_rows(root / _LABELS_FILE))


def _to_example(root: Path, row: dict[str, str]) -> Example:
    video_id = require_column(row, 'video_id', context='chsims row')
    context = f'chsims {video_id}'
    clip_id = require_column(row, 'clip_id', context=context)
    raw_score = require_column(row, 'label', context=context)
    score = parse_float(raw_score, context=f'{context}/{clip_id} label')
    media = root / 'Raw' / video_id / f'{clip_id}.mp4'
    return Example(
        example_id=f'chsims_{video_id}_{clip_id}',
        source_id=video_id,
        origin='chsims',
        kind=Kind.SENTIMENT,
        split=Split.BUILD,
        media_path=str(media),
        prompt=render_prompt_for_kind(Kind.SENTIMENT, SENTIMENT_LABELS),
        label_set=SENTIMENT_LABELS,
        labels=(collapse_sentiment(score),),
        language='zh',
    )
