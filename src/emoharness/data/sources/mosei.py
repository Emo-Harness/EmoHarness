"""CMU-MOSEI adapter: YouTube monologue segments collapsed to 3-class sentiment.

Assumes a normalized ``mosei_labels.csv`` at the corpus root with columns
``video_id, segment_id, sentiment, path``: ``sentiment`` is the native
[-3, 3] CMU Multimodal SDK score and ``path`` is the clip location relative to
the corpus root. The 7-level score is never stored in the manifest,
so only the collapsed 3-class label is kept; ``metadata['collapsed_from']``
records the source scale for provenance.
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

_LABELS_FILE: Final = 'mosei_labels.csv'


def load(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:  # noqa: ARG001
    """Load every row of ``mosei_labels.csv`` as a sentiment example.

    ``split_names`` is accepted for ``SourceAdapter`` conformity; this format
    carries no split column, so every row is loaded regardless of its value.
    """
    return tuple(_to_example(root, row) for row in read_csv_rows(root / _LABELS_FILE))


def _to_example(root: Path, row: dict[str, str]) -> Example:
    video_id = require_column(row, 'video_id', context='mosei row')
    context = f'mosei {video_id}'
    segment_id = require_column(row, 'segment_id', context=context)
    raw_score = require_column(row, 'sentiment', context=context)
    score = parse_float(raw_score, context=f'{context}/{segment_id} sentiment')
    clip_path = require_column(row, 'path', context=context)
    return Example(
        example_id=f'mosei_{video_id}_{segment_id}',
        source_id=video_id,
        origin='mosei',
        kind=Kind.SENTIMENT,
        split=Split.BUILD,
        media_path=str(root / clip_path),
        prompt=render_prompt_for_kind(Kind.SENTIMENT, SENTIMENT_LABELS),
        label_set=SENTIMENT_LABELS,
        labels=(collapse_sentiment(score),),
        language='en',
        metadata={'collapsed_from': 'mosei_7'},
    )
