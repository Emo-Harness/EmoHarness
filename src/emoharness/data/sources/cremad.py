"""CREMA-D adapter: acted lab emotion recognition clips (IEMOCAP fallback).

Assumes the official ``VideoFlash`` layout: files named
``<actor_id>_<sentence>_<code>_<level>.flv`` (or ``.mp4``), e.g.
``1001_DFA_ANG_XX.flv``. Used in place of IEMOCAP sessions 1-4 when the
IEMOCAP license has not arrived by milestone M2.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import render_prompt_for_kind
from emoharness.data.tasks import Kind, Split

LABELS: Final[tuple[str, ...]] = ('anger', 'disgust', 'fear', 'happy', 'neutral', 'sad')
_CODE_TO_LABEL: Final[dict[str, str]] = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}
_FILENAME_RE: Final = re.compile(
    r'^(?P<actor_id>\d+)_(?P<sentence>[A-Z]+)_(?P<code>[A-Z]+)_(?P<level>\w+)\.'
    r'(?:flv|mp4)$'
)
_CLIP_DIR: Final = 'VideoFlash'


def load(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:  # noqa: ARG001
    """Load every recognized clip in ``VideoFlash/`` as a lab ER example.

    ``split_names`` is accepted for ``SourceAdapter`` conformity; CREMA-D is a
    single flat corpus with no split partitioning, so every clip is loaded.
    """
    clip_dir = root / _CLIP_DIR
    matches = (
        (path, _FILENAME_RE.match(path.name)) for path in sorted(clip_dir.glob('*'))
    )
    return tuple(
        _to_example(path, match)
        for path, match in matches
        if match is not None and match.group('code') in _CODE_TO_LABEL
    )


def _to_example(path: Path, match: re.Match[str]) -> Example:
    return Example(
        example_id=f'cremad_{path.stem}',
        source_id=match.group('actor_id'),
        origin='cremad',
        kind=Kind.LAB_ER,
        split=Split.BUILD,
        media_path=str(path),
        prompt=render_prompt_for_kind(Kind.LAB_ER, LABELS),
        label_set=LABELS,
        labels=(_CODE_TO_LABEL[match.group('code')],),
        language='en',
    )
