r"""IEMOCAP adapter: pre-segmented lab emotion recognition utterances.

Assumes the official release layout: one ``Session<k>/dialog/EmoEvaluation/*.txt``
per session, with summary lines such as::

    [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]

giving the utterance id and a 3-letter categorical code (other lines in these
files -- comments, per-annotator detail -- do not match this shape and are
skipped). Clips are assumed pre-cut at ``<root>/clips/<utterance_id>.mp4``.
``xxx`` (no majority agreement) and ``dis`` (disgust, too rare to model) are
dropped; the remaining nine codes give the label set, in the order below.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from emoharness.data.manifest import Example
from emoharness.data.prompt_template import render_prompt_for_kind
from emoharness.data.sources.base import parse_int
from emoharness.data.tasks import Kind, Split

LABELS: Final[tuple[str, ...]] = (
    'Anger',
    'Sadness',
    'Surprise',
    'Happiness',
    'Excited',
    'Fear',
    'Frustration',
    'Neutral',
    'Other',
)
_CODE_TO_LABEL: Final[dict[str, str]] = {
    'ang': 'Anger',
    'sad': 'Sadness',
    'sur': 'Surprise',
    'hap': 'Happiness',
    'exc': 'Excited',
    'fea': 'Fear',
    'fru': 'Frustration',
    'neu': 'Neutral',
    'oth': 'Other',
}
_LINE_RE: Final = re.compile(
    r'^\[\s*[\d.]+\s*-\s*[\d.]+\s*\]\t(?P<utt_id>\S+)\t(?P<code>[a-z]{3})\t\['
)
_SOURCE_ID_RE: Final = re.compile(r'^(Ses\d{2})([A-Za-z])')


def load(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
    """``SourceAdapter``-conformant entry point; session numbers as strings."""
    sessions = tuple(
        parse_int(name, context='iemocap split_names') for name in split_names
    )
    return load_sessions(root, sessions)


def load_sessions(root: Path, sessions: tuple[int, ...]) -> tuple[Example, ...]:
    """Load specific IEMOCAP session numbers (1-indexed) as lab ER examples."""
    return tuple(
        _to_example(root, utt_id, code)
        for session in sessions
        for utt_id, code in _iter_utterances(root, session)
    )


def _iter_utterances(root: Path, session: int) -> Iterator[tuple[str, str]]:
    eval_dir = root / f'Session{session}' / 'dialog' / 'EmoEvaluation'
    for path in sorted(eval_dir.glob('*.txt')):
        for line in path.read_text(encoding='utf-8').splitlines():
            match = _LINE_RE.match(line)
            if match is not None and match.group('code') in _CODE_TO_LABEL:
                yield match.group('utt_id'), match.group('code')


def _to_example(root: Path, utt_id: str, code: str) -> Example:
    media = root / 'clips' / f'{utt_id}.mp4'
    return Example(
        example_id=f'iemocap_{utt_id}',
        source_id=_derive_source_id(utt_id),
        origin='iemocap',
        kind=Kind.LAB_ER,
        split=Split.BUILD,
        media_path=str(media),
        prompt=render_prompt_for_kind(Kind.LAB_ER, LABELS),
        label_set=LABELS,
        labels=(_CODE_TO_LABEL[code],),
        language='en',
    )


def _derive_source_id(utt_id: str) -> str:
    """'Ses01F_impro01_F000' -> 'Ses01_F': session number plus recording gender."""
    match = _SOURCE_ID_RE.match(utt_id)
    if match is None:
        return utt_id
    return f'{match.group(1)}_{match.group(2)}'
