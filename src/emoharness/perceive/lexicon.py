"""Emotion-word lexicon loading and sentence-level filtering (ADR-0004).

Removes sentences that would leak an emotion label or a common paraphrase into the
observable-cue audio description, keeping the rest of the description intact.
"""

from __future__ import annotations

import re
from pathlib import Path

from emoharness.common.errors import ValidationError

_LABEL_WORD_SPLIT_RE = re.compile(r'[ /\-]+')
_SENTENCE_SPLIT_RE = re.compile(r'[.!?\n]+')
_WORD_RE = re.compile(r"[A-Za-z']+")


def load_lexicon(path: Path) -> frozenset[str]:
    """One lowercase word per non-comment, non-blank line of ``path``."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        raise ValidationError(f'cannot read lexicon {path}: {exc}') from exc
    return frozenset(
        stripped.lower()
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith('#')
    )


def banned_words(label_set: tuple[str, ...], lexicon: frozenset[str]) -> frozenset[str]:
    """``lexicon`` plus every word of every candidate label, lowercased.

    Labels are split on spaces, slashes, and hyphens, e.g. ``'Doubt/Confusion'``
    contributes ``'doubt'`` and ``'confusion'``.
    """
    label_words = {
        word.lower()
        for label in label_set
        for word in _LABEL_WORD_SPLIT_RE.split(label)
        if word
    }
    return lexicon | frozenset(label_words)


def filter_description(text: str, banned: frozenset[str]) -> str:
    """Drop every sentence of ``text`` containing a banned word; rejoin the rest.

    Sentences are split on ``.``, ``!``, ``?``, and newlines; the survivors are
    joined back with ``'. '``.
    """
    sentences = (s.strip() for s in _SENTENCE_SPLIT_RE.split(text))
    kept = [s for s in sentences if s and not contains_banned(s, banned)]
    return '. '.join(kept)


def contains_banned(text: str, banned: frozenset[str]) -> bool:
    """True if any word of ``text`` matches ``banned`` at a word boundary, case-insensitively."""
    if not banned:
        return False
    words = {match.lower() for match in _WORD_RE.findall(text)}
    return not words.isdisjoint(banned)
