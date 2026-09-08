"""Sentence splitting, shared by the evidence builder and the output controller.

Both cut a block of prose into sentences: the evidence builder to number one cue
per sentence, the controller to drop the sentences that cite nothing. The rule
lives here once so the two cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Final

_SENTENCE_RE: Final = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> tuple[str, ...]:
    """Non-empty sentences of a block of text, each stripped of outer whitespace."""
    return tuple(part.strip() for part in _SENTENCE_RE.split(text) if part.strip())
