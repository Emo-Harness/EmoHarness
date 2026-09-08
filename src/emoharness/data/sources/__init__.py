"""External construction source adapters and the origin registry (ADR-0003)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from emoharness.data.sources import chsims, cremad, iemocap, meld, mosei
from emoharness.data.sources.base import FunctionAdapter, SourceAdapter

ADAPTERS: Final[Mapping[str, SourceAdapter]] = {
    'meld': FunctionAdapter(meld.load_emotion),
    'meld_sentiment': FunctionAdapter(meld.load_sentiment),
    'mosei': FunctionAdapter(mosei.load),
    'chsims': FunctionAdapter(chsims.load),
    'iemocap': FunctionAdapter(iemocap.load),
    'cremad': FunctionAdapter(cremad.load),
}

__all__ = ['ADAPTERS', 'FunctionAdapter', 'SourceAdapter']
