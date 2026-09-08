from __future__ import annotations

from emoharness.data.sources import ADAPTERS, chsims, cremad, iemocap, meld, mosei
from emoharness.data.sources.base import FunctionAdapter, SourceAdapter


def _loader_of(adapter: SourceAdapter) -> object:
    assert isinstance(adapter, FunctionAdapter)
    return adapter.loader


def test_registry_has_exactly_the_documented_origins() -> None:
    assert set(ADAPTERS) == {
        'meld',
        'meld_sentiment',
        'mosei',
        'chsims',
        'iemocap',
        'cremad',
    }


def test_registry_wires_each_origin_to_its_loader() -> None:
    assert _loader_of(ADAPTERS['meld']) is meld.load_emotion
    assert _loader_of(ADAPTERS['meld_sentiment']) is meld.load_sentiment
    assert _loader_of(ADAPTERS['mosei']) is mosei.load
    assert _loader_of(ADAPTERS['chsims']) is chsims.load
    assert _loader_of(ADAPTERS['iemocap']) is iemocap.load
    assert _loader_of(ADAPTERS['cremad']) is cremad.load
