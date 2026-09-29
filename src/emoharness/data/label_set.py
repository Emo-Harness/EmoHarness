"""The Label set: one item's candidate labels and every operation over them.

``emoharness.data.prompt_template`` owns the prompt grammar; this module owns
what the project does with the candidate list that grammar yields — canonical
spelling, answer parsing, and the local Recognition and PRF scores. It reads the
prompt and nothing else: a task name never reaches it, so a held-out wording is
handled by the same code as a seen one.

Normalization exists only to compare a prediction with the gold labels. Emitted
answers are always the verbatim candidate strings the prompt lists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    GOLD_LABEL_SEPARATORS,
    LABEL_SEPARATOR,
    OFFICIAL_SYNONYMS,
)
from emoharness.common.errors import ValidationError
from emoharness.data.prompt_template import (
    Cardinality,
    ParsedPrompt,
    Quantifier,
    cardinality_for,
    parse_prompt,
)

DEFAULT_NOUN: Final = 'emotion'

_SPLIT_RE: Final = re.compile('[' + re.escape(''.join(GOLD_LABEL_SEPARATORS)) + ']')
_ANSWER_RE: Final = re.compile(
    re.escape(ANSWER_OPEN) + r'(.*?)' + re.escape(ANSWER_CLOSE), re.DOTALL
)


def normalize_label(label: str) -> str:
    """Lowercase, collapse whitespace, and apply the official synonym map.

    Public because ingestion histograms and failure clustering need one label key
    that is comparable across items, and so across label sets.
    """
    collapsed = ' '.join(label.strip().lower().split())
    return OFFICIAL_SYNONYMS.get(collapsed, collapsed)


def split_labels(raw: str) -> tuple[str, ...]:
    """Split ``'anger,disgust'`` or ``'A;B'`` into its labels, blanks dropped.

    Public for the same reason as :func:`normalize_label`: a gold string arrives
    from a source file before any candidate list is known.
    """
    return tuple(part.strip() for part in _SPLIT_RE.split(raw) if part.strip())


class PRF(BaseModel):
    """Local secondary metric: precision, recall, and F1 over normalized label sets."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class LabelSet(BaseModel):
    """The candidate labels of one item, with the quantifier and noun they came with.

    Build one from the item's prompt and ask it everything: how many labels an
    answer may carry, how a candidate is spelled, what a response answered, and
    what that answer scores against the gold labels.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    labels: tuple[str, ...] = Field(
        min_length=2, description='candidates, verbatim as the prompt lists them'
    )
    quantifier: Quantifier
    noun: str = Field(min_length=1)

    @classmethod
    def from_prompt(cls, prompt: str) -> LabelSet:
        """Parse an official prompt and take its candidate list."""
        return cls.from_parsed(parse_prompt(prompt))

    @classmethod
    def from_parsed(cls, parsed: ParsedPrompt) -> LabelSet:
        """The label set an already-parsed prompt carries."""
        return cls(
            labels=parsed.label_set, quantifier=parsed.quantifier, noun=parsed.noun
        )

    @classmethod
    def from_labels(
        cls,
        labels: Iterable[str],
        quantifier: Quantifier = Quantifier.MOST_PROMINENT,
        noun: str = DEFAULT_NOUN,
    ) -> LabelSet:
        """A label set from a bare candidate list, for callers with no prompt."""
        return cls(labels=tuple(labels), quantifier=quantifier, noun=noun)

    @property
    def is_multi_label(self) -> bool:
        """Whether an answer may name more than one label.

        Decided by the prompt's quantifier alone, never by a task name, so a
        held-out wording is classified by the same rule as a seen one.
        """
        return self.quantifier is not Quantifier.MOST_PROMINENT

    def cardinality(self, cap: int) -> Cardinality:
        """How many labels this item's answer may carry, given the multi-label cap."""
        return cardinality_for(self.quantifier, cap)

    def canonical(self, text: str) -> str | None:
        """The candidate ``text`` names, spelled as the prompt spells it, or None."""
        return self._by_norm().get(normalize_label(text))

    def canonical_many(self, texts: Iterable[str]) -> tuple[str, ...]:
        """Canonical spelling of each text; unknowns dropped, order-preserving dedupe."""
        found = (self.canonical(text) for text in texts)
        return _dedupe(label for label in found if label is not None)

    def parse_answer(self, text: str) -> tuple[str, ...]:
        """The candidates a response answered, verbatim, or ``()``.

        The last ``<answer>...</answer>`` block wins. Its content is split on the
        gold separators and matched against the candidates; when that yields
        nothing — no block, or a block naming nothing recognizable — the whole
        response is searched the same way, then scanned for candidates mentioned
        anywhere in it, in candidate order.
        """
        block = _answer_block(text)
        labels = self._match(block) if block is not None else ()
        return labels or self._match(text)

    def render(self, labels: Iterable[str]) -> str:
        """Labels joined the way the official prompt and answer block join them."""
        return LABEL_SEPARATOR.join(labels)

    def overlap(self, predicted: Iterable[str], gold: Iterable[str]) -> int:
        """How many distinct labels the two sets share after normalization."""
        return len(_normalized(predicted) & _normalized(gold))

    def recognition(self, predicted: Iterable[str], gold: Iterable[str]) -> float:
        """Official Recognition score, computed locally.

        Single-label: 1.0 when the first predicted label matches any gold label
        after normalization, else 0.0. Multi-label: the official C/K — the size of
        the normalized intersection divided by the gold count.
        """
        gold_labels = tuple(gold)
        if not gold_labels:
            raise ValidationError('recognition requires at least one gold label')
        predicted_labels = tuple(predicted)
        if self.is_multi_label:
            return self.overlap(predicted_labels, gold_labels) / len(gold_labels)
        if not predicted_labels:
            return 0.0
        first = normalize_label(predicted_labels[0])
        return 1.0 if first in _normalized(gold_labels) else 0.0

    def prf(self, predicted: Iterable[str], gold: Iterable[str]) -> PRF:
        """Local secondary metric: precision, recall, F1 after normalization."""
        predicted_set, gold_set = _normalized(predicted), _normalized(gold)
        correct = len(predicted_set & gold_set)
        precision = correct / len(predicted_set) if predicted_set else 0.0
        recall = correct / len(gold_set) if gold_set else 0.0
        total = precision + recall
        f1 = 2 * precision * recall / total if total else 0.0
        return PRF(precision=precision, recall=recall, f1=f1)

    def _by_norm(self) -> dict[str, str]:
        return {normalize_label(label): label for label in self.labels}

    def _match(self, raw: str) -> tuple[str, ...]:
        """Candidates named by ``raw``: separated parts first, then bare mentions."""
        exact = self.canonical_many(split_labels(raw))
        if exact:
            return exact
        lowered = raw.lower()
        return _dedupe(
            label for label in self.labels if normalize_label(label) in lowered
        )


def _answer_block(text: str) -> str | None:
    """The content of the last answer block, or None when there is none."""
    matches = _ANSWER_RE.findall(text)
    return str(matches[-1]).strip() if matches else None


def _normalized(labels: Iterable[str]) -> frozenset[str]:
    return frozenset(normalize_label(label) for label in labels)


def _dedupe(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(labels))
