"""The public MME-Emotion prompt grammar: render a prompt, parse a prompt.

The harness never receives a task name. Everything it may know about the item
comes from parsing the prompt with this module (ADR-0001, ADR-0010).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import LABEL_SEPARATOR, OFFICIAL_PROMPT_TEMPLATE
from emoharness.common.errors import PromptParseError
from emoharness.data.tasks import Kind, Task


class Quantifier(str, Enum):
    """The three quantifiers the public template uses."""

    MOST_PROMINENT = 'the most prominent'
    ONE_OR_SEVERAL = 'one or several'
    MULTIPLE = 'multiple'


class Cardinality(BaseModel):
    """How many labels an answer may carry. ``maximum`` None means uncapped."""

    model_config = ConfigDict(frozen=True)

    minimum: int = Field(ge=1)
    maximum: int | None = Field(default=None, ge=1)


class ParsedPrompt(BaseModel):
    """Everything the harness may read from an item's prompt."""

    model_config = ConfigDict(frozen=True)

    quantifier: Quantifier
    noun: str
    label_set: tuple[str, ...] = Field(min_length=2)


TASK_PROMPT_SPEC: Final[Mapping[Task, tuple[Quantifier, str]]] = {
    Task.ER_LAB: (Quantifier.MOST_PROMINENT, 'emotion'),
    Task.ER_WILD: (Quantifier.MOST_PROMINENT, 'emotion'),
    Task.NOISE_ER: (Quantifier.MOST_PROMINENT, 'emotion'),
    Task.FG_ER: (Quantifier.ONE_OR_SEVERAL, 'emotions'),
    Task.ML_ER: (Quantifier.MULTIPLE, 'emotions'),
    Task.SA: (Quantifier.MOST_PROMINENT, 'sentiment'),
    Task.FG_SA: (Quantifier.MOST_PROMINENT, 'fine-grained sentiment'),
    Task.IR: (Quantifier.MOST_PROMINENT, 'intent'),
}

KIND_PROMPT_SPEC: Final[Mapping[Kind, tuple[Quantifier, str]]] = {
    Kind.LAB_ER: (Quantifier.MOST_PROMINENT, 'emotion'),
    Kind.WILD_ER: (Quantifier.MOST_PROMINENT, 'emotion'),
    Kind.SENTIMENT: (Quantifier.MOST_PROMINENT, 'sentiment'),
}

_QUANTIFIER_PATTERN = '|'.join(re.escape(q.value) for q in Quantifier)
_PROMPT_RE: Final = re.compile(
    r'identify (?P<quantifier>' + _QUANTIFIER_PATTERN + r') (?P<noun>.+?) displayed in '
    r'the video\. Candidate labels: (?P<labels>.+?)\. Please think step by step',
    re.DOTALL,
)


def render_prompt(quantifier: Quantifier, noun: str, label_set: tuple[str, ...]) -> str:
    """Fill the official template."""
    if len(label_set) < 2:
        raise PromptParseError('a prompt needs at least two candidate labels')
    return OFFICIAL_PROMPT_TEMPLATE.format(
        quantifier=quantifier.value, noun=noun, labels=LABEL_SEPARATOR.join(label_set)
    )


def render_prompt_for_task(task: Task, label_set: tuple[str, ...]) -> str:
    """Render the official prompt for a benchmark task."""
    quantifier, noun = TASK_PROMPT_SPEC[task]
    return render_prompt(quantifier, noun, label_set)


def render_prompt_for_kind(kind: Kind, label_set: tuple[str, ...]) -> str:
    """Render the seen-task wording for an external construction example."""
    quantifier, noun = KIND_PROMPT_SPEC[kind]
    return render_prompt(quantifier, noun, label_set)


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse quantifier, noun, and candidate labels from a prompt or raise."""
    match = _PROMPT_RE.search(prompt)
    if match is None:
        raise PromptParseError('prompt does not follow the MME-Emotion template')
    labels = tuple(
        part.strip() for part in match.group('labels').split(LABEL_SEPARATOR.strip())
    )
    labels = tuple(label for label in labels if label)
    if len(labels) < 2:
        raise PromptParseError('prompt lists fewer than two candidate labels')
    return ParsedPrompt(
        quantifier=Quantifier(match.group('quantifier')),
        noun=match.group('noun').strip(),
        label_set=labels,
    )


def cardinality_for(quantifier: Quantifier, cap: int) -> Cardinality:
    """Map a quantifier to an allowed label count. ``cap`` is the multi-label cap."""
    if quantifier is Quantifier.MOST_PROMINENT:
        return Cardinality(minimum=1, maximum=1)
    if quantifier is Quantifier.ONE_OR_SEVERAL:
        return Cardinality(minimum=1, maximum=cap)
    return Cardinality(minimum=min(2, cap), maximum=cap)
