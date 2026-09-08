"""Canonical example record shared by benchmark and external data (README §5.3)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.io import iter_jsonl, write_jsonl_atomic
from emoharness.data.label_set import LabelSet
from emoharness.data.tasks import Kind, Split, Task

MetadataValue = str | int | float | bool | None


class Example(BaseModel):
    """One item. Benchmark items carry ``task``; external items carry ``kind``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str = Field(min_length=1)
    source_id: str = Field(
        min_length=1, description='speaker, dialogue, or video id used for splitting'
    )
    origin: str = Field(min_length=1, description="'mme_emotion', 'meld', 'mosei', ...")
    task: Task | None = None
    kind: Kind | None = None
    split: Split
    media_path: str = Field(min_length=1)
    media_hash: str | None = None
    prompt: str = Field(min_length=1)
    label_set: tuple[str, ...] = Field(min_length=2)
    labels: tuple[str, ...] = Field(min_length=1, description='gold labels, verbatim')
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    metadata: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @property
    def is_benchmark(self) -> bool:
        """True for MME-Emotion items, which may only be used at test time."""
        return self.task is not None


def label_violations(example: Example) -> tuple[str, ...]:
    """Gold labels that are absent from the candidate set after normalization."""
    candidates = LabelSet.from_labels(example.label_set)
    return tuple(
        label for label in example.labels if candidates.canonical(label) is None
    )


def read_manifest(path: Path) -> tuple[Example, ...]:
    """Load a JSONL manifest."""
    return tuple(Example.model_validate(row) for row in iter_jsonl(path))


def write_manifest(path: Path, examples: Iterable[Example]) -> None:
    """Write a JSONL manifest atomically."""
    write_jsonl_atomic(path, (example.model_dump(mode='json') for example in examples))
