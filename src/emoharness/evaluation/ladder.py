"""The method ladder: `configs/experiments/ladder.yaml` as typed objects.

Every name a row can carry is a closed set, so a misspelt method, input, item set,
or priority fails when the ladder is loaded rather than halfway through a run that
has already spent Target calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MethodName = Literal[
    'direct', 'direct_audio', 'handcrafted', 'best_of_n', 'harness', 'teacher_direct'
]
#: 'native_video' is only answerable by a Gemini teacher; every other provider
#: takes the matched protocol (sampled frames plus the audio-evidence text).
InputsName = Literal['video', 'video+audio_text', 'native_video', 'matched_protocol']
ItemSetName = Literal['full', 'subset']
Priority = Literal['P0', 'P1', 'P2']


class RowSpec(BaseModel):
    """One ladder row. Optional fields are populated according to `method`."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(
        min_length=1, description="the YAML key, e.g. 'L0' or 'best_of_4'"
    )
    method: MethodName
    item_set: ItemSetName
    priority: Priority
    inputs: InputsName | None = None
    calls: int | None = Field(default=None, ge=1)
    stage: int | None = Field(
        default=None, ge=1, description='Builder stage, for method=harness'
    )
    base: str | None = Field(
        default=None, description='row this one is built on, e.g. best_of_n'
    )
    n: int | None = Field(
        default=None, ge=1, description='sample count, for method=best_of_n'
    )
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class SubsetSpec(BaseModel):
    """The pre-registered subset parameters shared by every 'subset' row."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    size_per_task: int = Field(gt=0)
    seed: int
    stratify: str = Field(min_length=1)


class Ladder(BaseModel):
    """The parsed method ladder: every row, keyed by name, plus the shared subset spec."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    rows: Mapping[str, RowSpec]
    subset: SubsetSpec

    @model_validator(mode='before')
    @classmethod
    def _name_rows(cls, document: Any) -> Any:
        """A row is named by its YAML key; the row body never repeats the name."""
        if not isinstance(document, Mapping):
            return document
        rows = document.get('rows')
        if not isinstance(rows, Mapping):
            return document
        named = {
            name: {**row, 'name': name} if isinstance(row, Mapping) else row
            for name, row in rows.items()
        }
        return {**document, 'rows': named}


def rows_for_priority(ladder: Ladder, priority: Priority) -> tuple[RowSpec, ...]:
    """Rows at a given priority (e.g. 'P0'), in the YAML's declared order."""
    return tuple(row for row in ladder.rows.values() if row.priority == priority)
