"""Benchmark tasks, capability groups, seen capability kinds, and splits."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Final


class Task(str, Enum):
    """The eight MME-Emotion task families, named as in the dataset files."""

    ER_LAB = 'ER_Lab'
    ER_WILD = 'ER_Wild'
    SA = 'SA'
    NOISE_ER = 'Noise_ER'
    FG_ER = 'FG_ER'
    ML_ER = 'ML_ER'
    FG_SA = 'FG_SA'
    IR = 'IR'


class Group(str, Enum):
    """Whether a task's kind was available during harness construction."""

    SEEN = 'seen'
    HELD_OUT = 'held_out'


class Kind(str, Enum):
    """Seen capability kinds; external construction data is labeled with these."""

    LAB_ER = 'lab_er'
    WILD_ER = 'wild_er'
    SENTIMENT = 'sentiment_3'


class Split(str, Enum):
    """Where an example may be used. ``TEST`` is reserved for benchmark items."""

    BUILD = 'build'
    VAL = 'val'
    EXT_TEST = 'ext_test'
    TEST = 'test'


TASK_GROUP: Final[Mapping[Task, Group]] = {
    Task.ER_LAB: Group.SEEN,
    Task.ER_WILD: Group.SEEN,
    Task.SA: Group.SEEN,
    Task.NOISE_ER: Group.HELD_OUT,
    Task.FG_ER: Group.HELD_OUT,
    Task.ML_ER: Group.HELD_OUT,
    Task.FG_SA: Group.HELD_OUT,
    Task.IR: Group.HELD_OUT,
}

TASK_KIND: Final[Mapping[Task, Kind]] = {
    Task.ER_LAB: Kind.LAB_ER,
    Task.ER_WILD: Kind.WILD_ER,
    Task.SA: Kind.SENTIMENT,
}

SEEN_TASKS: Final[tuple[Task, ...]] = tuple(
    t for t, g in TASK_GROUP.items() if g is Group.SEEN
)
HELD_OUT_TASKS: Final[tuple[Task, ...]] = tuple(
    t for t, g in TASK_GROUP.items() if g is Group.HELD_OUT
)
MULTI_LABEL_TASKS: Final[tuple[Task, ...]] = (Task.FG_ER, Task.ML_ER)
