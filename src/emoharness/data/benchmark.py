"""MME-Emotion benchmark ingestion (docs/benchmark-facts.md, ADR-0003).

The HF dataset ships one JSON annotation file per task,
``<root>/<Task.value>.json``, each a list of
``{"video_id": ..., "label": ..., "label_set": [...]}`` objects, plus the
referenced video files at ``<root>/<video_id>``. This module never reads the
official judge's audio evidence directory shipped alongside the videos
(ADR-0004): only the per-task JSON files, ``*.zip`` archives, and video paths
named by ``video_id`` are touched.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file
from emoharness.common.io import read_json
from emoharness.data.label_set import normalize_label, split_labels
from emoharness.data.manifest import Example, label_violations
from emoharness.data.prompt_template import render_prompt_for_task
from emoharness.data.tasks import Split, Task

_ALL_TASKS: Final[tuple[Task, ...]] = tuple(Task)
# The official judge's audio evidence directory name; never read from here (ADR-0004).
_AUDIO_EVIDENCE_DIR: Final = 'audio_clue'
_DIA_UTT_RE: Final = re.compile(r'dia_(\d+)_utt_(\d+)')
_IEMOCAP_RE: Final = re.compile(r'^Ses\d{2}[A-Za-z]_')
_TRAILING_INT_RE: Final = re.compile(r'^(?P<id>.+)_(?P<n>\d+)$')


class RawAnnotation(BaseModel):
    """One row of a per-task MME-Emotion annotation file, verbatim."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    video_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    label_set: tuple[str, ...] = Field(min_length=2)


class BenchmarkReport(BaseModel):
    """Summary statistics over a built benchmark manifest."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    task_counts: Mapping[Task, int]
    label_histograms: Mapping[Task, Mapping[str, int]]
    label_violation_counts: Mapping[Task, int]
    missing_media_count: int = Field(ge=0)


def load_task_annotations(root: Path, task: Task) -> tuple[RawAnnotation, ...]:
    """Load and validate every annotation row for one task."""
    path = root / f'{task.value}.json'
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValidationError(f'{path}: expected a JSON list of annotations')
    return tuple(_parse_annotation(path, index, item) for index, item in enumerate(raw))


def _parse_annotation(path: Path, index: int, item: object) -> RawAnnotation:
    try:
        return RawAnnotation.model_validate(item)
    except PydanticValidationError as exc:
        raise ValidationError(f'{path}[{index}]: invalid annotation: {exc}') from exc


_UNSAFE_ID_CHARS = re.compile(r'[^A-Za-z0-9_-]')


def safe_stem(video_id: str) -> str:
    """File stem with every character outside ``[A-Za-z0-9_-]`` replaced by ``_``."""
    return _UNSAFE_ID_CHARS.sub('_', Path(video_id).stem)


def derive_source_id(video_id: str) -> str:
    """Group a benchmark ``video_id`` by its underlying recording.

    Rules, in priority order:

    1. ``dia_<d>_utt_<u>...`` (MELD-derived sub-sources) -> ``dia_<d>``.
    2. IEMOCAP-style ``Ses05M_script01_1_F034`` -> ``Ses05M_script01_1`` (drop
       the trailing speaker/utterance tag).
    3. YouTube-like ``<id>_<n>`` where ``n`` is a trailing integer -> ``<id>``.
    4. Otherwise, the file stem unchanged.

    Rule 3 is a catch-all and may over-merge stems that happen to end in an
    underscore and digits without being YouTube segments; there is no ground
    truth to disambiguate this without the actual dataset.
    """
    stem = Path(video_id).stem
    dia_utt = _DIA_UTT_RE.search(stem)
    if dia_utt is not None:
        return f'dia_{dia_utt.group(1)}'
    if _IEMOCAP_RE.match(stem) is not None:
        parts = stem.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:-1])
    trailing_int = _TRAILING_INT_RE.match(stem)
    if trailing_int is not None:
        return trailing_int.group('id')
    return stem


def to_example(
    root: Path,
    task: Task,
    annotation: RawAnnotation,
    *,
    hash_media: bool = False,
) -> Example:
    """Build the canonical ``Example`` for one raw annotation.

    ``example_id`` derives from the video file stem, so ids stay stable when the
    annotation file is reordered; only a changed file name changes an id.
    """
    media = root / annotation.video_id
    return Example(
        example_id=f'mme_{task.value.lower()}_{safe_stem(annotation.video_id)}',
        source_id=derive_source_id(annotation.video_id),
        origin='mme_emotion',
        task=task,
        split=Split.TEST,
        media_path=str(media),
        media_hash=_hash_if_requested(media, hash_media),
        prompt=render_prompt_for_task(task, annotation.label_set),
        label_set=annotation.label_set,
        labels=split_labels(annotation.label),
    )


def _hash_if_requested(media: Path, hash_media: bool) -> str | None:
    if not hash_media:
        return None
    try:
        return sha256_file(media)
    except OSError as exc:
        raise ValidationError(f'cannot hash media {media}: {exc}') from exc


def build_benchmark_manifest(
    root: Path, tasks: tuple[Task, ...] = _ALL_TASKS, *, hash_media: bool = False
) -> tuple[Example, ...]:
    """Load every task's annotations and build the benchmark manifest."""
    return tuple(
        to_example(root, task, annotation, hash_media=hash_media)
        for task in tasks
        for annotation in load_task_annotations(root, task)
    )


def annotation_report(examples: Iterable[Example]) -> BenchmarkReport:
    """Summarize a benchmark manifest: counts, label histograms, data quality."""
    rows = tuple(e for e in examples if e.task is not None)
    tasks_seen = tuple(dict.fromkeys(cast(Task, e.task) for e in rows))
    return BenchmarkReport(
        task_counts={task: _count_for(rows, task) for task in tasks_seen},
        label_histograms={task: _label_histogram(rows, task) for task in tasks_seen},
        label_violation_counts={
            task: _violations_for(rows, task) for task in tasks_seen
        },
        missing_media_count=sum(1 for e in rows if not Path(e.media_path).exists()),
    )


def _count_for(rows: tuple[Example, ...], task: Task) -> int:
    return sum(1 for e in rows if e.task is task)


def _label_histogram(rows: tuple[Example, ...], task: Task) -> dict[str, int]:
    counter = Counter(
        normalize_label(label) for e in rows if e.task is task for label in e.labels
    )
    return dict(counter)


def _violations_for(rows: tuple[Example, ...], task: Task) -> int:
    return sum(1 for e in rows if e.task is task and label_violations(e))


def checksum_manifest(root: Path) -> Mapping[str, str]:
    """SHA-256 of every ``*.json`` and ``*.zip`` file under ``root``.

    Recurses, but skips the official judge's audio evidence tree wherever it
    is nested, matching the "never read" rule for that directory.
    """
    paths = sorted(
        path
        for pattern in ('*.json', '*.zip')
        for path in root.rglob(pattern)
        if _AUDIO_EVIDENCE_DIR not in path.relative_to(root).parts
    )
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
