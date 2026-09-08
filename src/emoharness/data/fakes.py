"""Synthetic corpora in the real on-disk formats, for ``--fake`` runs and tests.

The benchmark generator writes one annotation file per task exactly as
MME-Emotion ships them (``<root>/<Task.value>.json`` holding ``video_id``,
``label``, ``label_set``) plus a placeholder media file per ``video_id``. The
external generator writes the smallest files each ``emoharness.data.sources``
adapter knows how to read. Nothing here is a fixture for one test: both are the
inputs the real entry points consume, which is why they live beside the readers
they feed.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from emoharness.common.hashing import sha256_text
from emoharness.common.io import write_json_atomic
from emoharness.data.sources.base import SENTIMENT_LABELS
from emoharness.data.sources.meld import EMOTION_LABELS as MELD_LABELS
from emoharness.data.tasks import Task

DEFAULT_PER_TASK: Final = 6
DEFAULT_PER_SOURCE: Final = 4
DEFAULT_SEED: Final = 20260907
MEDIA_BYTES: Final = 96

#: Candidate lists in the spirit of each MME-Emotion task (docs/benchmark-facts.md).
TASK_LABELS: Final[Mapping[Task, tuple[str, ...]]] = {
    Task.ER_LAB: (
        'anger',
        'disgust',
        'fear',
        'happiness',
        'neutral',
        'sadness',
        'surprise',
    ),
    Task.ER_WILD: (
        'anger',
        'disgust',
        'fear',
        'joy',
        'neutral',
        'sadness',
        'surprise',
    ),
    Task.SA: SENTIMENT_LABELS,
    Task.NOISE_ER: ('anger', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'),
    Task.FG_ER: (
        'amusement',
        'contentment',
        'disappointment',
        'embarrassment',
        'frustration',
        'pride',
        'relief',
    ),
    Task.ML_ER: (
        'anger',
        'anxiety',
        'contempt',
        'disgust',
        'fear',
        'helplessness',
        'sadness',
    ),
    Task.FG_SA: (
        'strongly positive',
        'weakly positive',
        'neutral',
        'weakly negative',
        'strongly negative',
    ),
    Task.IR: (
        'agreeing',
        'complaining',
        'joking',
        'questioning',
        'reassuring',
        'refusing',
    ),
}

#: ``video_id`` stem patterns, chosen so ``derive_source_id`` groups them the way
#: the real sub-sources do (dialogue id, session, YouTube id).
_STEM_PATTERNS: Final[Mapping[Task, str]] = {
    Task.ER_LAB: 'Ses0{group}F_impro0{group}_F{index:03d}',
    Task.ER_WILD: 'dia_{group}_utt_{index}',
    Task.SA: 'ytclip{group}_{index}',
    Task.NOISE_ER: 'noise{group}_{index}',
    Task.FG_ER: 'fgclip{group}_{index}',
    Task.ML_ER: 'mlclip{group}_{index}',
    Task.FG_SA: 'fgsaclip{group}_{index}',
    Task.IR: 'irclip{group}_{index}',
}

_MULTI_SEPARATORS: Final[Mapping[Task, str]] = {Task.FG_ER: ';', Task.ML_ER: ','}


def write_media(path: Path, key: str) -> Path:
    """Write a placeholder media file whose bytes are unique to ``key``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_text(key).encode('ascii')
    body = (digest * (MEDIA_BYTES // len(digest) + 1))[:MEDIA_BYTES]
    path.write_bytes(b'FAKE' + body)
    return path


def _gold_for(task: Task, labels: tuple[str, ...], rng: random.Random) -> str:
    separator = _MULTI_SEPARATORS.get(task)
    if separator is None:
        return rng.choice(labels)
    count = rng.randint(2, 3) if task is Task.ML_ER else rng.randint(1, 2)
    return separator.join(rng.sample(labels, count))


def _annotation(task: Task, index: int, rng: random.Random) -> dict[str, object]:
    labels = TASK_LABELS[task]
    stem = _STEM_PATTERNS[task].format(group=index % 3, index=index)
    return {
        'video_id': f'{task.value}/{stem}.mp4',
        'label': _gold_for(task, labels, rng),
        'label_set': list(labels),
    }


def make_fake_benchmark(
    root: Path, per_task: int = DEFAULT_PER_TASK, seed: int = DEFAULT_SEED
) -> Path:
    """Write a synthetic MME-Emotion tree under ``root`` and return it.

    ``per_task`` rows are written for each of the eight tasks, with the gold
    string separated by ``;`` for FG-ER and ``,`` for ML-ER, matching the
    benchmark's own multi-label convention. Media are placeholder files whose
    bytes differ per ``video_id`` so media-hash leakage checks are meaningful.
    """
    root.mkdir(parents=True, exist_ok=True)
    for task in Task:
        rng = random.Random(f'{seed}:{task.value}')  # noqa: S311 -- reproducible
        rows = [_annotation(task, index, rng) for index in range(1, per_task + 1)]
        write_json_atomic(root / f'{task.value}.json', rows)
        for row in rows:
            write_media(root / str(row['video_id']), str(row['video_id']))
    return root


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _meld_rows(
    split: str, dialogues: int, per_dialogue: int, rng: random.Random
) -> tuple[dict[str, object], ...]:
    rows = []
    number = 1
    for dialogue in range(dialogues):
        for utterance in range(per_dialogue):
            emotion = rng.choice(MELD_LABELS)
            rows.append(
                {
                    'Sr No.': number,
                    'Utterance': f'{split} line {number}',
                    'Speaker': f'Speaker{dialogue}',
                    'Emotion': emotion,
                    'Sentiment': rng.choice(SENTIMENT_LABELS),
                    'Dialogue_ID': dialogue,
                    'Utterance_ID': utterance,
                }
            )
            number += 1
    return tuple(rows)


def _write_meld(root: Path, per_source: int, seed: int) -> None:
    columns = (
        'Sr No.',
        'Utterance',
        'Speaker',
        'Emotion',
        'Sentiment',
        'Dialogue_ID',
        'Utterance_ID',
    )
    for split, dialogues in (('train', per_source * 2), ('dev', per_source)):
        rng = random.Random(f'{seed}:meld:{split}')  # noqa: S311 -- reproducible
        rows = _meld_rows(split, dialogues, 2, rng)
        _write_csv(root / f'{split}_sent_emo.csv', columns, rows)
        for row in rows:
            name = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4'
            write_media(root / f'{split}_splits' / name, f'meld/{split}/{name}')


def _write_mosei(root: Path, per_source: int, seed: int) -> None:
    rng = random.Random(f'{seed}:mosei')  # noqa: S311 -- reproducible
    rows: list[dict[str, object]] = []
    for video in range(per_source * 2):
        for segment in range(2):
            relative = f'clips/mosei_vid{video}_{segment}.mp4'
            rows.append(
                {
                    'video_id': f'mosei_vid{video}',
                    'segment_id': str(segment),
                    'sentiment': round(rng.uniform(-3.0, 3.0), 3),
                    'path': relative,
                }
            )
            write_media(root / relative, f'mosei/{relative}')
    _write_csv(
        root / 'mosei_labels.csv', ('video_id', 'segment_id', 'sentiment', 'path'), rows
    )


def _write_chsims(root: Path, per_source: int, seed: int) -> None:
    rng = random.Random(f'{seed}:chsims')  # noqa: S311 -- reproducible
    rows: list[dict[str, object]] = []
    for video in range(per_source * 2):
        for clip in range(2):
            video_id, clip_id = f'sims_vid{video}', f'{clip:04d}'
            rows.append(
                {
                    'video_id': video_id,
                    'clip_id': clip_id,
                    'label': round(rng.uniform(-1.0, 1.0), 3),
                }
            )
            write_media(
                root / 'Raw' / video_id / f'{clip_id}.mp4',
                f'chsims/{video_id}/{clip_id}',
            )
    _write_csv(root / 'label.csv', ('video_id', 'clip_id', 'label'), rows)


def _write_cremad(root: Path, per_source: int, seed: int) -> None:
    rng = random.Random(f'{seed}:cremad')  # noqa: S311 -- reproducible
    codes = ('ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD')
    for actor in range(per_source * 3):
        for take in range(2):
            code = rng.choice(codes)
            name = f'{1001 + actor}_DFA_{code}_{"XX" if take == 0 else "MD"}.mp4'
            write_media(root / 'VideoFlash' / name, f'cremad/{name}')


def make_fake_external(
    root: Path, per_source: int = DEFAULT_PER_SOURCE, seed: int = DEFAULT_SEED
) -> Mapping[str, Path]:
    """Write tiny MELD, CMU-MOSEI, CH-SIMS, and CREMA-D trees under ``root``.

    Returns the per-origin corpus roots, in the shape ``configs/data/sources.yaml``
    expects. IEMOCAP is deliberately absent: CREMA-D is its declared fallback
    (ADR-0003), so a run with neither licence still exercises the whole path.
    """
    roots = {origin: root / origin for origin in ('meld', 'mosei', 'chsims', 'cremad')}
    _write_meld(roots['meld'], per_source, seed)
    _write_mosei(roots['mosei'], per_source, seed)
    _write_chsims(roots['chsims'], per_source, seed)
    _write_cremad(roots['cremad'], per_source, seed)
    return {**roots, 'meld_sentiment': roots['meld']}
