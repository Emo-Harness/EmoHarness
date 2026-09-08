"""Construction-vs-benchmark leakage checks (ADR-0003): four independent gates.

One way in, one way out: gather every id, hash, pHash, and transcript into a
:class:`LeakageInputs`, call :func:`run_leakage_checks`, get back one
:class:`LeakageReport`. A clip is dropped from construction if any check fires.

``phash_image`` is the only place ``PIL`` and ``imagehash`` are needed, and it
imports both lazily inside the function (as does the private ``_hamming``
helper), so importing this module never pulls in the image stack (candidate 6,
docs/architecture-review-2026-09-07.md): a ``--fake`` run that never computes a
pHash never pays for them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import (
    PHASH_HAMMING_THRESHOLD,
    TRANSCRIPT_JACCARD_THRESHOLD,
    TRANSCRIPT_MIN_WORDS,
)
from emoharness.common.errors import ValidationError
from emoharness.common.io import write_json_atomic
from emoharness.data.manifest import Example
from emoharness.data.tasks import Task

CheckName = Literal['media_hash', 'phash', 'transcript', 'source_id']

DEFAULT_SOURCE_ID_RULES: Final[Mapping[str, tuple[Task, ...]]] = {
    'meld': (Task.ER_WILD, Task.SA, Task.IR),
    'meld_sentiment': (Task.ER_WILD, Task.SA, Task.IR),
    'mosei': (Task.SA, Task.FG_SA),
    'chsims': (Task.SA,),
    'iemocap': (Task.ER_LAB,),
}
_WORD_RE: Final = re.compile(r'[^\W_]+')
_V = TypeVar('_V')


class LeakageFlag(BaseModel):
    """One flagged construction/benchmark pair and the check that fired."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    construction_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    check: CheckName
    score: float = Field(ge=0.0, le=1.0)


class LeakageReport(BaseModel):
    """The combined result of every leakage check run in one sweep."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    flags: tuple[LeakageFlag, ...]
    checked_pairs: int = Field(ge=0)
    construction_count: int = Field(ge=0)
    benchmark_count: int = Field(ge=0)
    thresholds: Mapping[str, float]

    def flagged_construction_ids(self) -> frozenset[str]:
        """Every construction id flagged by at least one check."""
        return frozenset(flag.construction_id for flag in self.flags)


class LeakageThresholds(BaseModel):
    """Threshold knobs for the pHash and transcript checks (ADR-0003)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    phash_hamming: int = Field(default=PHASH_HAMMING_THRESHOLD, ge=0)
    transcript_jaccard: float = Field(
        default=TRANSCRIPT_JACCARD_THRESHOLD, ge=0.0, le=1.0
    )
    transcript_min_words: int = Field(default=TRANSCRIPT_MIN_WORDS, ge=0)


class LeakageInputs(BaseModel):
    """Everything one leakage sweep needs, gathered up front (ADR-0003).

    ``media_hashes``, ``phashes``, and ``transcripts`` are keyed by example id
    across both sides; an id missing from one of them simply has nothing to
    compare for that one check.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    construction: tuple[Example, ...]
    benchmark: tuple[Example, ...]
    media_hashes: Mapping[str, str]
    phashes: Mapping[str, tuple[str, ...]]
    transcripts: Mapping[str, str]
    source_id_rules: Mapping[str, tuple[Task, ...]] = DEFAULT_SOURCE_ID_RULES


def _by_id(mapping: Mapping[str, _V], ids: frozenset[str]) -> dict[str, _V]:
    """The subset of ``mapping`` whose key is in ``ids``."""
    return {key: value for key, value in mapping.items() if key in ids}


def _check_media_hashes(
    construction: Mapping[str, str], benchmark: Mapping[str, str]
) -> tuple[LeakageFlag, ...]:
    """Flag exact SHA-256 matches between construction and benchmark media."""
    by_hash: dict[str, list[str]] = {}
    for benchmark_id, digest in benchmark.items():
        by_hash.setdefault(digest, []).append(benchmark_id)
    return tuple(
        LeakageFlag(
            construction_id=construction_id,
            benchmark_id=benchmark_id,
            check='media_hash',
            score=1.0,
        )
        for construction_id, digest in construction.items()
        for benchmark_id in by_hash.get(digest, ())
    )


def phash_image(path: Path) -> str:
    """Perceptual hash (``imagehash.phash``) of an image frame, as a hex string."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image))
    except (OSError, ValueError) as exc:
        raise ValidationError(f'cannot compute phash for {path}: {exc}') from exc


def _hamming(hex_a: str, hex_b: str) -> int:
    """Hamming distance between two ``imagehash``-style hex hash strings."""
    import imagehash

    return int(imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b))


def _check_phashes(
    construction: Mapping[str, tuple[str, ...]],
    benchmark: Mapping[str, tuple[str, ...]],
    threshold: int,
) -> tuple[LeakageFlag, ...]:
    """Flag a construction/benchmark pair if any of their frame pHashes match.

    Brute force over every (construction id, benchmark id, frame, frame)
    combination: O(|construction| * |benchmark| * frames_c * frames_b) hamming
    comparisons. This matches ADR-0003's "all pairs" method and is fine at the
    corpus sizes it specifies (hundreds to low thousands of clips, up to
    ~10 frames each); bucket hashes by prefix first if this ever needs to run
    at a larger scale.
    """
    candidates = (
        _phash_flag(c_id, c_hashes, b_id, b_hashes, threshold)
        for c_id, c_hashes in construction.items()
        for b_id, b_hashes in benchmark.items()
    )
    return tuple(flag for flag in candidates if flag is not None)


def _phash_flag(
    construction_id: str,
    c_hashes: tuple[str, ...],
    benchmark_id: str,
    b_hashes: tuple[str, ...],
    threshold: int,
) -> LeakageFlag | None:
    best = _min_hamming(c_hashes, b_hashes)
    if best is None or best > threshold:
        return None
    bits = len(c_hashes[0]) * 4
    return LeakageFlag(
        construction_id=construction_id,
        benchmark_id=benchmark_id,
        check='phash',
        score=1 - best / bits,
    )


def _min_hamming(a_hashes: tuple[str, ...], b_hashes: tuple[str, ...]) -> int | None:
    distances = [_hamming(a, b) for a in a_hashes for b in b_hashes]
    return min(distances) if distances else None


def _normalize_transcript(text: str) -> tuple[str, ...]:
    """Lowercase word tokens, with punctuation stripped."""
    return tuple(_WORD_RE.findall(text.lower()))


def _check_transcripts(
    construction: Mapping[str, str],
    benchmark: Mapping[str, str],
    threshold: float,
    min_words: int,
) -> tuple[LeakageFlag, ...]:
    """Flag construction/benchmark transcript pairs by token Jaccard.

    Clips whose normalized transcript has fewer than ``min_words`` tokens are
    never compared: short transcripts are too easy to match by chance.
    """
    c_tokens = {
        cid: frozenset(_normalize_transcript(text))
        for cid, text in construction.items()
    }
    b_tokens = {
        bid: frozenset(_normalize_transcript(text)) for bid, text in benchmark.items()
    }
    candidates = (
        _transcript_flag(cid, c_set, bid, b_set, threshold, min_words)
        for cid, c_set in c_tokens.items()
        for bid, b_set in b_tokens.items()
    )
    return tuple(flag for flag in candidates if flag is not None)


def _transcript_flag(
    construction_id: str,
    c_set: frozenset[str],
    benchmark_id: str,
    b_set: frozenset[str],
    threshold: float,
    min_words: int,
) -> LeakageFlag | None:
    if len(c_set) < min_words or len(b_set) < min_words:
        return None
    union = c_set | b_set
    if not union:
        return None
    jaccard = len(c_set & b_set) / len(union)
    if jaccard < threshold:
        return None
    return LeakageFlag(
        construction_id=construction_id,
        benchmark_id=benchmark_id,
        check='transcript',
        score=jaccard,
    )


def _check_source_ids(
    construction_examples: tuple[Example, ...],
    benchmark_examples: tuple[Example, ...],
    rules: Mapping[str, tuple[Task, ...]],
) -> tuple[LeakageFlag, ...]:
    """Flag construction/benchmark pairs that share a source id under ``rules``.

    ``rules`` maps a construction ``origin`` to the benchmark tasks it must be
    compared against, per ADR-0003's source-id leakage table.
    """
    index = _index_benchmark_by_task_and_source(benchmark_examples)
    return tuple(
        flag
        for construction_example in construction_examples
        for flag in _source_id_flags(construction_example, index, rules)
    )


def _index_benchmark_by_task_and_source(
    benchmark_examples: tuple[Example, ...],
) -> Mapping[Task, Mapping[str, tuple[str, ...]]]:
    index: dict[Task, dict[str, list[str]]] = {}
    for example in benchmark_examples:
        if example.task is None:
            continue
        by_source = index.setdefault(example.task, {})
        by_source.setdefault(example.source_id, []).append(example.example_id)
    return {
        task: {sid: tuple(ids) for sid, ids in by_source.items()}
        for task, by_source in index.items()
    }


def _source_id_flags(
    construction_example: Example,
    index: Mapping[Task, Mapping[str, tuple[str, ...]]],
    rules: Mapping[str, tuple[Task, ...]],
) -> tuple[LeakageFlag, ...]:
    tasks = rules.get(construction_example.origin, ())
    return tuple(
        LeakageFlag(
            construction_id=construction_example.example_id,
            benchmark_id=benchmark_id,
            check='source_id',
            score=1.0,
        )
        for task in tasks
        for benchmark_id in index.get(task, {}).get(construction_example.source_id, ())
    )


def run_leakage_checks(
    inputs: LeakageInputs,
    thresholds: LeakageThresholds = LeakageThresholds(),  # noqa: B008 -- frozen, safe to share
) -> LeakageReport:
    """Run every check and combine the flags into one report.

    ``inputs``'s evidence mappings cover both sides by example id; this splits
    each one by ``inputs.construction`` / ``inputs.benchmark`` before handing
    it to the check that needs it, so a caller only ever gathers evidence once.
    """
    c_ids = frozenset(example.example_id for example in inputs.construction)
    b_ids = frozenset(example.example_id for example in inputs.benchmark)
    flags = (
        _check_media_hashes(
            _by_id(inputs.media_hashes, c_ids), _by_id(inputs.media_hashes, b_ids)
        )
        + _check_phashes(
            _by_id(inputs.phashes, c_ids),
            _by_id(inputs.phashes, b_ids),
            thresholds.phash_hamming,
        )
        + _check_transcripts(
            _by_id(inputs.transcripts, c_ids),
            _by_id(inputs.transcripts, b_ids),
            thresholds.transcript_jaccard,
            thresholds.transcript_min_words,
        )
        + _check_source_ids(
            inputs.construction, inputs.benchmark, inputs.source_id_rules
        )
    )
    return LeakageReport(
        flags=flags,
        checked_pairs=len(inputs.construction) * len(inputs.benchmark),
        construction_count=len(inputs.construction),
        benchmark_count=len(inputs.benchmark),
        thresholds={
            'phash_hamming': float(thresholds.phash_hamming),
            'transcript_jaccard': thresholds.transcript_jaccard,
            'transcript_min_words': float(thresholds.transcript_min_words),
        },
    )


def drop_flagged(
    examples: tuple[Example, ...], report: LeakageReport
) -> tuple[Example, ...]:
    """Return construction examples with every flagged id removed."""
    flagged = report.flagged_construction_ids()
    return tuple(example for example in examples if example.example_id not in flagged)


def write_report(path: Path, report: LeakageReport) -> None:
    """Write a leakage report to disk atomically."""
    write_json_atomic(path, report.model_dump(mode='json'))
