#!/usr/bin/env python3
"""The student-specific full-evaluation report (coordination/full_run_plan_20260918.md).

Sixteen arms -- four students x {L0, L1, L3 scaffold, L5 the student's own final
bundle} -- scored on one frozen item set and compared pairwise. The item set is
NOT ``full``: it is the batch-root ``evaluation_items.jsonl``, the 5,507 rows that
survive the media-level isolation audit, and every number here uses that clean
denominator. ``data/manifests/mme_emotion.jsonl`` has 5,577 rows; the 70 removed
ones share a media hash with a construction, validation, or reserved ext_test
item, so scoring them would be scoring the Builder's own training material.

Three things this script refuses to do, because each would quietly produce a
publishable-looking table that means something else:

* Compare arms that scored different item sets. If two arms disagree on which
  ids they cover, that pair is withheld rather than silently evaluated on the
  intersection, which would move the denominator from one row of the table to
  the next. Comparability is judged per pair so that a lagging arm elsewhere
  does not blank the table, but ``--final`` additionally requires every arm to
  cover the exact frozen set: two arms agreeing on the same 4,000 rows would
  otherwise yield a self-consistent table about a set nobody registered.
* Present an L5 arm whose bundle came from somewhere other than this student's
  own construction. The test is lineage -- a path or recorded ``construction_run``
  outside this batch root -- not the bundle hash, since an independent build
  could in principle coincide byte for byte with the 8B transfer bundle.
  ``qwen3vl_8b`` legitimately reuses it and is whitelisted by name.
* Expose D_ext-test. The 129 reserved items live in a different manifest, and
  the three that shared media with the benchmark are already excluded, but the
  check is re-run here and recorded rather than assumed.

Aggregates follow ADR-0013 §2: items paired by id, resampled within their task
stratum, each resample's statistic the macro over strata. The plan fixes the
task-macro as primary and the sample-weighted pooled mean as secondary, so both
are printed side by side. Holm correction (§6) is applied across the arm family
because sixteen paired tests are read off one table.

``--media-cluster-diagnostic`` adds a third interval that resamples whole media
clusters instead of items, enabled automatically under ``--final``. It is a
sensitivity check, not a protocol change: 423 media hashes in the clean set
carry more than one item and 412 of those span two different tasks (an ER_Wild
clip reappearing as an IR item), so item-level resampling treats correlated
observations as independent, which can miscalibrate the interval. Clusters are
drawn globally with joint weights across tasks, and the macro is taken over a
task set fixed before resampling -- a draw missing a task is redrawn rather than
averaged over fewer strata, so the denominator cannot drift between resamples.

Usage:
    scripts/report_student_specific_full.py --resamples 200    # hourly check
    scripts/report_student_specific_full.py --final            # publication gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np

from emoharness.common.ledger import read_entries
from emoharness.data.manifest import Example
from emoharness.data.tasks import TASK_GROUP, Group, Task
from emoharness.evaluation.bootstrap import holm_adjust, paired_bootstrap
from emoharness.evaluation.schema import PredictionRecord
from emoharness.evaluation.scoreboard import Scoreboard

ROOT: Final = Path(__file__).resolve().parent.parent
BATCH: Final = ROOT / 'results/v3/student_specific_full_20260918'
ITEM_SET: Final = 'full5507'
RESAMPLES: Final = 10_000

#: Marker order, which is also ladder order: information, then architecture,
#: then the student's own construction.
SLOTS: Final[tuple[str, ...]] = ('L0', 'L1', 'L3_scaffold', 'L5_own')
MODELS: Final[tuple[str, ...]] = (
    'qwen3vl_8b',
    'qwen3vl_4b',
    'qwen3_5_4b',
    'gemma4_e4b',
)
#: The v3 bundle the previous batch transferred. Only ``qwen3vl_8b`` may reuse
#: it; for anyone else an L5 arm carrying this hash means no construction ran.
TRANSFER_BUNDLE_HASH: Final = (
    '569c76f9edc0e2e6473fdad9f8854fdfa473fcf94964961d48877a338478e9e2'
)
BUNDLE_REUSE_ALLOWED: Final = 'qwen3vl_8b'

#: b - a, and what the difference is evidence about.
COMPARISONS: Final[tuple[tuple[str, str, str], ...]] = (
    ('L1 - L0', 'L0', 'L1'),
    ('L3 - L1', 'L1', 'L3_scaffold'),
    ('L5 - L3', 'L3_scaffold', 'L5_own'),
    ('L5 - L0', 'L0', 'L5_own'),
)
COMPARISON_MEANING: Final[Mapping[str, str]] = {
    'L1 - L0': 'fixed perception information, one call',
    'L3 - L1': 'scaffold architecture over the same information',
    'L5 - L3': "this student's own constructed bundle over the scaffold",
    'L5 - L0': 'total effect over the student’s own direct baseline',
}


# --------------------------------------------------------------------------
# The frozen item set and the strata cut from it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemSet:
    """The clean evaluation items, plus every id partition the report needs.

    ``old`` is the pre-registered 1,000-item development subset intersected with
    the clean set, and ``new`` is everything else. The intersection is the point:
    14 of the 1,000 fall inside the 70 media-excluded ids, so the strata are 986
    and 4,521 rather than a round 1,000 and 4,577. The labels are generated from
    the real counts, so they cannot drift from the data they describe.
    """

    examples: Mapping[str, Example]
    excluded: frozenset[str]
    old: frozenset[str]
    new: frozenset[str]
    items_sha256: str
    excluded_sha256: str

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self.examples)

    @property
    def old_label(self) -> str:
        return f'old_{len(self.old)}'

    @property
    def new_label(self) -> str:
        return f'new_{len(self.new)}'

    def clusters(self) -> Mapping[str, str]:
        """Each item's media cluster, for the clustered diagnostic.

        An item with no media hash is its own cluster: absent evidence of
        sharing, treating it as independent is the conservative reading of the
        item and the honest one about what we know.
        """
        return {
            i: (e.media_hash or f'item:{i}') for i, e in self.examples.items()
        }

    def views(self) -> Mapping[str, frozenset[str]]:
        """The id subsets every arm is summarised over, in report order."""
        return {
            'full': self.ids,
            'seen': self._group(Group.SEEN),
            'held_out': self._group(Group.HELD_OUT),
            self.old_label: self.old,
            self.new_label: self.new,
        }

    def _group(self, group: Group) -> frozenset[str]:
        return frozenset(
            i
            for i, e in self.examples.items()
            if e.task is not None and TASK_GROUP.get(e.task) is group
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_item_set(batch: Path, subset_path: Path) -> ItemSet:
    """Read the batch-root item list, the exclusion list, and the old/new cut."""
    items_path = batch / 'evaluation_items.jsonl'
    excluded_path = batch / 'evaluation_excluded_ids.json'
    examples = {
        example.example_id: example
        for example in (
            Example.model_validate(json.loads(line))
            for line in items_path.read_text().splitlines()
            if line.strip()
        )
    }
    excluded = frozenset(json.loads(excluded_path.read_text())['ids'])
    overlap = excluded & frozenset(examples)
    if overlap:
        raise SystemExit(
            f'{len(overlap)} excluded ids are still present in {items_path}: '
            f'{sorted(overlap)[:5]}'
        )
    subset = frozenset(json.loads(subset_path.read_text())['ids'])
    old = subset & frozenset(examples)
    return ItemSet(
        examples=examples,
        excluded=excluded,
        old=old,
        new=frozenset(examples) - old,
        items_sha256=_sha256(items_path),
        excluded_sha256=_sha256(excluded_path),
    )


def reserved_media_check(item_set: ItemSet, external: Path) -> dict:
    """Re-run the isolation audit's conclusion: no construction media may appear.

    This fails closed. It is the one assumption whose failure would invalidate
    every number in the report, so a nonzero overlap *or* a missing manifest is
    a blocking condition, not a diagnostic line someone might skim past.
    """
    if not external.is_file():
        return {
            'ok': False,
            'blocking': f'cannot verify isolation: {external} is missing',
            'counts': {},
        }
    scored = {e.media_hash for e in item_set.examples.values() if e.media_hash}
    by_split: dict[str, set[str]] = defaultdict(set)
    for line in external.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('media_hash'):
            by_split[row['split']].add(row['media_hash'])
    counts = {split: len(hashes & scored) for split, hashes in by_split.items()}
    leaked = {split: n for split, n in counts.items() if n}
    return {
        'ok': not leaked,
        'blocking': (
            None
            if not leaked
            else f'construction media present in the scored set: {leaked}'
        ),
        'counts': counts,
    }


#: How ``execution_manifest.json`` spells the two digests this report depends
#: on. Aliases are listed because the manifest is the coordinator's document,
#: not this script's, and it should be free to rename a key without silently
#: turning the check off.
ITEM_SET_DIGEST_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    'evaluation_items.jsonl': ('evaluation_sha256', 'evaluation_items_sha256'),
    'evaluation_excluded_ids.json': (
        'exclusions_sha256',
        'evaluation_excluded_ids_sha256',
        'excluded_ids_sha256',
    ),
}


def manifest_hash_check(batch: Path, item_set: ItemSet) -> dict:
    """Confirm the item list and exclusion list match what the batch declared.

    The denominator is only trustworthy if the file this report read is the file
    the run was executed against, so the digests are compared against
    ``execution_manifest.json`` rather than merely recorded. Fails closed: a
    missing manifest, an undeclared digest, or a mismatch all block a final
    report, because each leaves the item set unverified.
    """
    path = batch / 'execution_manifest.json'
    if not path.is_file():
        return {
            'ok': False,
            'blocking': f'cannot verify item-set provenance: {path} is missing',
            'checked': {},
        }
    try:
        manifest = json.loads(path.read_text())
    except ValueError as exc:
        return {
            'ok': False,
            'blocking': f'{path.name} is not readable JSON: {exc}',
            'checked': {},
        }
    actual = {
        'evaluation_items.jsonl': item_set.items_sha256,
        'evaluation_excluded_ids.json': item_set.excluded_sha256,
    }
    checked: dict[str, str] = {}
    problems: dict[str, str] = {}
    for filename, keys in ITEM_SET_DIGEST_KEYS.items():
        declared = next(
            (manifest[key] for key in keys if _is_digest(manifest.get(key))), None
        )
        if declared is None:
            problems[filename] = (
                'no digest declared under ' + ' or '.join(keys)
            )
        elif declared != actual[filename]:
            problems[filename] = (
                f'declared {declared[:16]}..., read {actual[filename][:16]}...'
            )
        else:
            checked[filename] = declared[:16] + '...'
    return {
        'ok': not problems,
        'blocking': None if not problems else f'item-set hash mismatch: {problems}',
        'checked': checked,
        'declared_counts': {
            key: manifest.get(key)
            for key in ('n_canonical_full', 'n_full', 'n_excluded', 'n_subset', 'n_new')
            if key in manifest
        },
    }


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in '0123456789abcdef' for c in value.lower())
    )


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One student on one ladder rung: its scores, its cost, and its provenance.

    Deviations are split by what they actually cost. ``fatal`` means the arm is
    not a usable result and every difference touching it is withheld -- an L5
    built from someone else's bundle answers a different question than the one
    asked. ``warnings`` means something is wrong with the run but the scores
    still stand: a prediction for an excluded id is dropped before scoring, so
    it is a driver bug to shout about, not a reason to blank the table.
    """

    model: str
    slot: str
    board: Scoreboard
    marker: Mapping[str, object]
    meta: Mapping[str, object]
    fatal: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def problems(self) -> tuple[str, ...]:
        return (*(f'FATAL: {p}' for p in self.fatal), *self.warnings)


def _prediction_path(marker: Mapping[str, object]) -> Path:
    """Where the arm's predictions live.

    ``predictions`` is the field asked of the driver; without it the path is
    reconstructed the way the transfer batch laid it out, which is only correct
    while a slot's directory name and its ``row`` agree.
    """
    declared = marker.get('predictions')
    if isinstance(declared, str) and declared:
        return ROOT / declared
    results_dir = ROOT / str(marker['results_dir'])
    return results_dir / 'raw_predictions' / str(marker['row']) / f'{ITEM_SET}.jsonl'


def _actual_target_calls(marker: Mapping[str, object]) -> int:
    """Uncached student calls, from the ledger rather than from the predictions."""
    ledger = ROOT / str(marker['results_dir']) / 'ledger.jsonl'
    if not ledger.is_file():
        return 0
    try:
        entries = read_entries(ledger)
    except (OSError, ValueError):
        return 0
    return sum(
        getattr(entry, 'calls', 1) for entry in entries if entry.role == 'target'
    )


def _bundle_problems(
    model: str, marker: Mapping[str, object], batch: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Whether an L5 arm's bundle came from this student's own construction.

    Returns ``(fatal, warnings)``. The test is *lineage*, not the hash: a
    genuinely independent construction could in principle land on the same
    bundle bytes as the 8B transfer bundle and hash identically, so a hash
    match on its own is suspicious rather than disqualifying. What actually
    disqualifies an arm is a bundle that demonstrably came from somewhere else
    -- a path outside this batch root, or a recorded lineage pointing at the
    previous transfer run.
    """
    if marker.get('slot') != 'L5_own' and marker.get('row') != 'L5':
        return (), ()
    if model == BUNDLE_REUSE_ALLOWED:
        return (), ()
    fatal: list[str] = []
    warnings: list[str] = []
    bundle = str(marker.get('bundle') or '')
    lineage = str(_provenance(marker, 'construction_run') or '')
    if bundle and not _within(bundle, batch):
        fatal.append(
            f'L5 bundle path is outside this batch root, so it was not built '
            f'for this student: {bundle}'
        )
    if lineage and not _within(lineage, batch):
        fatal.append(f'L5 construction lineage points outside this batch: {lineage}')
    if _provenance(marker, 'bundle_hash') == TRANSFER_BUNDLE_HASH and not fatal:
        warnings.append(
            'L5 bundle_hash equals the 8B transfer bundle while its path and '
            'lineage are inside this batch. An independent build can coincide '
            'byte for byte, so this is not proof of reuse, but confirm the '
            'construction actually ran for this student before publishing'
        )
    if not bundle and not lineage:
        warnings.append(
            'L5 marker records neither a bundle path nor a construction_run, '
            'so its lineage cannot be verified from the marker alone'
        )
    return tuple(fatal), tuple(warnings)


def lineage_verified(model: str, marker: Mapping[str, object], batch: Path) -> bool:
    """Whether this L5 arm's construction lineage can actually be checked.

    A bundle path or ``construction_run`` inside the batch root is evidence the
    build happened here. Absence of both is not evidence of wrongdoing, but it
    is also not provenance, so a final publish cannot rest on it.
    """
    if model == BUNDLE_REUSE_ALLOWED:
        return True
    return any(
        _within(str(marker.get(key) or _provenance(marker, key) or ''), batch)
        for key in ('bundle', 'construction_run')
    )


def _within(candidate: str, batch: Path) -> bool:
    """True when a recorded path sits inside the batch root."""
    try:
        Path(candidate).resolve().relative_to(batch.resolve())
    except (ValueError, OSError):
        return False
    return True


def marker_model(marker: Mapping[str, object]) -> str:
    """The student a marker is about.

    The driver spells this field ``student`` and this report grew up calling it
    ``model``; accept either rather than making the two scripts agree on a name
    before either can run.
    """
    for key in ('model', 'student'):
        value = marker.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError('marker names no model/student')


def load_arm(marker: Mapping[str, object], item_set: ItemSet, batch: Path) -> Arm:
    """Score one arm, keeping every deviation as a reported problem, not a crash."""
    model, slot = marker_model(marker), str(marker['slot'])
    lineage_fatal, lineage_warnings = _bundle_problems(model, marker, batch)
    fatal = list(lineage_fatal)
    warnings = list(lineage_warnings)
    path = _prediction_path(marker)
    if not path.is_file():
        return Arm(
            model=model,
            slot=slot,
            board=Scoreboard(items={}, source='local'),
            marker=marker,
            meta={'n': 0, 'predictions': _display_path(path)},
            fatal=(*fatal, f'prediction file missing: {path}'),
            warnings=(),
        )
    records, malformed = _read_predictions(path)
    if malformed:
        fatal.append(
            f'{malformed} prediction lines could not be parsed as '
            'PredictionRecord; the arm is incomplete in an unknown way'
        )
    known = tuple(r for r in records if r.example_id in item_set.examples)
    stray = tuple(r for r in records if r.example_id not in item_set.examples)
    if stray:
        excluded_hits = sum(1 for r in stray if r.example_id in item_set.excluded)
        warnings.append(
            f'{len(stray)} predictions are outside the clean item set '
            f'({excluded_hits} of them are media-excluded ids); they are dropped '
            'before scoring, so the scores below are unaffected, but the driver '
            'iterated the wrong item list'
        )
    contradictory, repeated = _duplicate_ids(known)
    if contradictory:
        fatal.append(
            f'{len(contradictory)} example_ids appear more than once with '
            f'different labels, so which response was scored is arbitrary: '
            f'{sorted(contradictory)[:3]}'
        )
    elif repeated:
        warnings.append(f'{repeated} duplicate example_ids, all in agreement')
    board = Scoreboard.from_predictions(known, item_set.examples)
    seconds = float(marker.get('seconds') or 0.0)
    actual = _actual_target_calls(marker)
    meta = {
        'n': board.count,
        'predictions': _display_path(path),
        'logical_calls': sum(r.target_calls for r in known),
        'actual_calls': actual,
        'tokens': sum(r.input_tokens + r.output_tokens for r in known),
        'parse_failures': sum(r.parse_failures for r in known),
        'cost_usd': round(sum(r.cost_usd for r in known), 4),
        'wall_s': seconds,
        'uncached_calls_per_s': round(actual / seconds, 2) if seconds else None,
        'attempt': marker.get('attempt'),
        'item_set_sha256': _provenance(marker, 'item_set_sha256'),
        'git_rev': _provenance(marker, 'git_rev'),
        'vllm_version': _provenance(marker, 'vllm_version'),
        'bundle': marker.get('bundle'),
    }
    return Arm(
        model=model, slot=slot, board=board, marker=marker,
        meta=meta, fatal=tuple(fatal), warnings=tuple(warnings),
    )


def _read_predictions(path: Path) -> tuple[tuple[PredictionRecord, ...], int]:
    """Parse a prediction file, counting rather than raising on bad lines.

    An hourly report runs against a tree being written to, so one truncated or
    schema-violating line must cost that arm its credibility and nothing else.
    """
    good: list[PredictionRecord] = []
    bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            good.append(PredictionRecord.model_validate(json.loads(line)))
        except (ValueError, TypeError):
            bad += 1
    return tuple(good), bad


def _duplicate_ids(
    records: Sequence[PredictionRecord],
) -> tuple[set[str], int]:
    """Ids written more than once, split by whether the repeats agree.

    Agreeing repeats are noise; disagreeing ones mean the scoreboard silently
    kept whichever line came last.
    """
    labels: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for record in records:
        labels[record.example_id].add(record.labels)
    contradictory = {i for i, seen in labels.items() if len(seen) > 1}
    return contradictory, len(records) - len(labels)


def _display_path(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    A marker is free to point outside the checkout, and one arm doing so must
    not take down a report the run is being supervised with.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _provenance(marker: Mapping[str, object], key: str) -> object:
    provenance = marker.get('provenance')
    if isinstance(provenance, Mapping) and key in provenance:
        return provenance[key]
    return marker.get(key)


def discover(
    batch: Path, item_set: ItemSet
) -> tuple[Mapping[str, Mapping[str, Arm]], list[str]]:
    """Every completed arm under the batch root, keyed model then slot.

    A marker that cannot be read or scored is collected as a defect and the
    sweep continues. This runs hourly against a tree the driver is still
    writing to, so one bad file must not cost the supervisor its whole report.
    """
    arms: dict[str, dict[str, Arm]] = defaultdict(dict)
    defects: list[str] = []
    for marker_path in sorted((batch / '.done').glob(f'*__{ITEM_SET}.done')):
        try:
            marker = json.loads(marker_path.read_text())
            if marker.get('status') not in (None, 'ok'):
                continue
            arm = load_arm(marker, item_set, batch)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            defects.append(f'{marker_path.name}: {type(exc).__name__}: {exc}')
            continue
        previous = arms[arm.model].get(arm.slot)
        if previous is None or _attempt(arm) >= _attempt(previous):
            arms[arm.model][arm.slot] = arm
    return arms, defects


def _attempt(arm: Arm) -> int:
    try:
        return int(arm.marker.get('attempt') or 0)
    except (TypeError, ValueError):
        return 0


def coverage(arms: Mapping[str, Mapping[str, Arm]], item_set: ItemSet) -> dict:
    """Which arms exist, and how each one's coverage compares to the frozen set.

    Two different questions live here. For an hourly check it is enough that
    the two arms in a comparison cover the same ids. For the final main
    comparison that is *not* enough: every arm must cover the exact frozen
    item set, since two arms agreeing on the same 4,000 rows would produce a
    self-consistent table about a set nobody registered.
    """
    expected = [(m, s) for m in MODELS for s in SLOTS]
    present = [(m, s) for m, s in expected if s in arms.get(m, {})]
    missing = [(m, s) for m, s in expected if (m, s) not in present]
    complete = {
        f'{m}/{s}': sorted(item_set.ids - arms[m][s].board.ids)[:5]
        for m, s in present
        if arms[m][s].board.ids != item_set.ids
    }
    return {
        'expected_arms': len(expected),
        'present_arms': len(present),
        'missing_arms': [f'{m}/{s}' for m, s in missing],
        'arms_not_covering_full_item_set': {
            key: {
                'n_scored': arms[key.split('/')[0]][key.split('/')[1]].board.count,
                'n_expected': len(item_set.ids),
                'example_missing_ids': ids,
            }
            for key, ids in complete.items()
        },
        'item_sets_identical_across_arms': _identical(arms),
        'arms_covering_exact_frozen_set': len(present) - len(complete),
        'ready_for_final_comparison': (
            len(present) == len(expected) and not complete
        ),
    }


def _identical(arms: Mapping[str, Mapping[str, Arm]]) -> bool:
    seen = {arm.board.ids for slots in arms.values() for arm in slots.values()}
    return len(seen) <= 1


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def _complete_draws(
    macro: Callable[[Iterable[int]], float | None],
    n_clusters: int,
    resamples: int,
    seed: int,
) -> tuple[list[float], int]:
    """Cluster resamples that cover the fixed task set, and how many were redrawn."""
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    redraws = 0
    limit = resamples * 20
    while len(draws) < resamples and redraws < limit:
        value = macro(rng.integers(0, n_clusters, size=n_clusters))
        if value is None:
            redraws += 1
            continue
        draws.append(value)
    return draws, redraws


def _cluster_differences(
    a: Scoreboard,
    b: Scoreboard,
    clusters: Mapping[str, str],
    tasks: Mapping[str, Task | None],
    shared: Sequence[str],
) -> tuple[Mapping[str, list[tuple[Task, float]]], frozenset[Task]]:
    """Per-cluster (task, difference) pairs, and the fixed task set to macro over."""
    by_cluster: dict[str, list[tuple[Task, float]]] = defaultdict(list)
    fixed: set[Task] = set()
    for item_id in shared:
        task = tasks.get(item_id)
        if task is None:
            continue
        diff = b.items[item_id].recognition - a.items[item_id].recognition
        by_cluster[clusters[item_id]].append((task, diff))
        fixed.add(task)
    return by_cluster, frozenset(fixed)


def _clustered_bootstrap(
    a: Scoreboard,
    b: Scoreboard,
    clusters: Mapping[str, str],
    tasks: Mapping[str, Task | None],
    resamples: int,
    seed: int,
) -> dict | None:
    """Task-macro paired difference with whole media clusters as the unit.

    Clusters are resampled globally, with joint weights across tasks, because
    412 of the 423 multi-item clusters span two tasks -- resampling inside a
    stratum would tear exactly the correlation this column exists to respect.

    The macro is taken over a *fixed* task set: the tasks present in the real
    data, decided once before resampling. A draw that happens to contain no
    item for one of those tasks is redrawn rather than averaged over a smaller
    set, since letting the denominator vary between resamples would widen the
    interval for a bookkeeping reason instead of a statistical one. Redraws are
    counted and reported so an unstable view is visible rather than hidden.
    """
    shared = [i for i in a.items if i in b.items]
    if not shared:
        return None
    by_cluster, target = _cluster_differences(a, b, clusters, tasks, shared)
    if not by_cluster or not target:
        return None
    keys = list(by_cluster)
    payload = [by_cluster[k] for k in keys]

    def macro(indices: Iterable[int]) -> float | None:
        sums: dict[Task, list[float]] = defaultdict(list)
        for index in indices:
            for task, diff in payload[index]:
                sums[task].append(diff)
        if set(sums) != target:
            return None
        return float(np.mean([float(np.mean(sums[t])) for t in target]))

    estimate = macro(range(len(keys)))
    if estimate is None:
        return None
    draws, redraws = _complete_draws(macro, len(keys), resamples, seed)
    if len(draws) < resamples:
        return {
            'estimate': estimate,
            'unstable': (
                f'only {len(draws)} of {resamples} cluster draws covered all '
                f'{len(target)} tasks; interval withheld'
            ),
            'n_clusters': len(keys),
            'n_items': len(shared),
        }
    array = np.asarray(draws, dtype=float)
    low, high = (float(x) for x in np.percentile(array, [2.5, 97.5]))
    return {
        'estimate': estimate,
        'ci_low': low,
        'ci_high': high,
        'n_clusters': len(keys),
        'n_items': len(shared),
        'tasks_in_macro': len(target),
        'redrawn_incomplete_draws': redraws,
    }


def compare(
    a: Scoreboard,
    b: Scoreboard,
    item_set: ItemSet,
    resamples: int,
    clustered: bool,
) -> dict | None:
    """The plan's primary and secondary intervals, and the optional diagnostic."""
    primary = b_minus_a(a, b, resamples, macro=True)
    if primary is None:
        return None
    result = {
        'task_macro': primary,
        'sample_weighted': b_minus_a(a, b, resamples, macro=False),
    }
    if clustered:
        result['media_clustered_diagnostic'] = _clustered_bootstrap(
            a,
            b,
            item_set.clusters(),
            {i: e.task for i, e in item_set.examples.items()},
            resamples,
            seed=0,
        )
    return result


def b_minus_a(
    a: Scoreboard, b: Scoreboard, resamples: int, macro: bool
) -> dict | None:
    shared = [i for i in a.items if i in b.items]
    if not shared:
        return None
    strata = {i: a.items[i].task or 'pooled' for i in shared}
    outcome = paired_bootstrap(
        {i: a.items[i].recognition for i in shared},
        {i: b.items[i].recognition for i in shared},
        strata,
        n_resamples=resamples,
        seed=0,
        macro=macro,
    )
    return {
        'estimate': outcome.estimate,
        'ci_low': outcome.ci_low,
        'ci_high': outcome.ci_high,
        'p_value': outcome.p_value,
        'n_items': outcome.n_items,
    }


def summarise(arm: Arm, item_set: ItemSet) -> dict:
    """One arm's aggregates over every reported view, plus its per-task means."""
    board = arm.board
    views = {
        name: _view(board.restrict(ids))
        for name, ids in item_set.views().items()
    }
    return {
        'model': arm.model,
        'slot': arm.slot,
        'views': views,
        'by_task': {
            task.value: round(value, 4) for task, value in board.by_task().items()
        },
        'multilabel': {
            task.value: {
                'precision': round(prf.precision, 4),
                'recall': round(prf.recall, 4),
                'f1': round(prf.f1, 4),
            }
            for task, prf in board.multilabel().items()
        },
        'cost': arm.meta,
        'problems': list(arm.problems),
    }


def _view(board: Scoreboard) -> dict:
    macro = board.macro()
    flat = board.mean()
    return {
        'n': board.count,
        'task_macro': None if macro is None else round(macro, 4),
        'sample_weighted': None if flat is None else round(flat, 4),
    }


def comparisons_for(
    slots: Mapping[str, Arm],
    item_set: ItemSet,
    resamples: int,
    clustered: bool,
) -> dict:
    """Every pairwise difference for one student, over every reported view.

    Comparability is judged per pair, not per batch. A single lagging arm
    elsewhere in the matrix must not blank the differences between two arms
    that did both finish -- during a live run that would hide most of what the
    report exists to show.
    """
    out: dict[str, dict] = {}
    for label, left, right in COMPARISONS:
        if left not in slots or right not in slots:
            continue
        a, b = slots[left], slots[right]
        if a.fatal or b.fatal:
            out[label] = {
                'withheld': 'one side is not a usable result',
                'reasons': [*a.fatal, *b.fatal],
            }
            continue
        if a.board.ids != b.board.ids:
            only_a, only_b = len(a.board.ids - b.board.ids), len(
                b.board.ids - a.board.ids
            )
            out[label] = {
                'withheld': (
                    f'these two arms cover different ids ({left} has {only_a} '
                    f'the other lacks, {right} has {only_b}); pairing them '
                    'would mix denominators'
                )
            }
            continue
        out[label] = {
            'meaning': COMPARISON_MEANING[label],
            'views': {
                name: compare(
                    a.board.restrict(ids),
                    b.board.restrict(ids),
                    item_set,
                    resamples,
                    clustered,
                )
                for name, ids in item_set.views().items()
            },
        }
    return out


def holm_over_family(report: Mapping[str, dict]) -> Mapping[str, float]:
    """Holm-adjusted p across every model x comparison read off the full view."""
    raw = {}
    for model, block in report.items():
        for label, entry in block.get('comparisons', {}).items():
            full = (entry.get('views') or {}).get('full') or {}
            macro = full.get('task_macro') or {}
            if 'p_value' in macro:
                raw[f'{model} {label}'] = macro['p_value']
    return holm_adjust(raw) if raw else {}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(entry: Mapping[str, object] | None) -> str:
    if not entry:
        return 'n/a'
    return (
        f"{entry['estimate']:+.4f} [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]"
        + (f" p={entry['p_value']:.3f}" if 'p_value' in entry else '')
    )

def _item_set_section(item_set: Mapping[str, object]) -> list[str]:
    """Which items were scored, and why the plan's own counts are not used."""
    lines = [
        '## Item set',
        '',
        f"Scored on the {item_set['n']} rows of "
        '`results/v3/student_specific_full_20260918/evaluation_items.jsonl` '
        f"(sha256 {str(item_set['items_sha256'])[:16]}...). "
        f"`data/manifests/mme_emotion.jsonl` has {item_set['n_source']} rows; "
        f"{item_set['n_excluded']} were removed for sharing a media hash with a "
        'build, val, or reserved ext_test item.',
        '',
        f"The development strata are **{item_set['old_label']}** and "
        f"**{item_set['new_label']}**. "
        f"{item_set['n_old_dropped']} of the 1,000 pre-registered subset ids fall "
        'inside the excluded set, so the intersection with the clean set is '
        f"{item_set['n_old']} and the remainder is {item_set['n_new']} -- matching "
        'the amended plan and `execution_manifest.json`. Labels carry the counts '
        'so that no reader takes them for a round 1,000 and 4,577.',
        '',
        '| view | n |',
        '|---|---|',
    ]
    lines += [f'| {name} | {count} |' for name, count in item_set['views'].items()]
    return lines


def _coverage_section(coverage: Mapping[str, object]) -> list[str]:
    """How much of the matrix exists, and where the arms disagree."""
    lines = [
        '',
        '## Coverage',
        '',
        f"{coverage['present_arms']} of {coverage['expected_arms']} arms present. "
        'Identical item set across every arm: '
        f"{coverage['item_sets_identical_across_arms']}.",
        '',
    ]
    lines += [
        f"Arms covering the exact frozen set: "
        f"{coverage['arms_covering_exact_frozen_set']} / "
        f"{coverage['expected_arms']}. Ready for a final comparison: "
        f"{coverage['ready_for_final_comparison']}.",
        '',
    ]
    if coverage['missing_arms']:
        lines += ['Missing: ' + ', '.join(coverage['missing_arms']), '']
    if coverage.get('marker_defects'):
        lines += ['Markers that could not be read:', '']
        lines += [f'- {defect}' for defect in coverage['marker_defects']]
        lines.append('')
    shortfalls = coverage['arms_not_covering_full_item_set']
    if shortfalls:
        lines += ['Arms not covering the full item set:', '']
        lines += [
            f"- `{key}`: {info['n_scored']} / {info['n_expected']} scored"
            for key, info in shortfalls.items()
        ]
        lines.append('')
    return lines


def _arm_table(block: Mapping[str, object], views: Sequence[str]) -> list[str]:
    """One student's rungs, their macros over every view, and what they cost."""
    header = ' | '.join(f'{v} macro' for v in views)
    lines = [
        f'| slot | n | {header} | logical | actual | parse fail | wall s |',
        '|---' * (7 + len(views)) + '|',
    ]
    for slot in SLOTS:
        arm = block['arms'].get(slot)
        if not arm:
            continue
        cells = [
            'n/a' if arm['views'][v]['task_macro'] is None
            else f"{arm['views'][v]['task_macro']:.4f}"
            for v in views
        ]
        cost = arm['cost']
        lines.append(
            f"| {slot} | {arm['views']['full']['n']} | "
            + ' | '.join(cells)
            + f" | {cost.get('logical_calls', 0)} | {cost.get('actual_calls', 0)}"
            f" | {cost.get('parse_failures', 0)} | {cost.get('wall_s', 0):.0f} |"
        )
    problems = [
        (slot, problem)
        for slot in SLOTS
        for problem in (block['arms'].get(slot, {}).get('problems') or [])
    ]
    if problems:
        lines += ['', '**Problems**', '']
        lines += [f'- `{slot}`: {problem}' for slot, problem in problems]
    return lines


def _task_table(block: Mapping[str, object]) -> list[str]:
    tasks = sorted(
        {
            task
            for slot in SLOTS
            for task in (block['arms'].get(slot, {}).get('by_task') or {})
        }
    )
    if not tasks:
        return []
    lines = [
        '',
        '### Per-task Recognition',
        '',
        '| slot | ' + ' | '.join(tasks) + ' |',
        '|---' * (1 + len(tasks)) + '|',
    ]
    for slot in SLOTS:
        arm = block['arms'].get(slot)
        if not arm:
            continue
        row = arm['by_task']
        lines.append(
            f'| {slot} | '
            + ' | '.join(f'{row[t]:.3f}' if t in row else '-' for t in tasks)
            + ' |'
        )
    return lines


def _difference_table(
    block: Mapping[str, object], views: Sequence[str], key: str, title: str
) -> list[str]:
    """One difference table: the same comparisons read through one interval kind."""
    lines = [
        '',
        f'### {title}',
        '',
        '| comparison | ' + ' | '.join(views) + ' |',
        '|---' * (1 + len(views)) + '|',
    ]
    for label, _, _ in COMPARISONS:
        entry = block['comparisons'].get(label)
        if not entry:
            continue
        if 'withheld' in entry:
            # Only say it once; the secondary table repeats nothing.
            if key != 'task_macro':
                continue
            lines.append(
                f"| {label} | **withheld: {entry['withheld']}** "
                + '| ' * (len(views) - 1)
                + '|'
            )
            continue
        cells = [_fmt((entry['views'].get(v) or {}).get(key)) for v in views]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return lines


def _clustered_table(block: Mapping[str, object]) -> list[str]:
    """The off-protocol clustered interval, full view only."""
    rows = []
    for label, _, _ in COMPARISONS:
        entry = block['comparisons'].get(label)
        if not entry or 'withheld' in entry:
            continue
        diagnostic = (entry['views'].get('full') or {}).get(
            'media_clustered_diagnostic'
        )
        if diagnostic:
            rows.append((label, diagnostic))
    if not rows:
        return []
    lines = [
        '',
        '### Media-clustered diagnostic, full view',
        '',
        'Whole media clusters drawn instead of items, with joint weights across '
        'tasks and the macro over a task set fixed before resampling. Not the '
        'pre-registered interval: it measures what the within-clip dependence '
        'actually does to these intervals, in whichever direction it goes.',
        '',
        '| comparison | clustered estimate [95% CI] | clusters |',
        '|---|---|---|',
    ]
    lines += [
        f"| {label} | {_fmt(entry)} | {entry['n_clusters']} |"
        for label, entry in rows
    ]
    return lines


def _model_section(model: str, block: Mapping[str, object], views: Sequence[str]) -> list[str]:
    lines = [f'## {model}', '', *_arm_table(block, views), *_task_table(block)]
    if block['comparisons']:
        lines += _difference_table(
            block, views, 'task_macro', 'Paired differences (task-macro primary)'
        )
        lines += _difference_table(
            block, views, 'sample_weighted', 'Same differences, sample-weighted (secondary)'
        )
        lines += _clustered_table(block)
    lines.append('')
    return lines


def _holm_section(holm: Mapping[str, float]) -> list[str]:
    if not holm:
        return []
    lines = [
        '## Holm-adjusted p, full view, across the whole arm family',
        '',
        f'Family size {len(holm)}, counting only comparisons that were actually '
        'computed. On a partial run that is a partial family, so these adjusted '
        'values tighten as more arms land and only the final 16-arm report should '
        'be read as the corrected result.',
        '',
        '| comparison | adjusted p |',
        '|---|---|',
    ]
    lines += [
        f'| {name} | {value:.3f} |'
        for name, value in sorted(holm.items(), key=lambda kv: kv[1])
    ]
    lines.append('')
    return lines


FOOTER: Final[tuple[str, ...]] = (
    '## Reading these numbers',
    '',
    'Development evaluation. The old stratum has been inspected repeatedly across '
    'this project, and each L5 bundle was selected on D_val by the promotion gate '
    'that produced it, so an L5 - L3 interval is not a held-out estimate of '
    'construction value. D_ext-test (129 items) remains unevaluated. Overlapping '
    'intervals are not evidence of equality.',
    '',
    'The clean item set holds more items than media: one clip can appear as '
    'several items, and mostly under different tasks -- FG_SA most of all, where '
    '260 of 461 items share a clip with another item. The primary and secondary '
    'intervals resample items independently, so that dependence can miscalibrate '
    'them. Which direction is an empirical question, not something to assert: '
    'correlated errors within a clip usually widen the true interval, but the '
    'effect is not guaranteed to go one way. The clustered column measures the '
    'actual difference on these predictions rather than predicting its sign.',
    '',
)


def _blocking_banner(payload: Mapping[str, object]) -> list[str]:
    """What stops this from being a publishable result, stated before the numbers."""
    if payload['mode'] != 'final':
        return [
            '> **Provisional.** Run with `--final` for the publication gate: it '
            'checks isolation and item-set provenance and requires all sixteen '
            'arms to cover the exact frozen item set before presenting a main '
            'comparison.',
            '',
        ]
    if payload['blocking']:
        return [
            '> **NOT PUBLISHABLE.** The following must be resolved first:',
            '',
            *[f'> - {reason}' for reason in payload['blocking']],
            '',
        ]
    return ['> Publication gate passed.', '']


def render(payload: Mapping[str, object]) -> str:
    """The whole report, assembled from one section builder per block."""
    item_set = payload['item_set']
    views: Sequence[str] = list(item_set['views'])
    lines = [
        '# Student-specific optimization + full evaluation',
        '',
        f"Generated {payload['generated_utc']}. "
        'Plan: coordination/full_run_plan_20260918.md. '
        f"Resamples: {payload['resamples']}. "
        f"Mode: **{payload['mode']}**.",
        '',
        *_blocking_banner(payload),
        *_item_set_section(item_set),
        '',
        'Reserved-media re-check (fails closed): '
        f"`{payload['reserved_media']['counts']}`, ok="
        f"{payload['reserved_media']['ok']}. Item-set hash check against "
        f"execution_manifest.json: ok={payload['manifest_check']['ok']}.",
        *_coverage_section(payload['coverage']),
    ]
    for model in MODELS:
        block = payload['models'].get(model)
        if block:
            lines += _model_section(model, block, views)
    lines += _holm_section(payload['holm'])
    lines += FOOTER
    return '\n'.join(lines) + '\n'

# --------------------------------------------------------------------------


def build(batch: Path, resamples: int, clustered: bool, final: bool) -> dict:
    """Assemble the whole report.

    ``final`` is the publication gate. It refuses to present a main comparison
    unless the isolation and provenance checks pass and all sixteen arms cover
    the exact frozen item set; without it the same numbers come out labelled
    provisional, which is what the hourly supervision actually needs.
    """
    item_set = load_item_set(batch, ROOT / 'data/splits/subset_1000.json')
    arms, defects = discover(batch, item_set)
    cover = coverage(arms, item_set)
    cover['marker_defects'] = defects
    reserved = reserved_media_check(
        item_set, ROOT / 'data/manifests/external_v3.jsonl'
    )
    manifest = manifest_hash_check(batch, item_set)
    blocking = [
        message
        for message in (reserved.get('blocking'), manifest.get('blocking'))
        if message
    ]
    if final and not cover['ready_for_final_comparison']:
        blocking.append(
            f"{cover['arms_covering_exact_frozen_set']} of "
            f"{cover['expected_arms']} arms cover the exact frozen "
            f'{len(item_set.ids)}-item set; a final comparison needs all of them'
        )
    if final and defects:
        blocking.append(f'{len(defects)} markers could not be read')
    if final:
        unverifiable = sorted(
            model
            for model, slots in arms.items()
            if 'L5_own' in slots
            and not lineage_verified(model, slots['L5_own'].marker, batch)
        )
        if unverifiable:
            blocking.append(
                'L5 construction lineage cannot be verified for '
                f'{", ".join(unverifiable)}; a final publish needs a bundle path '
                'or construction_run inside this batch root'
            )
    models = {
        model: {
            'arms': {
                slot: summarise(arm, item_set) for slot, arm in slots.items()
            },
            'comparisons': comparisons_for(slots, item_set, resamples, clustered),
        }
        for model, slots in arms.items()
    }
    source = ROOT / 'data/manifests/mme_emotion.jsonl'
    payload = {
        'generated_utc': f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}',
        'plan': 'coordination/full_run_plan_20260918.md',
        'resamples': resamples,
        'item_set': {
            'n': len(item_set.ids),
            'n_source': sum(1 for _ in source.open()) if source.is_file() else None,
            'n_excluded': len(item_set.excluded),
            'n_old': len(item_set.old),
            'n_new': len(item_set.new),
            'n_old_dropped': 1000 - len(item_set.old),
            'old_label': item_set.old_label,
            'new_label': item_set.new_label,
            'items_sha256': item_set.items_sha256,
            'excluded_sha256': item_set.excluded_sha256,
            'views': {n: len(i) for n, i in item_set.views().items()},
        },
        'reserved_media': reserved,
        'manifest_check': manifest,
        'mode': 'final' if final else 'provisional',
        'blocking': blocking,
        'publishable': final and not blocking,
        'coverage': cover,
        'models': models,
    }
    payload['holm'] = holm_over_family(models)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=BATCH)
    parser.add_argument(
        '--resamples',
        type=int,
        default=RESAMPLES,
        help='bootstrap resamples; use a small value for hourly checks',
    )
    parser.add_argument(
        '--media-cluster-diagnostic',
        action='store_true',
        help='add the off-protocol clustered interval described in the docstring',
    )
    parser.add_argument(
        '--out-prefix',
        default='student_specific_full',
        help='basename under <batch>/reports/',
    )
    parser.add_argument(
        '--final',
        action='store_true',
        help='publication mode: refuse a main comparison unless the isolation '
        'and provenance checks pass and all 16 arms cover the exact frozen set',
    )
    parser.add_argument(
        '--stdout', action='store_true', help='print the markdown instead of writing'
    )
    args = parser.parse_args()

    payload = build(
        args.batch,
        args.resamples,
        args.media_cluster_diagnostic or args.final,
        args.final,
    )
    markdown = render(payload)
    if args.stdout:
        print(markdown, end='')
        return 0
    reports = args.batch / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    stamp = payload['generated_utc'].replace(':', '').replace('-', '')
    for name, text in (
        (f'{args.out_prefix}_{stamp}.md', markdown),
        (f'{args.out_prefix}_{stamp}.json', json.dumps(payload, indent=1) + '\n'),
        (f'{args.out_prefix}_latest.md', markdown),
        (f'{args.out_prefix}_latest.json', json.dumps(payload, indent=1) + '\n'),
    ):
        target = reports / name
        temp = reports / f'.{name}.tmp'
        temp.write_text(text, encoding='utf-8')
        temp.replace(target)
        print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
