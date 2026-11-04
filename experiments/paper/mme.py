"""MME-Emotion: Table 1, Table 4, Figures 1c, 3, 8 and 9, and the numbers of Sections 4-6.

Every number comes from the raw predictions of the four conditions on the 5,507
evaluation items (``experiments.layout``), scored locally: Recognition is the
first normalised label for single-label prompts and the fraction of gold labels
recovered for multi-label ones (FG-ER, ML-ER). Intervals are task-stratified
paired bootstrap draws whose statistic is the task macro.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from emoharness.common.stats import mean_or_none
from emoharness.data.label_set import normalize_label
from emoharness.data.manifest import Example, read_manifest
from emoharness.data.tasks import TASK_GROUP, Group, Task
from emoharness.evaluation.bootstrap import holm_adjust, paired_bootstrap
from emoharness.evaluation.schema import PredictionRecord
from emoharness.evaluation.scoreboard import Scoreboard
from emoharness.runner.items import CLEAN, examples_for
from emoharness.runner.paths import ProjectPaths
from experiments.common import ORDER
from experiments.layout import CONDITION_ROWS, Layout

RESAMPLES: Final = 10_000
SEED: Final = 0
#: Table 1's column order.
TASKS: Final = (
    Task.ER_LAB,
    Task.ER_WILD,
    Task.SA,
    Task.NOISE_ER,
    Task.FG_ER,
    Task.ML_ER,
    Task.FG_SA,
    Task.IR,
)
MULTI_LABEL_TASKS: Final = (Task.FG_ER, Task.ML_ER)
BASE_CONDITIONS: Final = ('video', 'audio', 'shared')


def mhe(builder: str) -> str:
    """The condition key of one Builder's MHE harness."""
    return f'mhe:{builder}'


# --- loading -------------------------------------------------------------


def evaluation_examples(layout: Layout) -> dict[str, Example]:
    """The evaluation items (the ``clean`` item set), keyed by id."""
    paths = ProjectPaths.from_root(layout.root)
    return {e.example_id: e for e in examples_for(CLEAN, paths)}


def read_predictions(path: Path) -> tuple[PredictionRecord, ...] | None:
    if not path.is_file():
        return None
    return tuple(
        PredictionRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )


@dataclass(frozen=True)
class Condition:
    """One condition's predictions for one student, and their scores."""

    predictions: Mapping[str, PredictionRecord]
    board: Scoreboard


def load_student(
    layout: Layout,
    student: str,
    builders: Sequence[str],
    examples: Mapping[str, Example],
) -> dict[str, Condition | None]:
    """Video, Audio, Shared and each Builder's MHE for one student; None if not run."""
    paths = {
        name: layout.row_predictions(student, CONDITION_ROWS[name])
        for name in BASE_CONDITIONS
    }
    paths.update({mhe(b): layout.mhe_predictions(b, student) for b in builders})
    out: dict[str, Condition | None] = {}
    for name, path in paths.items():
        records = read_predictions(path)
        out[name] = (
            None
            if records is None
            else Condition(
                predictions={r.example_id: r for r in records},
                board=Scoreboard.from_predictions(records, examples),
            )
        )
    return out


# --- per-condition summaries (Table 1, Figure 3, App C) ------------------------


def summary(board: Scoreboard, ci: bool = True) -> dict[str, Any]:
    """One Table 1 row: per-task Recognition, Seen, Held-out, Macro and its interval."""
    by_task = board.by_task()
    low_high = board.ci(RESAMPLES, seed=SEED) if ci else None
    return {
        'n_items': board.count,
        'per_task': {t.value: by_task.get(t) for t in TASKS},
        'seen': board.macro(Group.SEEN),
        'held_out': board.macro(Group.HELD_OUT),
        'macro': board.macro(),
        'ci95': list(low_high) if low_high else None,
        'seen_flat': board.in_group(Group.SEEN).mean(),
    }


def f1_board(board: Scoreboard) -> Scoreboard:
    """The same scores with every multi-label item scored by set F1 (App C)."""
    items = {
        i: s.model_copy(update={'recognition': s.prf.f1}) if s.prf is not None else s
        for i, s in board.items.items()
    }
    return Scoreboard(items=items, source=board.source)


def mean_labels(condition: Condition, examples: Mapping[str, Example]) -> float | None:
    """Mean number of predicted labels per FG-ER / ML-ER item (App C)."""
    counts = [
        len(set(map(normalize_label, p.labels)))
        for i, p in condition.predictions.items()
        if examples[i].task in MULTI_LABEL_TASKS
    ]
    return mean_or_none(map(float, counts))


def student_calls(condition: Condition) -> int:
    return sum(p.target_calls for p in condition.predictions.values())


# --- paired contrasts (Table 4, Figures 1c, 8, 9) --------------------------------


def contrast(
    old: Scoreboard, new: Scoreboard, examples: Mapping[str, Example]
) -> dict[str, Any]:
    """``new - old`` on the shared items: task-macro estimate, 95% CI, two-sided p."""
    shared = sorted(set(old.items) & set(new.items))
    result = paired_bootstrap(
        {i: old.items[i].recognition for i in shared},
        {i: new.items[i].recognition for i in shared},
        {i: examples[i].task for i in shared},
        n_resamples=RESAMPLES,
        seed=SEED,
    )
    return {
        'estimate': result.estimate,
        'ci95': [result.ci_low, result.ci_high],
        'p': result.p_value,
        'n_items': result.n_items,
    }


def per_task_delta(old: Scoreboard, new: Scoreboard) -> dict[str, float | None]:
    """Figure 8: ``new - old`` per task."""
    a, b = old.by_task(), new.by_task()
    return {t.value: (b[t] - a[t]) if t in a and t in b else None for t in TASKS}


def group_delta(old: Scoreboard, new: Scoreboard, group: Group) -> float | None:
    a, b = old.macro(group), new.macro(group)
    return None if a is None or b is None else b - a


def label_set_changes(old: Condition, new: Condition) -> dict[str, Any]:
    """How often ``new`` predicts a different label set than ``old`` (Sections 5.3, 6)."""
    changed = improved = worsened = 0
    shared = sorted(set(old.predictions) & set(new.predictions))
    for i in shared:
        before = {normalize_label(x) for x in old.predictions[i].labels}
        after = {normalize_label(x) for x in new.predictions[i].labels}
        if before == after:
            continue
        changed += 1
        delta = new.board.items[i].recognition - old.board.items[i].recognition
        improved += delta > 0
        worsened += delta < 0
    return {
        'n_items': len(shared),
        'changed': changed,
        'changed_fraction': changed / len(shared) if shared else None,
        'improved': improved,
        'worsened': worsened,
        'same_score': changed - improved - worsened,
    }


# --- the whole MME-Emotion part ------------------------------------------------


def analyse(
    layout: Layout,
    builders: Sequence[str],
    students: Sequence[str] = ORDER,
    examples: Mapping[str, Example] | None = None,
) -> dict[str, Any]:
    """Every MME-Emotion result, per student; Holm runs over the 4 x B MHE tests."""
    if examples is None:
        examples = evaluation_examples(layout)
    out: dict[str, Any] = {'builders': list(builders), 'students': {}}
    raw_p: dict[str, float] = {}
    for st in students:
        rows = load_student(layout, st, builders, examples)
        boards = {k: v.board for k, v in rows.items() if v is not None}
        entry: dict[str, Any] = {
            'table1': {k: summary(b) for k, b in boards.items()},
            'f1': {k: summary(f1_board(b), ci=False) for k, b in boards.items()},
            'mean_labels_multilabel': {
                k: mean_labels(v, examples) for k, v in rows.items() if v is not None
            },
            'student_calls': {
                k: student_calls(v) for k, v in rows.items() if v is not None
            },
            'contrasts': {},
            'mhe_minus_shared': {},
        }
        for key, new, old in (
            ('audio_minus_video', 'audio', 'video'),
            ('shared_minus_audio', 'shared', 'audio'),
        ):
            if new in boards and old in boards:
                entry['contrasts'][key] = contrast(boards[old], boards[new], examples)
        for b in builders:
            sh_row, mhe_row = rows.get('shared'), rows.get(mhe(b))
            if sh_row is None or mhe_row is None:
                continue
            shared, harness = sh_row.board, mhe_row.board
            test = contrast(shared, harness, examples)
            raw_p[f'{st}|{b}'] = test['p']
            entry['mhe_minus_shared'][b] = {
                **test,
                'seen': group_delta(shared, harness, Group.SEEN),
                'held_out': group_delta(shared, harness, Group.HELD_OUT),
                'seen_flat': _flat_delta(shared, harness, Group.SEEN),
                'per_task': per_task_delta(shared, harness),
                'f1_estimate': _macro_delta(f1_board(shared), f1_board(harness)),
                'label_sets': label_set_changes(sh_row, mhe_row),
            }
        out['students'][st] = entry
    family = len(students) * len(builders)
    out['holm_family'] = family
    if len(raw_p) == family:
        for key, p in holm_adjust(raw_p).items():
            st, b = key.split('|')
            out['students'][st]['mhe_minus_shared'][b]['p_holm'] = p
    out['ranges'] = _ranges(out, builders)
    return out


def _macro_delta(old: Scoreboard, new: Scoreboard) -> float | None:
    a, b = old.macro(), new.macro()
    return None if a is None or b is None else b - a


def _flat_delta(old: Scoreboard, new: Scoreboard, group: Group) -> float | None:
    a, b = old.in_group(group).mean(), new.in_group(group).mean()
    return None if a is None or b is None else b - a


def _ranges(out: Mapping[str, Any], builders: Iterable[str]) -> dict[str, Any]:
    """The min-max ranges the text quotes (Abstract, Sections 1, 5.1, 5.2, App C)."""
    students = out['students'].values()

    def span(values: Iterable[float | None]) -> list[float] | None:
        kept = [v for v in values if v is not None]
        return [min(kept), max(kept)] if kept else None

    def macro(entry: Mapping[str, Any], table: str, key: str) -> float | None:
        row = entry[table].get(key)
        return row['macro'] if row else None

    def diff(entry: Mapping[str, Any], table: str, new: str, old: str) -> float | None:
        a, b = macro(entry, table, old), macro(entry, table, new)
        return None if a is None or b is None else b - a

    pairs = [(e, b) for e in students for b in builders]
    return {
        'audio_minus_video': span(
            diff(e, 'table1', 'audio', 'video') for e in students
        ),
        'shared_minus_audio': span(
            diff(e, 'table1', 'shared', 'audio') for e in students
        ),
        'mhe_minus_shared': span(
            e['mhe_minus_shared'].get(b, {}).get('estimate') for e, b in pairs
        ),
        'mhe_minus_video': span(diff(e, 'table1', mhe(b), 'video') for e, b in pairs),
        'mhe_minus_audio': span(diff(e, 'table1', mhe(b), 'audio') for e, b in pairs),
        'mhe_minus_shared_seen': span(
            e['mhe_minus_shared'].get(b, {}).get('seen') for e, b in pairs
        ),
        'mhe_minus_shared_held_out': span(
            e['mhe_minus_shared'].get(b, {}).get('held_out') for e, b in pairs
        ),
        'f1_audio_minus_video': span(diff(e, 'f1', 'audio', 'video') for e in students),
        'f1_shared_minus_audio': span(
            diff(e, 'f1', 'shared', 'audio') for e in students
        ),
        'f1_mhe_minus_shared': span(
            e['mhe_minus_shared'].get(b, {}).get('f1_estimate') for e, b in pairs
        ),
        'f1_mhe_minus_audio': span(diff(e, 'f1', mhe(b), 'audio') for e, b in pairs),
        'mean_labels_multilabel': {
            name: span(e['mean_labels_multilabel'].get(name) for e in students)
            for name in ('audio', 'shared')
        },
        'student_calls': {
            name: sum(e['student_calls'].get(name, 0) for e in students)
            for name in (*BASE_CONDITIONS, *(mhe(b) for b in builders))
        },
    }


# --- data summary (Section 4.1, App A) -----------------------------------------


def data_summary(layout: Layout, examples: Mapping[str, Example]) -> dict[str, Any]:
    """Item counts behind Section 4.1: evaluation, construction, validation, reserved."""
    by_task: dict[str, int] = {}
    for e in examples.values():
        if e.task is not None:
            by_task[e.task.value] = by_task.get(e.task.value, 0) + 1
    seen = sum(n for t, n in by_task.items() if TASK_GROUP[Task(t)] is Group.SEEN)
    out: dict[str, Any] = {
        'evaluation_items': len(examples),
        'evaluation_media': len(
            {e.media_hash or e.media_path for e in examples.values()}
        ),
        'evaluation_by_task': {t.value: by_task.get(t.value, 0) for t in TASKS},
        'evaluation_seen': seen,
        'evaluation_held_out': len(examples) - seen,
    }
    full = layout.manifests / 'mme_emotion.full.jsonl'
    if full.is_file():
        out['benchmark_items_total'] = len(read_manifest(full))
    if layout.benchmark_manifest.is_file():
        remaining = len(read_manifest(layout.benchmark_manifest))
        out['benchmark_items_after_construction_split'] = remaining
        out['excluded_media_overlap'] = remaining - len(examples)
    if layout.construction_manifest.is_file():
        splits: dict[str, int] = {}
        for e in read_manifest(layout.construction_manifest):
            splits[e.split.value] = splits.get(e.split.value, 0) + 1
        out['construction_splits'] = splits
    return out
