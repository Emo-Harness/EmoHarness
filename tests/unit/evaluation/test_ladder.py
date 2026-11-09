from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.evaluation.ladder import Ladder, rows_for_priority

_SUBSET = {'size_per_task': 5, 'seed': 1, 'stratify': 'proportional_by_label'}


def _ladder_file(path: Path, rows: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump({'rows': rows, 'subset': _SUBSET}), encoding='utf-8')
    return path


def test_load_model_reads_the_repo_config(repo_root: Path) -> None:
    ladder = load_model(repo_root / 'configs' / 'experiments' / 'ladder.yaml', Ladder)

    assert set(ladder.rows) == {'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'best_of_4'}
    assert ladder.subset.size_per_task == 125
    assert ladder.subset.seed == 20260907
    assert ladder.subset.stratify == 'proportional_by_label'

    l0 = ladder.rows['L0']
    assert l0.name == 'L0'
    assert l0.method == 'direct'
    assert l0.inputs == 'video'
    assert l0.item_set == 'full'
    assert l0.priority == 'P0'
    assert l0.calls == 1

    l3 = ladder.rows['L3']
    assert l3.method == 'harness'
    assert l3.stage == 1
    assert l3.calls == 4

    l6 = ladder.rows['L6']
    assert l6.method == 'teacher_direct'
    assert l6.inputs == 'matched_protocol'
    assert l6.item_set == 'subset'

    best_of_4 = ladder.rows['best_of_4']
    assert best_of_4.method == 'best_of_n'
    assert best_of_4.base == 'L2'
    assert best_of_4.n == 4
    assert best_of_4.temperature == pytest.approx(0.7)
    assert best_of_4.top_p == pytest.approx(0.9)


def test_rows_for_priority_returns_declared_order(repo_root: Path) -> None:
    ladder = load_model(repo_root / 'configs' / 'experiments' / 'ladder.yaml', Ladder)
    p0_rows = rows_for_priority(ladder, 'P0')
    assert [row.name for row in p0_rows] == [
        'L0',
        'L1',
        'L2',
        'L3',
        'L4',
        'L5',
        'L6',
        'best_of_4',
    ]
    assert rows_for_priority(ladder, 'P1') == ()


def test_a_row_is_named_by_its_yaml_key(tmp_path: Path) -> None:
    path = _ladder_file(
        tmp_path / 'ladder.yaml',
        {'my_row': {'method': 'direct', 'item_set': 'full', 'priority': 'P2'}},
    )

    assert load_model(path, Ladder).rows['my_row'].name == 'my_row'


def test_a_malformed_document_is_rejected(tmp_path: Path) -> None:
    bad_path = tmp_path / 'bad_ladder.yaml'
    bad_path.write_text(yaml.safe_dump({'rows': {'L0': {'method': 'direct'}}}))
    # missing 'item_set' and 'priority', and no top-level 'subset' key
    with pytest.raises(ValidationError):
        load_model(bad_path, Ladder)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('inputs', 'video+audio'),
        ('method', 'telepathy'),
        ('item_set', 'everything'),
        ('priority', 'P9'),
    ],
)
def test_a_misspelt_name_fails_at_load_time(
    tmp_path: Path, field: str, value: str
) -> None:
    row = {'method': 'direct', 'item_set': 'full', 'priority': 'P0', field: value}
    path = _ladder_file(tmp_path / 'ladder.yaml', {'L0': row})

    with pytest.raises(ValidationError, match=f'rows.L0.{field}'):
        load_model(path, Ladder)


@pytest.mark.parametrize('document', ['- a\n- b\n', 'rows: [L0]\nsubset: {}\n'])
def test_a_document_of_the_wrong_shape_is_rejected(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / 'ladder.yaml'
    path.write_text(document, encoding='utf-8')

    with pytest.raises(ValidationError, match=str(path)):
        load_model(path, Ladder)
