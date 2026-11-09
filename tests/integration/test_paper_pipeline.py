"""The paper's pipeline driver end to end on fakes: data, mme, mhe, then the report.

``experiments.paper.pipeline all --fake`` must run every MME-Emotion stage a
reproduction runs, in order, into one scratch root, and the report must read
what they wrote. IntentQA has no fake mode and is skipped by the driver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.layout import ROOT, Layout
from experiments.paper import pipeline

pytestmark = pytest.mark.integration

STUDENTS = ('qwen3vl_4b', 'gemma4_e4b')


def test_the_whole_pipeline_runs_on_fakes(tmp_path: Path) -> None:
    lines: list[str] = []
    code = pipeline.main(
        ['all', '--fake', '--root', str(tmp_path), '--students', *STUDENTS],
    )
    assert code == 0, lines
    layout = Layout(root=tmp_path, checkout=ROOT)
    assert layout.construction_manifest.is_file()
    assert (tmp_path / 'data/checksums/leakage_report.json').is_file()
    for st in STUDENTS:
        for row in ('L0', 'L1', 'L3', 'L5'):
            assert (
                layout.main_dir(st) / 'raw_predictions' / row / 'clean.jsonl'
            ).is_file()
        for builder in ('gpt5_6_luna', 'gpt6_1_sol'):
            assert (layout.construction_dir(builder, st) / 'loop_result.json').is_file()
        assert layout.mhe_predictions('gpt6_1_sol', st).is_file()
    doc = json.loads((layout.paper / 'paper.json').read_text())
    for st in STUDENTS:
        entry = doc['mme']['students'][st]
        assert set(entry['table1']) == {
            'video',
            'audio',
            'shared',
            'mhe:gpt5_6_luna',
            'mhe:gpt6_1_sol',
        }
        assert set(entry['mhe_minus_shared']) == {'gpt5_6_luna', 'gpt6_1_sol'}
        assert doc['construction']['runs']['gpt6_1_sol'][st]['generations'] >= 1
    assert (layout.paper / 'figures/figure6.json').is_file()


def test_a_second_run_only_skips(tmp_path: Path) -> None:
    args = ['mme', '--fake', '--root', str(tmp_path), '--students', STUDENTS[0]]
    assert pipeline.main(['data', '--fake', '--root', str(tmp_path)]) == 0
    assert pipeline.main(args) == 0
    lines: list[str] = []
    runner = pipeline.Pipeline(
        Layout(root=tmp_path, checkout=ROOT), fake=True, echo=lines.append
    )
    runner.mme([STUDENTS[0]])
    assert lines and all('done, skipped' in line for line in lines)
