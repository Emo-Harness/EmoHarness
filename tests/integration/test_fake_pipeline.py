"""Every entry point, in phase order, with fakes, into one throwaway project root.

This is the acceptance test for Phase 0 in ``docs/plan.md``: the whole pipeline
runs end to end without a GPU, without the benchmark, and without an API key.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from emoharness.common.io import read_json

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'
ROWS = ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'best_of_4')
JUDGED_ROWS = ('L0', 'L3', 'L5')


def load_example(name: str) -> ModuleType:
    """Import one ``examples/<name>.py`` by path; they are scripts, not a package."""
    path = EXAMPLES / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'emoharness_example_{name}', path)
    if spec is None or spec.loader is None:  # pragma: no cover -- path always exists
        raise AssertionError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(name: str, root: Path, *args: str) -> int:
    """Call one entry point's ``main`` with ``--fake`` and this run's root."""
    argv: Sequence[str] = ['--fake', '--root', str(root), *args]
    return int(load_example(name).main(list(argv)))


def run_pipeline(root: Path) -> tuple[tuple[str, int], ...]:
    """Every step in phase order; returns each step's exit code."""
    steps: list[tuple[str, tuple[str, ...]]] = [
        ('prepare_benchmark', ('--per-task', '2')),
        ('prepare_external', ('--per-source', '2', '--per-kind', '2,2,2')),
        ('preprocess', ('--items', 'all')),
        ('leakage_report', ()),
        ('preregister', ()),
        ('run_ladder', ('--rows', 'L0,L1,L2,L3,best_of_4')),
        ('build_harness', ('--stage', '2', '--rounds', '1')),
        ('run_ladder', ('--rows', 'L4')),
        ('build_harness', ('--stage', '3', '--rounds', '1', '--freeze')),
        ('run_ladder', ('--rows', 'L5,L6')),
        *[('judge_official', ('--row', row, '--items', 'full')) for row in JUDGED_ROWS],
        ('calibrate_proxy_judge', ('--row', 'L3', '--items', 'full')),
        ('analyze', ('--resamples', '100')),
    ]
    return tuple((name, run(name, root, *args)) for name, args in steps)


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the whole pipeline once and hand every test the project root."""
    root = tmp_path_factory.mktemp('fake_pipeline')
    for name, code in run_pipeline(root):
        assert code == 0, f'{name} exited {code}'
    return root


class TestPhaseOneAndTwo:
    def test_the_benchmark_manifest_and_its_companions_exist(
        self, pipeline: Path
    ) -> None:
        data = pipeline / 'data'
        assert (data / 'manifests' / 'mme_emotion.jsonl').is_file()
        assert (data / 'checksums' / 'mme_emotion.sha256').is_file()
        assert (data / 'manifests' / 'mme_emotion_report.json').is_file()

    def test_the_construction_manifest_carries_three_splits(
        self, pipeline: Path
    ) -> None:
        splits = read_json(pipeline / 'data' / 'splits' / 'external_splits.json')
        assert set(splits['counts']) == {'build', 'val', 'ext_test'}

    def test_the_leakage_report_is_written_and_clean(self, pipeline: Path) -> None:
        report = read_json(pipeline / 'data' / 'checksums' / 'leakage_report.json')
        assert report['flags'] == []
        assert report['checked_pairs'] > 0


class TestPhaseThreeAndFive:
    def test_the_clip_cache_is_populated(self, pipeline: Path) -> None:
        entries = list((pipeline / 'data' / 'derived').glob('*/*.json'))
        assert len(entries) > 8

    def test_the_subset_and_pre_registration_exist(self, pipeline: Path) -> None:
        subset = read_json(pipeline / 'data' / 'splits' / 'subset_1000.json')
        assert subset['ids']
        text = (pipeline / 'results' / 'fake' / 'preregistration.md').read_text()
        assert 'H1 seen improvement' in text
        assert subset['subset_hash'] in text

    def test_the_calibration_report_exists(self, pipeline: Path) -> None:
        payload = read_json(pipeline / 'results' / 'calibration' / 'proxy_judge.json')
        assert payload['sample_size'] > 0
        assert payload['gate_margin_points'] >= 2.0


class TestPhaseFourSixSeven:
    @pytest.mark.parametrize('row', ROWS)
    def test_every_row_wrote_predictions(self, row: str, pipeline: Path) -> None:
        directory = pipeline / 'results' / 'raw_predictions' / row
        written = [p for p in directory.glob('*.jsonl') if '.outputs.' not in p.name]
        assert len(written) == 1
        assert written[0].read_text(encoding='utf-8').strip()

    @pytest.mark.parametrize('row', ['L3', 'L4', 'L5'])
    def test_harness_rows_persist_their_records(self, row: str, pipeline: Path) -> None:
        directory = pipeline / 'results' / 'raw_predictions' / row
        assert list(directory.glob('*.outputs.jsonl'))

    @pytest.mark.parametrize('stage', [2, 3])
    def test_each_construction_stage_left_provenance(
        self, stage: int, pipeline: Path
    ) -> None:
        out_dir = pipeline / 'results' / 'construction' / f'stage{stage}' / '0'
        assert (out_dir / 'loop_result.json').is_file()
        assert (out_dir / 'round_00' / 'packet.txt').is_file()
        assert (out_dir / 'final' / 'manifest.json').is_file()

    def test_no_benchmark_id_reached_the_construction_artifacts(
        self, pipeline: Path
    ) -> None:
        from emoharness.data.manifest import read_manifest

        ids = {
            e.example_id
            for e in read_manifest(
                pipeline / 'data' / 'manifests' / 'mme_emotion.jsonl'
            )
        }
        text = '\n'.join(
            path.read_text(encoding='utf-8', errors='ignore')
            for path in (pipeline / 'results' / 'construction').rglob('*')
            if path.is_file() and path.suffix in {'.json', '.txt', '.md', '.yaml'}
        )
        assert not [example_id for example_id in ids if example_id in text]

    def test_the_freeze_record_names_a_content_hash(self, pipeline: Path) -> None:
        freeze = read_json(pipeline / 'results' / 'freeze.json')
        assert len(freeze['content_hash']) == 64

    @pytest.mark.parametrize('row', JUDGED_ROWS)
    def test_every_judged_row_has_per_task_outputs(
        self, row: str, pipeline: Path
    ) -> None:
        run_dir = pipeline / 'results' / 'judge' / row / 'full'
        assert len(list(run_dir.glob('*/responses.json'))) == 8
        assert len(list(run_dir.glob('*/eval.json'))) == 8


class TestPhaseEight:
    def test_every_metrics_artifact_exists(self, pipeline: Path) -> None:
        metrics = pipeline / 'results' / 'metrics'
        for name in (
            'ladder.json',
            'ladder.md',
            'per_task.json',
            'hypotheses.json',
            'multilabel.json',
            'cost.json',
        ):
            assert (metrics / name).is_file(), name

    def test_the_ladder_table_has_every_row_that_was_run(self, pipeline: Path) -> None:
        document = read_json(pipeline / 'results' / 'metrics' / 'ladder.json')
        assert {row['row'] for row in document['rows']} == set(ROWS)

    def test_the_markdown_table_names_every_row(self, pipeline: Path) -> None:
        body = (pipeline / 'results' / 'metrics' / 'ladder.md').read_text()
        assert all(f'| {row} |' in body for row in ROWS)

    def test_judged_rows_report_official_scores(self, pipeline: Path) -> None:
        document = read_json(pipeline / 'results' / 'metrics' / 'ladder.json')
        by_row = {row['row']: row for row in document['rows']}
        assert all(by_row[row]['source'] == 'official' for row in JUDGED_ROWS)
        assert by_row['L1']['source'] == 'local'

    def test_every_pre_registered_test_is_reported(self, pipeline: Path) -> None:
        report = read_json(pipeline / 'results' / 'metrics' / 'hypotheses.json')
        assert [test['id'] for test in report['tests']] == [
            'H1',
            'H2',
            'H-B_seen',
            'H-B_held_out',
            'H3',
        ]

    def test_the_cost_table_totals_the_ledger(self, pipeline: Path) -> None:
        cost = read_json(pipeline / 'results' / 'metrics' / 'cost.json')
        assert cost['total']['calls'] > 0
        assert set(cost['by_role']) >= {'target', 'judge_official'}
