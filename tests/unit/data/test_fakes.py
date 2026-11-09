"""Synthetic corpora in the real on-disk formats: what a ``--fake`` run ingests."""

from __future__ import annotations

from pathlib import Path

from emoharness.common.io import read_json
from emoharness.data.benchmark import build_benchmark_manifest, load_task_annotations
from emoharness.data.fakes import make_fake_benchmark, make_fake_external
from emoharness.data.sources import ADAPTERS
from emoharness.data.tasks import Task


class TestFakeBenchmark:
    def test_writes_one_annotation_file_per_task(self, tmp_path: Path) -> None:
        root = make_fake_benchmark(tmp_path, per_task=3)

        assert all((root / f'{task.value}.json').is_file() for task in Task)

    def test_annotations_parse_and_reference_real_media(self, tmp_path: Path) -> None:
        root = make_fake_benchmark(tmp_path, per_task=3)

        rows = load_task_annotations(root, Task.ER_WILD)

        assert len(rows) == 3
        assert all((root / row.video_id).is_file() for row in rows)

    def test_multi_label_tasks_use_their_own_separators(self, tmp_path: Path) -> None:
        root = make_fake_benchmark(tmp_path, per_task=6)

        fg = read_json(root / f'{Task.FG_ER.value}.json')
        ml = read_json(root / f'{Task.ML_ER.value}.json')

        assert any(';' in row['label'] for row in fg)
        assert all(',' in row['label'] for row in ml)

    def test_media_bytes_differ_per_clip(self, tmp_path: Path) -> None:
        root = make_fake_benchmark(tmp_path, per_task=4)

        examples = build_benchmark_manifest(root)

        payloads = {Path(e.media_path).read_bytes() for e in examples}
        assert len(payloads) == len(examples)

    def test_is_deterministic_in_its_seed(self, tmp_path: Path) -> None:
        first = read_json(make_fake_benchmark(tmp_path / 'a', 3, 7) / 'SA.json')
        second = read_json(make_fake_benchmark(tmp_path / 'b', 3, 7) / 'SA.json')

        assert first == second


class TestFakeExternal:
    def test_every_written_corpus_is_readable_by_its_adapter(
        self, tmp_path: Path
    ) -> None:
        roots = make_fake_external(tmp_path, per_source=2)

        for origin in ('meld', 'meld_sentiment', 'mosei', 'chsims', 'cremad'):
            splits = ('train', 'dev') if origin.startswith('meld') else ()
            rows = ADAPTERS[origin].load(roots[origin], splits)
            assert rows, origin

    def test_meld_clips_exist_where_the_adapter_looks(self, tmp_path: Path) -> None:
        roots = make_fake_external(tmp_path, per_source=2)

        rows = ADAPTERS['meld'].load(roots['meld'], ('train',))

        assert all(Path(row.media_path).is_file() for row in rows)
