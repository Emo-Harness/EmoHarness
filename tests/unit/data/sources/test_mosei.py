from __future__ import annotations

from pathlib import Path

from emoharness.data.sources import mosei
from emoharness.data.tasks import Kind, Split

_CSV = (
    'video_id,segment_id,sentiment,path\n'
    'abc123,0,2.5,clips/abc123_0.mp4\n'
    'abc123,1,-1.0,clips/abc123_1.mp4\n'
    'xyz789,0,0.0,clips/xyz789_0.mp4\n'
)


def _write_labels(root: Path) -> None:
    (root / 'mosei_labels.csv').write_text(_CSV, encoding='utf-8')


def test_load_collapses_the_seven_level_score(tmp_path: Path) -> None:
    _write_labels(tmp_path)
    examples = mosei.load(tmp_path, ('train',))
    by_id = {e.example_id: e for e in examples}
    assert by_id['mosei_abc123_0'].labels == ('positive',)
    assert by_id['mosei_abc123_1'].labels == ('negative',)
    assert by_id['mosei_xyz789_0'].labels == ('neutral',)


def test_raw_score_never_appears_in_the_manifest(tmp_path: Path) -> None:
    _write_labels(tmp_path)
    example = mosei.load(tmp_path, ('train',))[0]
    dumped = example.model_dump(mode='json')
    assert '2.5' not in str(dumped)
    assert dumped['metadata'] == {'collapsed_from': 'mosei_7'}


def test_fields_match_the_expected_shape(tmp_path: Path) -> None:
    _write_labels(tmp_path)
    example = mosei.load(tmp_path, ('train',))[0]
    assert example.source_id == 'abc123'
    assert example.origin == 'mosei'
    assert example.kind is Kind.SENTIMENT
    assert example.split is Split.BUILD
    assert example.media_path == str(tmp_path / 'clips' / 'abc123_0.mp4')
    assert example.label_set == ('positive', 'negative', 'neutral')


def test_split_names_is_ignored_because_the_format_has_no_split_column(
    tmp_path: Path,
) -> None:
    _write_labels(tmp_path)
    assert mosei.load(tmp_path, ()) == mosei.load(tmp_path, ('anything',))
