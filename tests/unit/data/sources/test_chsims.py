from __future__ import annotations

from pathlib import Path

from emoharness.data.sources import chsims
from emoharness.data.tasks import Kind, Split

_CSV = 'video_id,clip_id,label\nvideo_0001,0001,0.6\nvideo_0001,0002,-0.2\nvideo_0002,0001,0.0\n'


def _write_labels(root: Path) -> None:
    (root / 'label.csv').write_text(_CSV, encoding='utf-8')


def test_load_collapses_the_continuous_score(tmp_path: Path) -> None:
    _write_labels(tmp_path)
    examples = chsims.load(tmp_path, ('train',))
    by_id = {e.example_id: e for e in examples}
    assert by_id['chsims_video_0001_0001'].labels == ('positive',)
    assert by_id['chsims_video_0001_0002'].labels == ('negative',)
    assert by_id['chsims_video_0002_0001'].labels == ('neutral',)


def test_fields_match_the_expected_shape(tmp_path: Path) -> None:
    _write_labels(tmp_path)
    example = chsims.load(tmp_path, ('train',))[0]
    assert example.source_id == 'video_0001'
    assert example.origin == 'chsims'
    assert example.kind is Kind.SENTIMENT
    assert example.split is Split.BUILD
    assert example.language == 'zh'
    assert example.media_path == str(tmp_path / 'Raw' / 'video_0001' / '0001.mp4')
    assert example.label_set == ('positive', 'negative', 'neutral')
