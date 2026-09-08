from __future__ import annotations

from pathlib import Path

from emoharness.data.sources import cremad
from emoharness.data.tasks import Kind, Split


def _touch(root: Path, name: str) -> None:
    clip_dir = root / 'VideoFlash'
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / name).write_bytes(b'\x00')


def test_load_maps_codes_to_labels(tmp_path: Path) -> None:
    _touch(tmp_path, '1001_DFA_ANG_XX.flv')
    _touch(tmp_path, '1002_IEO_HAP_HI.mp4')
    examples = cremad.load(tmp_path, ())
    by_id = {e.example_id: e for e in examples}
    assert by_id['cremad_1001_DFA_ANG_XX'].labels == ('anger',)
    assert by_id['cremad_1002_IEO_HAP_HI'].labels == ('happy',)


def test_fields_match_the_expected_shape(tmp_path: Path) -> None:
    _touch(tmp_path, '1001_DFA_ANG_XX.flv')
    example = cremad.load(tmp_path, ())[0]
    assert example.source_id == '1001'
    assert example.origin == 'cremad'
    assert example.kind is Kind.LAB_ER
    assert example.split is Split.BUILD
    assert example.label_set == ('anger', 'disgust', 'fear', 'happy', 'neutral', 'sad')
    assert example.media_path == str(tmp_path / 'VideoFlash' / '1001_DFA_ANG_XX.flv')


def test_unrecognized_files_are_skipped(tmp_path: Path) -> None:
    _touch(tmp_path, '1001_DFA_ANG_XX.flv')
    _touch(tmp_path, 'readme.txt')
    _touch(tmp_path, '1001_DFA_XXX_XX.flv')
    examples = cremad.load(tmp_path, ())
    assert len(examples) == 1


def test_missing_clip_directory_contributes_nothing(tmp_path: Path) -> None:
    assert cremad.load(tmp_path, ()) == ()
