from __future__ import annotations

from pathlib import Path

from emoharness.data.sources import iemocap
from emoharness.data.tasks import Kind, Split

_EVAL_TEXT = (
    '% Ses01F_impro01\n'
    '[6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]\n'
    'C-E1:\tNeutral;\t()\n'
    '\n'
    '[10.0000 - 11.0000]\tSes01F_impro01_F001\txxx\t[2.5000, 2.5000, 2.5000]\n'
    '[12.0000 - 13.0000]\tSes01F_impro01_M002\tang\t[3.5000, 3.5000, 3.5000]\n'
    '[14.0000 - 15.0000]\tSes01F_impro01_M003\tdis\t[3.5000, 3.5000, 3.5000]\n'
)


def _write_session(root: Path, session: int) -> None:
    eval_dir = root / f'Session{session}' / 'dialog' / 'EmoEvaluation'
    eval_dir.mkdir(parents=True)
    (eval_dir / 'Ses01F_impro01.txt').write_text(_EVAL_TEXT, encoding='utf-8')


def test_load_sessions_maps_codes_and_skips_xxx_and_dis(tmp_path: Path) -> None:
    _write_session(tmp_path, 1)
    examples = iemocap.load_sessions(tmp_path, (1,))
    ids = {e.example_id for e in examples}
    assert ids == {'iemocap_Ses01F_impro01_F000', 'iemocap_Ses01F_impro01_M002'}
    by_id = {e.example_id: e for e in examples}
    assert by_id['iemocap_Ses01F_impro01_F000'].labels == ('Neutral',)
    assert by_id['iemocap_Ses01F_impro01_M002'].labels == ('Anger',)


def test_label_set_is_the_nine_names_in_order() -> None:
    assert iemocap.LABELS == (
        'Anger',
        'Sadness',
        'Surprise',
        'Happiness',
        'Excited',
        'Fear',
        'Frustration',
        'Neutral',
        'Other',
    )


def test_fields_match_the_expected_shape(tmp_path: Path) -> None:
    _write_session(tmp_path, 1)
    example = iemocap.load_sessions(tmp_path, (1,))[0]
    assert example.source_id == 'Ses01_F'
    assert example.origin == 'iemocap'
    assert example.kind is Kind.LAB_ER
    assert example.split is Split.BUILD
    assert example.media_path == str(tmp_path / 'clips' / 'Ses01F_impro01_F000.mp4')


def test_load_is_the_string_session_entry_point_for_the_registry(
    tmp_path: Path,
) -> None:
    _write_session(tmp_path, 1)
    assert iemocap.load(tmp_path, ('1',)) == iemocap.load_sessions(tmp_path, (1,))


def test_missing_session_directory_contributes_nothing(tmp_path: Path) -> None:
    assert iemocap.load_sessions(tmp_path, (3,)) == ()


def test_non_standard_utterance_ids_fall_back_to_the_id_itself(tmp_path: Path) -> None:
    eval_dir = tmp_path / 'Session2' / 'dialog' / 'EmoEvaluation'
    eval_dir.mkdir(parents=True)
    (eval_dir / 'custom.txt').write_text(
        '[1.0000 - 2.0000]\tcustom_utt_1\tneu\t[2.5000, 2.5000, 2.5000]\n',
        encoding='utf-8',
    )
    example = iemocap.load_sessions(tmp_path, (2,))[0]
    assert example.source_id == 'custom_utt_1'
