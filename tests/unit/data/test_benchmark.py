from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file
from emoharness.data.benchmark import (
    RawAnnotation,
    annotation_report,
    build_benchmark_manifest,
    checksum_manifest,
    derive_source_id,
    load_task_annotations,
    to_example,
)
from emoharness.data.tasks import Split, Task

ER_WILD_LABELS = ('anger', 'happiness', 'sadness', 'neutral')


def _write_task_json(root: Path, task: Task, rows: list[dict[str, object]]) -> None:
    (root / f'{task.value}.json').write_text(json.dumps(rows), encoding='utf-8')


def _annotation(
    video_id: str = 'dia_1_utt_1.mp4', label: str = 'anger'
) -> RawAnnotation:
    return RawAnnotation(video_id=video_id, label=label, label_set=ER_WILD_LABELS)


def test_raw_annotation_requires_at_least_two_labels() -> None:
    with pytest.raises(PydanticValidationError):
        RawAnnotation(video_id='a.mp4', label='x', label_set=('only_one',))


class TestLoadTaskAnnotations:
    def test_reads_every_row(self, tmp_path: Path) -> None:
        _write_task_json(
            tmp_path,
            Task.ER_WILD,
            [
                {
                    'video_id': 'dia_1_utt_1.mp4',
                    'label': 'anger',
                    'label_set': list(ER_WILD_LABELS),
                },
                {
                    'video_id': 'dia_1_utt_2.mp4',
                    'label': 'happy',
                    'label_set': list(ER_WILD_LABELS),
                },
            ],
        )
        rows = load_task_annotations(tmp_path, Task.ER_WILD)
        assert rows == (
            RawAnnotation(
                video_id='dia_1_utt_1.mp4', label='anger', label_set=ER_WILD_LABELS
            ),
            RawAnnotation(
                video_id='dia_1_utt_2.mp4', label='happy', label_set=ER_WILD_LABELS
            ),
        )

    def test_missing_file_raises_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            load_task_annotations(tmp_path, Task.ER_LAB)

    def test_non_list_json_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / f'{Task.SA.value}.json').write_text('{}', encoding='utf-8')
        with pytest.raises(ValidationError, match='expected a JSON list'):
            load_task_annotations(tmp_path, Task.SA)

    def test_invalid_row_is_rejected_with_context(self, tmp_path: Path) -> None:
        _write_task_json(tmp_path, Task.SA, [{'video_id': 'a.mp4'}])
        with pytest.raises(ValidationError, match=r'SA\.json\[0\]'):
            load_task_annotations(tmp_path, Task.SA)


@pytest.mark.parametrize(
    ('video_id', 'expected'),
    [
        ('ER_Wild/dia_12_utt_3.mp4', 'dia_12'),
        ('SA/test_dia_7_utt_2.mp4', 'dia_7'),
        ('ER_Lab/Ses05M_script01_1_F034.mp4', 'Ses05M_script01_1'),
        ('SA/dQw4w9WgXcQ_3.mp4', 'dQw4w9WgXcQ'),
        ('FG_SA/aqgyfjt.mp4', 'aqgyfjt'),
    ],
)
def test_derive_source_id(video_id: str, expected: str) -> None:
    assert derive_source_id(video_id) == expected


class TestToExample:
    def test_builds_the_canonical_fields(self, tmp_path: Path) -> None:
        example = to_example(tmp_path, Task.ER_WILD, _annotation())
        assert example.example_id == 'mme_er_wild_dia_1_utt_1'
        assert example.source_id == 'dia_1'
        assert example.origin == 'mme_emotion'
        assert example.task is Task.ER_WILD
        assert example.split is Split.TEST
        assert example.media_path == str(tmp_path / 'dia_1_utt_1.mp4')
        assert example.media_hash is None
        assert example.label_set == ER_WILD_LABELS
        assert example.labels == ('anger',)

    def test_example_id_is_a_sanitized_stem(self, tmp_path: Path) -> None:
        example = to_example(
            tmp_path, Task.ER_WILD, _annotation(video_id='ER_Wild/dia 1.utt.mp4')
        )
        assert example.example_id == 'mme_er_wild_dia_1_utt'

    def test_multi_label_gold_uses_comma_for_ml_er(self, tmp_path: Path) -> None:
        labels = ('anger', 'disgust', 'fear', 'sadness')
        annotation = RawAnnotation(
            video_id='03419.mp4', label='anger,disgust', label_set=labels
        )
        example = to_example(tmp_path, Task.ML_ER, annotation)
        assert example.labels == ('anger', 'disgust')

    def test_multi_label_gold_uses_semicolon_for_fg_er(self, tmp_path: Path) -> None:
        labels = ('Anticipation', 'Confidence', 'Doubt/Confusion')
        annotation = RawAnnotation(
            video_id='e1.mp4', label='Anticipation;Confidence', label_set=labels
        )
        example = to_example(tmp_path, Task.FG_ER, annotation)
        assert example.labels == ('Anticipation', 'Confidence')

    def test_media_hash_is_none_unless_requested(self, tmp_path: Path) -> None:
        media = tmp_path / 'dia_1_utt_1.mp4'
        media.write_bytes(b'\x00' * 32)
        example = to_example(tmp_path, Task.ER_WILD, _annotation(), hash_media=False)
        assert example.media_hash is None

    def test_media_hash_is_computed_when_requested(self, tmp_path: Path) -> None:
        media = tmp_path / 'dia_1_utt_1.mp4'
        media.write_bytes(b'\x00' * 32)
        example = to_example(tmp_path, Task.ER_WILD, _annotation(), hash_media=True)
        assert example.media_hash == sha256_file(media)

    def test_hash_media_on_missing_file_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValidationError, match='cannot hash media'):
            to_example(tmp_path, Task.ER_WILD, _annotation(), hash_media=True)


class TestBuildBenchmarkManifest:
    def test_concatenates_every_requested_task(self, tmp_path: Path) -> None:
        _write_task_json(
            tmp_path,
            Task.ER_WILD,
            [
                {
                    'video_id': 'dia_1_utt_1.mp4',
                    'label': 'anger',
                    'label_set': list(ER_WILD_LABELS),
                }
            ],
        )
        _write_task_json(
            tmp_path,
            Task.SA,
            [
                {
                    'video_id': 'v1.mp4',
                    'label': 'positive',
                    'label_set': ['positive', 'negative', 'neutral'],
                },
                {
                    'video_id': 'v2.mp4',
                    'label': 'negative',
                    'label_set': ['positive', 'negative', 'neutral'],
                },
            ],
        )
        examples = build_benchmark_manifest(tmp_path, tasks=(Task.ER_WILD, Task.SA))
        assert len(examples) == 3
        assert [e.example_id for e in examples] == [
            'mme_er_wild_dia_1_utt_1',
            'mme_sa_v1',
            'mme_sa_v2',
        ]
        assert all(e.split is Split.TEST for e in examples)


class TestAnnotationReport:
    def test_counts_histograms_violations_and_missing_media(
        self, tmp_path: Path
    ) -> None:
        media = tmp_path / 'dia_1_utt_1.mp4'
        media.write_bytes(b'\x00' * 8)
        rows = [
            {
                'video_id': media.name,
                'label': 'Happy',
                'label_set': list(ER_WILD_LABELS),
            },
            {
                'video_id': media.name,
                'label': 'happiness',
                'label_set': list(ER_WILD_LABELS),
            },
            {
                'video_id': 'missing.mp4',
                'label': 'not_a_candidate',
                'label_set': list(ER_WILD_LABELS),
            },
        ]
        _write_task_json(tmp_path, Task.ER_WILD, rows)
        examples = build_benchmark_manifest(tmp_path, tasks=(Task.ER_WILD,))
        report = annotation_report(examples)
        assert report.task_counts[Task.ER_WILD] == 3
        assert report.label_histograms[Task.ER_WILD] == {
            'happiness': 2,
            'not_a_candidate': 1,
        }
        assert report.label_violation_counts[Task.ER_WILD] == 1
        assert report.missing_media_count == 1

    def test_empty_input_yields_empty_report(self) -> None:
        report = annotation_report(())
        assert report.task_counts == {}
        assert report.missing_media_count == 0


class TestChecksumManifest:
    def test_hashes_json_and_zip_but_skips_audio_evidence(self, tmp_path: Path) -> None:
        (tmp_path / 'ER_Lab.json').write_text('[]', encoding='utf-8')
        nested = tmp_path / 'archives'
        nested.mkdir()
        (nested / 'SA.zip').write_bytes(b'PK\x03\x04')
        excluded_dir = tmp_path / 'audio_clue'
        excluded_dir.mkdir()
        (excluded_dir / 'ER_Lab_qwen2_audio_clues.json').write_text(
            '[]', encoding='utf-8'
        )

        checksums = checksum_manifest(tmp_path)

        assert set(checksums) == {'ER_Lab.json', 'archives/SA.zip'}
        assert checksums['ER_Lab.json'] == sha256_file(tmp_path / 'ER_Lab.json')
        assert checksums['archives/SA.zip'] == sha256_file(nested / 'SA.zip')
