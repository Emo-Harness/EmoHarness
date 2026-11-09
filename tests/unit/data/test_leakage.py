from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.constants import (
    PHASH_HAMMING_THRESHOLD,
    TRANSCRIPT_JACCARD_THRESHOLD,
    TRANSCRIPT_MIN_WORDS,
)
from emoharness.common.errors import ValidationError
from emoharness.data.leakage import (
    DEFAULT_SOURCE_ID_RULES,
    LeakageFlag,
    LeakageInputs,
    LeakageReport,
    LeakageThresholds,
    drop_flagged,
    phash_image,
    run_leakage_checks,
    write_report,
)
from emoharness.data.manifest import Example
from emoharness.data.tasks import Kind, Split, Task

_UNMAPPED_ORIGIN = 'other_corpus'


def _construction_example(
    example_id: str, source_id: str = 's1', origin: str = _UNMAPPED_ORIGIN
) -> Example:
    return Example(
        example_id=example_id,
        source_id=source_id,
        origin=origin,
        kind=Kind.WILD_ER,
        split=Split.BUILD,
        media_path=f'media/{example_id}.mp4',
        prompt='dummy prompt',
        label_set=('anger', 'joy'),
        labels=('anger',),
    )


def _benchmark_example(
    example_id: str, source_id: str = 'b1', task: Task = Task.ER_WILD
) -> Example:
    return Example(
        example_id=example_id,
        source_id=source_id,
        origin='mme_emotion',
        task=task,
        split=Split.TEST,
        media_path=f'media/{example_id}.mp4',
        prompt='dummy prompt',
        label_set=('anger', 'joy'),
        labels=('anger',),
    )


def _inputs(
    construction: tuple[Example, ...] = (),
    benchmark: tuple[Example, ...] = (),
    media_hashes: dict[str, str] | None = None,
    phashes: dict[str, tuple[str, ...]] | None = None,
    transcripts: dict[str, str] | None = None,
    source_id_rules: dict[str, tuple[Task, ...]] | None = None,
) -> LeakageInputs:
    kwargs: dict[str, object] = {
        'construction': construction,
        'benchmark': benchmark,
        'media_hashes': media_hashes or {},
        'phashes': phashes or {},
        'transcripts': transcripts or {},
    }
    if source_id_rules is not None:
        kwargs['source_id_rules'] = source_id_rules
    return LeakageInputs.model_validate(kwargs)


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array, mode='RGB').save(path)


def _skip_without_scipy() -> None:
    pytest.importorskip(
        'scipy.fftpack',
        reason='imagehash.phash needs scipy.fftpack',
        exc_type=ImportError,
    )


class TestLeakageModels:
    def test_flagged_construction_ids_deduplicates(self) -> None:
        report = LeakageReport(
            flags=(
                LeakageFlag(
                    construction_id='c1',
                    benchmark_id='b1',
                    check='media_hash',
                    score=1.0,
                ),
                LeakageFlag(
                    construction_id='c1',
                    benchmark_id='b2',
                    check='source_id',
                    score=1.0,
                ),
                LeakageFlag(
                    construction_id='c2', benchmark_id='b1', check='phash', score=0.9
                ),
            ),
            checked_pairs=10,
            construction_count=2,
            benchmark_count=5,
            thresholds={'phash_hamming': 8.0},
        )
        assert report.flagged_construction_ids() == {'c1', 'c2'}

    def test_score_must_be_in_unit_interval(self) -> None:
        with pytest.raises(PydanticValidationError):
            LeakageFlag(
                construction_id='c', benchmark_id='b', check='media_hash', score=1.5
            )


class TestLeakageThresholds:
    def test_defaults_match_the_constants(self) -> None:
        thresholds = LeakageThresholds()
        assert thresholds.phash_hamming == PHASH_HAMMING_THRESHOLD
        assert thresholds.transcript_jaccard == TRANSCRIPT_JACCARD_THRESHOLD
        assert thresholds.transcript_min_words == TRANSCRIPT_MIN_WORDS

    def test_is_frozen(self) -> None:
        thresholds = LeakageThresholds()
        with pytest.raises(PydanticValidationError):
            thresholds.phash_hamming = 1  # type: ignore[misc]


class TestLeakageInputs:
    def test_source_id_rules_default_to_the_documented_table(self) -> None:
        assert _inputs().source_id_rules == DEFAULT_SOURCE_ID_RULES

    def test_is_frozen(self) -> None:
        inputs = _inputs()
        with pytest.raises(PydanticValidationError):
            inputs.media_hashes = {'x': 'y'}  # type: ignore[misc]


class TestPhashImage:
    def test_computes_a_hash_for_a_pillow_generated_jpeg(self, tmp_path: Path) -> None:
        _skip_without_scipy()
        frame = tmp_path / 'frame.jpg'
        _make_image(frame, seed=1)
        digest = phash_image(frame)
        assert digest
        assert all(c in '0123456789abcdef' for c in digest)

    def test_rejects_non_image_files(self, tmp_path: Path) -> None:
        bogus = tmp_path / 'not_an_image.png'
        bogus.write_text('hello', encoding='utf-8')
        with pytest.raises(ValidationError, match='cannot compute phash'):
            phash_image(bogus)


class TestRunLeakageChecksMediaHash:
    def test_fires_on_an_exact_match(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(construction, benchmark, media_hashes={'c1': 'same', 'b1': 'same'})
        )
        assert report.flags == (
            LeakageFlag(
                construction_id='c1', benchmark_id='b1', check='media_hash', score=1.0
            ),
        )

    def test_does_not_fire_on_a_mismatch(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction, benchmark, media_hashes={'c1': 'hash-a', 'b1': 'hash-b'}
            )
        )
        assert report.flags == ()

    def test_an_id_missing_from_the_mapping_has_nothing_to_compare(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(construction, benchmark, media_hashes={'b1': 'same'})
        )
        assert report.flags == ()


class TestRunLeakageChecksPhash:
    def test_fires_within_the_threshold(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                phashes={'c1': ('0' * 16,), 'b1': (('1' + '0' * 15),)},
            )
        )
        assert len(report.flags) == 1
        flag = report.flags[0]
        assert flag.check == 'phash'
        assert flag.score == pytest.approx(1 - 1 / 64)

    def test_does_not_fire_beyond_the_threshold(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction, benchmark, phashes={'c1': ('0' * 16,), 'b1': ('f' * 16,)}
            )
        )
        assert report.flags == ()

    def test_empty_hash_tuples_are_never_flagged(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(construction, benchmark, phashes={'c1': (), 'b1': ('0' * 16,)})
        )
        assert report.flags == ()

    def test_thresholds_respected(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        inputs = _inputs(
            construction,
            benchmark,
            phashes={'c1': ('0' * 16,), 'b1': (('1' + '0' * 15),)},
        )
        loose = run_leakage_checks(inputs, LeakageThresholds(phash_hamming=8))
        strict = run_leakage_checks(inputs, LeakageThresholds(phash_hamming=0))
        assert len(loose.flags) == 1
        assert strict.flags == ()

    def test_real_jpegs_of_the_same_scene_are_flagged(self, tmp_path: Path) -> None:
        _skip_without_scipy()
        same_a = tmp_path / 'same_a.jpg'
        same_b = tmp_path / 'same_b.jpg'
        _make_image(same_a, seed=1)
        _make_image(same_b, seed=1)
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                phashes={'c1': (phash_image(same_a),), 'b1': (phash_image(same_b),)},
            )
        )
        assert {flag.check for flag in report.flags} == {'phash'}

    def test_real_jpegs_of_different_scenes_are_not_flagged(
        self, tmp_path: Path
    ) -> None:
        _skip_without_scipy()
        a = tmp_path / 'a.jpg'
        b = tmp_path / 'b.jpg'
        _make_image(a, seed=1)
        _make_image(b, seed=2)
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                phashes={'c1': (phash_image(a),), 'b1': (phash_image(b),)},
            )
        )
        assert report.flags == ()


class TestRunLeakageChecksTranscript:
    def test_fires_on_near_duplicate_transcripts(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                transcripts={
                    'c1': 'the quick brown fox jumps over the lazy dog',
                    'b1': 'the quick brown fox jumped over a lazy dog',
                },
            ),
            LeakageThresholds(transcript_jaccard=0.5),
        )
        assert len(report.flags) == 1
        assert report.flags[0].check == 'transcript'

    def test_does_not_fire_below_the_jaccard_threshold(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                transcripts={
                    'c1': 'completely unrelated sentence about cars and roads',
                    'b1': 'a totally different topic involving cooking recipes',
                },
            )
        )
        assert report.flags == ()

    def test_short_transcripts_are_never_compared(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                transcripts={'c1': 'hi there', 'b1': 'hi there'},
            ),
            LeakageThresholds(transcript_jaccard=0.1, transcript_min_words=5),
        )
        assert report.flags == ()

    def test_two_empty_token_sets_are_never_flagged(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        report = run_leakage_checks(
            _inputs(construction, benchmark, transcripts={'c1': '...', 'b1': '???'}),
            LeakageThresholds(transcript_jaccard=0.0, transcript_min_words=0),
        )
        assert report.flags == ()

    def test_thresholds_respected(self) -> None:
        construction = (_construction_example('c1'),)
        benchmark = (_benchmark_example('b1'),)
        inputs = _inputs(
            construction,
            benchmark,
            transcripts={
                'c1': 'the quick brown fox jumps over the lazy dog',
                'b1': 'the quick brown fox jumped over a lazy dog',
            },
        )
        loose = run_leakage_checks(inputs, LeakageThresholds(transcript_jaccard=0.5))
        strict = run_leakage_checks(inputs, LeakageThresholds(transcript_jaccard=0.99))
        assert len(loose.flags) == 1
        assert strict.flags == ()


class TestRunLeakageChecksSourceId:
    def test_default_rules_flag_meld_against_wild_sa_and_ir(self) -> None:
        construction = (
            _construction_example(
                'meld_train_dia5_utt0', source_id='meld_train_dia5', origin='meld'
            ),
        )
        benchmark = (
            _benchmark_example(
                'mme_er_wild_000001', source_id='meld_train_dia5', task=Task.ER_WILD
            ),
        )
        report = run_leakage_checks(_inputs(construction, benchmark))
        assert report.flags == (
            LeakageFlag(
                construction_id='meld_train_dia5_utt0',
                benchmark_id='mme_er_wild_000001',
                check='source_id',
                score=1.0,
            ),
        )

    def test_unmatched_task_is_not_flagged(self) -> None:
        construction = (
            _construction_example('chsims_video_0001_0001', 'video_0001', 'chsims'),
        )
        benchmark = (
            _benchmark_example('mme_er_lab_000001', 'video_0001', Task.ER_LAB),
        )
        report = run_leakage_checks(_inputs(construction, benchmark))
        assert report.flags == ()

    def test_origin_absent_from_rules_is_never_flagged(self) -> None:
        construction = (_construction_example('cremad_1', 'shared_id', 'cremad'),)
        benchmark = (_benchmark_example('mme_er_lab_000001', 'shared_id', Task.ER_LAB),)
        report = run_leakage_checks(_inputs(construction, benchmark))
        assert report.flags == ()

    def test_non_benchmark_items_on_the_benchmark_side_are_ignored(self) -> None:
        construction = (_construction_example('c1', 'shared_id', 'meld'),)
        not_a_benchmark_item = _construction_example(
            'not_benchmark', 'shared_id', 'meld'
        )
        report = run_leakage_checks(_inputs(construction, (not_a_benchmark_item,)))
        assert report.flags == ()

    def test_custom_rules_override_the_default(self) -> None:
        construction = (_construction_example('c1', 'shared_id', 'cremad'),)
        benchmark = (_benchmark_example('mme_er_lab_000001', 'shared_id', Task.ER_LAB),)
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                source_id_rules={'cremad': (Task.ER_LAB,)},
            )
        )
        assert len(report.flags) == 1
        assert report.flags[0].check == 'source_id'


class TestRunLeakageChecksCombined:
    def test_combines_every_check_and_records_thresholds(self) -> None:
        construction = (
            _construction_example('iemocap_x', source_id='Ses01_F', origin='iemocap'),
        )
        benchmark = (
            _benchmark_example(
                'mme_er_lab_000001', source_id='Ses01_F', task=Task.ER_LAB
            ),
        )
        report = run_leakage_checks(
            _inputs(
                construction,
                benchmark,
                media_hashes={'iemocap_x': 'same', 'mme_er_lab_000001': 'same'},
            )
        )
        checks = {flag.check for flag in report.flags}
        assert checks == {'media_hash', 'source_id'}
        assert report.construction_count == 1
        assert report.benchmark_count == 1
        assert report.checked_pairs == 1
        assert report.thresholds['phash_hamming'] == float(PHASH_HAMMING_THRESHOLD)
        assert report.thresholds['transcript_jaccard'] == TRANSCRIPT_JACCARD_THRESHOLD
        assert report.thresholds['transcript_min_words'] == float(TRANSCRIPT_MIN_WORDS)

    def test_empty_inputs_yield_an_empty_report(self) -> None:
        report = run_leakage_checks(_inputs())
        assert report.flags == ()
        assert report.checked_pairs == 0
        assert report.construction_count == 0
        assert report.benchmark_count == 0


class TestDropFlagged:
    def test_removes_only_flagged_examples(self) -> None:
        keep = _construction_example('keep_me', 's1', 'meld')
        drop = _construction_example('drop_me', 's2', 'meld')
        report = LeakageReport(
            flags=(
                LeakageFlag(
                    construction_id='drop_me',
                    benchmark_id='b',
                    check='source_id',
                    score=1.0,
                ),
            ),
            checked_pairs=1,
            construction_count=2,
            benchmark_count=1,
            thresholds={},
        )
        assert drop_flagged((keep, drop), report) == (keep,)


class TestWriteReport:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        report = LeakageReport(
            flags=(
                LeakageFlag(
                    construction_id='c', benchmark_id='b', check='media_hash', score=1.0
                ),
            ),
            checked_pairs=1,
            construction_count=1,
            benchmark_count=1,
            thresholds={'phash_hamming': 8.0},
        )
        path = tmp_path / 'leakage_report.json'
        write_report(path, report)
        loaded = json.loads(path.read_text(encoding='utf-8'))
        assert LeakageReport.model_validate(loaded) == report
