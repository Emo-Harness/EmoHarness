"""Timed utterances: an opt-in perception channel that must leave v1 untouched."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from emoharness.common.config import load_model
from emoharness.data.manifest import Example
from emoharness.harness.schema import AudioEvidence, PerceivedInputs, Utterance
from emoharness.perceive import pipeline as pipeline_module
from emoharness.perceive.audio import TimedText
from emoharness.perceive.cache import (
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
    UtteranceConfig,
    UtteranceRecord,
)
from emoharness.perceive.descriptors import ReferenceStats
from emoharness.perceive.pipeline import PerceptionTools, preprocess_clip

#: The hash every cached v1 clip and every earlier experiment is keyed by.
V1_HASH = '6ce8cc9178cefb39a0f109c5d10586a2e2041550a7ac515e8851c63c11cb2c00'
LOUD = ReferenceStats(
    mean={'loudness_sma3_amean': 0.0}, std={'loudness_sma3_amean': 1.0}
)


@pytest.fixture
def v1(repo_root: Path) -> PreprocessConfig:
    return load_model(repo_root / 'configs' / 'preprocess.yaml', PreprocessConfig)


@pytest.fixture
def v2(v1: PreprocessConfig) -> PreprocessConfig:
    return v1.model_copy(
        update={
            'utterances': UtteranceConfig(
                reference_stats='ref.json', min_seconds=0.5, speakers='ecapa_cluster'
            )
        }
    )


@pytest.fixture
def no_video_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def frames(
        _p: Path, timestamps: Sequence[float], out: Path, prefix: str
    ) -> tuple[Path, ...]:
        out.mkdir(parents=True, exist_ok=True)
        paths = tuple(out / f'{prefix}_{i:04d}.jpg' for i in range(len(timestamps)))
        for path in paths:
            path.write_bytes(b'jpeg')
        return paths

    monkeypatch.setattr(pipeline_module, 'probe_duration', lambda _p: 4.0)
    monkeypatch.setattr(pipeline_module, 'extract_frames', frames)
    monkeypatch.setattr(pipeline_module, 'extract_audio_track', lambda _p, w: w)


def test_the_v1_config_hashes_as_it_did_before_utterances_existed(
    v1: PreprocessConfig, v2: PreprocessConfig
) -> None:
    assert v1.config_hash() == V1_HASH
    assert v2.config_hash() != V1_HASH


def test_a_v1_record_serialises_without_the_new_fields() -> None:
    audio = AudioEvidence(speech_present=True, transcript='hi', prosody=('loud voice',))
    assert 'utterances' not in audio.model_dump(mode='json')
    clip = PreprocessedClip(
        media_hash='m',
        config_hash='c',
        duration_seconds=1.0,
        audio=audio,
        transcript_words=1,
    )
    assert 'utterances' not in clip.model_dump(mode='json')


def test_utterances_need_a_transcriber(v1: PreprocessConfig) -> None:
    with pytest.raises(ValueError, match='need a transcriber'):
        PreprocessConfig.model_validate(
            {
                **v1.model_dump(mode='json'),
                'asr': {'model': 'x', 'backend': 'none'},
                'utterances': {'reference_stats': 'ref.json'},
            }
        )


def test_each_segment_keeps_its_span_features_and_voice(
    benchmark_example: Example,
    v2: PreprocessConfig,
    tmp_path: Path,
    no_video_io: None,
) -> None:
    tools = PerceptionTools.fakes(
        transcript_text='Hi there. Fine.',
        prosody_features={'loudness_sma3_amean': 3.0},
        timed=(
            TimedText(text='Hi there.', start=0.0, end=1.5),
            TimedText(text='Fine.', start=1.6, end=1.9),  # shorter than min_seconds
        ),
        speaker_labels=('S1', 'S2'),
    )
    clip = preprocess_clip(
        benchmark_example, v2, tools, PreprocessCache(tmp_path / 'cache')
    )
    assert clip.utterances == (
        UtteranceRecord(
            start=0.0,
            end=1.5,
            text='Hi there.',
            speaker='S1',
            features={'loudness_sma3_amean': 3.0},
        ),
        UtteranceRecord(start=1.6, end=1.9, text='Fine.'),
    )


def test_a_v1_config_caches_no_utterances(
    benchmark_example: Example, v1: PreprocessConfig, tmp_path: Path, no_video_io: None
) -> None:
    tools = PerceptionTools.fakes(
        transcript_text='Hi.', timed=(TimedText(text='Hi.', start=0.0, end=1.0),)
    )
    clip = preprocess_clip(
        benchmark_example, v1, tools, PreprocessCache(tmp_path / 'cache')
    )
    assert clip.utterances == ()


def test_phrases_are_rendered_when_the_clip_is_read() -> None:
    clip = PreprocessedClip(
        media_hash='m',
        config_hash='c',
        duration_seconds=1.5,
        audio=AudioEvidence(speech_present=True, transcript='Hi.'),
        transcript_words=1,
        utterances=(
            UtteranceRecord(
                start=0.0,
                end=1.5,
                text='Hi.',
                speaker='S1',
                features={'loudness_sma3_amean': 3.0},
            ),
        ),
    )
    with_ref = clip.to_perceived('e', 'v.mp4', LOUD)
    without = clip.to_perceived('e', 'v.mp4')
    assert with_ref.audio.utterances == (
        Utterance(
            start=0.0, end=1.5, text='Hi.', speaker='S1', prosody=('loud voice',)
        ),
    )
    assert without.audio.utterances[0].prosody == ()


def test_the_text_block_lists_utterances_instead_of_the_flat_transcript() -> None:
    inputs = PerceivedInputs(
        example_id='e',
        video_path='v.mp4',
        duration_seconds=4.0,
        audio=AudioEvidence(
            speech_present=True,
            transcript='Hi. Fine.',
            prosody=('loud voice',),
            utterances=(
                Utterance(
                    start=0.0,
                    end=1.5,
                    text='Hi.',
                    speaker='S1',
                    prosody=('high pitch',),
                ),
                Utterance(start=1.6, end=1.9, text='Fine.'),
            ),
        ),
    )
    assert inputs.audio_text().splitlines() == [
        'Speech present: yes',
        'Utterances:',
        '[0.0-1.5s] S1: Hi. (high pitch)',
        '[1.6-1.9s]: Fine.',
        'Prosody: loud voice',
    ]
