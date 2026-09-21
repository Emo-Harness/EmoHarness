"""The strict profile cannot instantiate model tools or keep cached ASR evidence."""

from pathlib import Path

import pytest
from pydantic import ValidationError as SchemaError

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.harness.schema import AudioEvidence
from emoharness.perceive.audio import NullDiarizer, NullTranscriber
from emoharness.perceive.cache import PreprocessConfig, PreprocessedClip
from emoharness.perceive.projection import without_asr
from emoharness.runner.paths import ProjectPaths
from emoharness.runner.perception import real_perception_tools


def test_strict_tools_never_construct_whisper_or_diarizer(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object) -> None:
        raise AssertionError('model tool was constructed')

    monkeypatch.setattr('emoharness.runner.perception.WhisperTranscriber', forbidden)
    monkeypatch.setattr('emoharness.runner.perception.PyannoteDiarizer', forbidden)
    config = load_model(
        repo_root / 'configs/preprocess_code_only.yaml', PreprocessConfig
    )
    tools = real_perception_tools(ProjectPaths.from_root(repo_root), config)
    assert isinstance(tools.transcriber, NullTranscriber)
    assert isinstance(tools.diarizer, NullDiarizer)
    assert tools.transcriber.transcribe(Path('never-opened.wav')).available is False
    payload = config.model_dump()
    payload['diarization']['backend'] = 'pyannote'
    with pytest.raises(SchemaError, match='cannot invoke'):
        PreprocessConfig.model_validate(payload)


def test_projection_drops_speech_and_preserves_only_compatible_code_artifacts(
    repo_root: Path,
) -> None:
    source = load_model(repo_root / 'configs/preprocess.yaml', PreprocessConfig)
    target = load_model(
        repo_root / 'configs/preprocess_code_only.yaml', PreprocessConfig
    )
    clip = PreprocessedClip(
        media_hash='clip',
        config_hash=source.config_hash(),
        duration_seconds=1,
        frame_paths=('frame.jpg',),
        transcript_words=2,
        audio=AudioEvidence(
            speech_present=True,
            transcript='secret speech',
            language='en',
            prosody=('high pitch',),
        ),
    )
    projected = without_asr(clip, source, target, audio_present=True)
    assert projected.audio.transcript == '' and projected.audio.language is None
    assert projected.audio.speech_present is None
    assert projected.audio.prosody == clip.audio.prosody
    assert projected.frame_paths == clip.frame_paths
    assert projected.config_hash != clip.config_hash
    assert clip.audio.transcript == 'secret speech'
    altered = target.model_copy(
        update={'frames': target.frames.model_copy(update={'fps': 3.0})}
    )
    with pytest.raises(ValidationError, match='do not match'):
        without_asr(clip, source, altered, audio_present=True)
