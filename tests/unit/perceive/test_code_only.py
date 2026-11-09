"""The strict profile cannot instantiate model tools."""

from pathlib import Path

import pytest
from pydantic import ValidationError as SchemaError

from emoharness.common.config import load_model
from emoharness.perceive.audio import NullDiarizer, NullTranscriber
from emoharness.perceive.cache import PreprocessConfig
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
