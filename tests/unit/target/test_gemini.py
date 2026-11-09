"""Tests for emoharness.target.gemini, using FakeGenAIClient (no network, no SDK)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from emoharness.common.errors import EmoHarnessError, TargetError, ValidationError
from emoharness.common.ledger import read_entries
from emoharness.target.base import TargetRequest
from emoharness.target.gemini import GeminiConfig, GeminiDirectTarget
from emoharness.target.pricing import Pricing

from .fake_genai import (
    FakeGenAIClient,
    FakeGenerateContentResponse,
    FakeUploadedFile,
    FakeUsageMetadata,
)


@pytest.fixture
def gemini_config() -> GeminiConfig:
    return GeminiConfig(
        model_id='gemini-3.1-pro',
        temperature=0.0,
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
    )


def test_model_id_property(gemini_config: GeminiConfig) -> None:
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='native_video',
        client_factory=FakeGenAIClient,
    )

    assert target.model_id == 'gemini-3.1-pro'


def test_native_video_inlines_small_file_and_orders_media_before_text(
    gemini_config: GeminiConfig, make_video: Callable[..., Path]
) -> None:
    video = make_video()
    fake_client = FakeGenAIClient()
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='native_video',
        client_factory=lambda: fake_client,
    )

    response = target.generate(
        TargetRequest(prompt='Describe the clip.', video_path=str(video))
    )

    assert response.text == '<answer>joy</answer>'
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    (call,) = fake_client.models.calls
    assert call['model'] == 'gemini-3.1-pro'
    assert call['config'] == {'temperature': 0.0}
    (content,) = call['contents']
    assert content['role'] == 'user'
    assert 'inline_data' in content['parts'][0]
    assert content['parts'][0]['inline_data']['mime_type'] == 'video/mp4'
    assert content['parts'][-1] == {'text': 'Describe the clip.'}
    assert fake_client.files.uploads == []


def test_native_video_uploads_large_file(
    gemini_config: GeminiConfig,
    make_video: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('emoharness.target.gemini._INLINE_MEDIA_MAX_BYTES', 4)
    video = make_video(extra_bytes=64)
    uploaded = FakeUploadedFile(uri='files/big-video', mime_type='video/mp4')
    fake_client = FakeGenAIClient(uploaded_file=uploaded)
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='native_video',
        client_factory=lambda: fake_client,
    )

    target.generate(TargetRequest(prompt='Describe.', video_path=str(video)))

    assert fake_client.files.uploads == [str(video)]
    (call,) = fake_client.models.calls
    (content,) = call['contents']
    assert content['parts'][0] == {
        'file_data': {'file_uri': 'files/big-video', 'mime_type': 'video/mp4'}
    }


def test_native_video_requires_video_path(gemini_config: GeminiConfig) -> None:
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='native_video',
        client_factory=FakeGenAIClient,
    )

    with pytest.raises(ValidationError):
        target.generate(TargetRequest(prompt='Describe.'))


def test_matched_protocol_builds_image_parts_then_text(
    gemini_config: GeminiConfig, make_image: Callable[..., Path]
) -> None:
    image_a = make_image('a.jpg')
    image_b = make_image('b.jpg')
    fake_client = FakeGenAIClient()
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    target.generate(
        TargetRequest(
            prompt='Speech present: yes\nTranscript: hi',
            image_paths=(str(image_a), str(image_b)),
        )
    )

    (call,) = fake_client.models.calls
    (content,) = call['contents']
    parts = content['parts']
    assert len(parts) == 3
    assert 'inline_data' in parts[0]
    assert 'inline_data' in parts[1]
    assert parts[2] == {'text': 'Speech present: yes\nTranscript: hi'}


def test_matched_protocol_ignores_video_path(
    gemini_config: GeminiConfig, make_video: Callable[..., Path]
) -> None:
    video = make_video()
    fake_client = FakeGenAIClient()
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    target.generate(TargetRequest(prompt='p', video_path=str(video)))

    (call,) = fake_client.models.calls
    (content,) = call['contents']
    assert content['parts'] == [{'text': 'p'}]
    assert fake_client.files.uploads == []


def test_ledger_entry_written_with_the_teacher_direct_role(
    tmp_path: Path, gemini_config: GeminiConfig
) -> None:
    ledger_path = tmp_path / 'ledger.jsonl'
    fake_client = FakeGenAIClient()
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=ledger_path,
        experiment='h3-gemini',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    target.generate(TargetRequest(prompt='p'))

    (entry,) = read_entries(ledger_path)
    assert entry.role == 'teacher_direct'
    assert entry.model_id == 'gemini-3.1-pro'
    assert entry.experiment == 'h3-gemini'
    assert entry.input_tokens == 10
    assert entry.output_tokens == 5
    assert entry.cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1_000_000)


def test_no_ledger_written_when_path_is_none(gemini_config: GeminiConfig) -> None:
    fake_client = FakeGenAIClient()
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    response = target.generate(TargetRequest(prompt='p'))

    assert response.text == '<answer>joy</answer>'


def test_malformed_response_raises_target_error(gemini_config: GeminiConfig) -> None:
    fake_client = FakeGenAIClient()
    fake_client.models.response = object()  # type: ignore[assignment]
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    with pytest.raises(TargetError):
        target.generate(TargetRequest(prompt='p'))


def test_custom_response_id_does_not_affect_returned_response(
    gemini_config: GeminiConfig,
) -> None:
    response = FakeGenerateContentResponse(
        text='<answer>anger</answer>',
        usage_metadata=FakeUsageMetadata(
            prompt_token_count=1, candidates_token_count=1
        ),
        response_id='resp-42',
    )
    fake_client = FakeGenAIClient(response=response)
    target = GeminiDirectTarget(
        gemini_config,
        ledger_path=None,
        experiment='e',
        mode='matched_protocol',
        client_factory=lambda: fake_client,
    )

    result = target.generate(TargetRequest(prompt='p'))

    assert result.text == '<answer>anger</answer>'


def test_missing_google_genai_package_raises_emoharness_error(
    gemini_config: GeminiConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, 'google.genai', None)
    target = GeminiDirectTarget(
        gemini_config, ledger_path=None, experiment='e', mode='matched_protocol'
    )

    with pytest.raises(EmoHarnessError):
        target.generate(TargetRequest(prompt='p'))
