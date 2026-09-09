"""Tests for emoharness.target.openai_compat, using httpx.MockTransport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from emoharness.common.errors import TargetError, TargetUnavailableError
from emoharness.common.ledger import read_entries
from emoharness.target.base import TargetRequest
from emoharness.target.config import TargetConfig
from emoharness.target.openai_compat import (
    JSON_SCHEMA_NAME,
    MM_PROCESSOR_KWARGS_FIELD,
    OpenAICompatTarget,
)

_OK_BODY = {
    'choices': [{'message': {'content': '<answer>joy</answer>'}}],
    'usage': {'prompt_tokens': 42, 'completion_tokens': 7},
}


class _ScriptedHandler:
    """Replays a fixed script of Response objects or exceptions; the last repeats."""

    def __init__(self, script: Sequence[httpx.Response | Exception]) -> None:
        self._script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        index = min(len(self.requests), len(self._script) - 1)
        self.requests.append(request)
        item = self._script[index]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def call_count(self) -> int:
        return len(self.requests)


def _client_for(handler: _ScriptedHandler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _target(
    target_config: TargetConfig,
    handler: _ScriptedHandler,
    ledger_path: Path | None = None,
    **kwargs: Any,
) -> OpenAICompatTarget:
    return OpenAICompatTarget(
        config=target_config,
        ledger_path=ledger_path,
        experiment='exp1',
        client=_client_for(handler),
        backoff_s=0.0,
        **kwargs,
    )


def test_happy_path_parses_text_and_usage(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler)

    response = target.generate(TargetRequest(prompt='describe'))

    assert response.text == '<answer>joy</answer>'
    assert response.input_tokens == 42
    assert response.output_tokens == 7
    assert response.model_id == target_config.model_id
    assert response.cached is False
    assert response.latency_s >= 0.0
    assert handler.call_count == 1


def test_model_id_property(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler)

    assert target.model_id == target_config.model_id


def test_request_body_shape_without_system_or_schema(
    target_config: TargetConfig,
) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler)

    target.generate(TargetRequest(prompt='describe', max_tokens=99, seed=7))

    sent = handler.requests[0]
    assert sent.url == httpx.URL(f'{target_config.endpoint}/chat/completions')
    body = _json_body(sent)
    assert body['model'] == target_config.model_id
    assert body['max_tokens'] == 99
    assert body['seed'] == 7
    assert len(body['messages']) == 1
    assert body['messages'][0]['role'] == 'user'
    assert 'response_format' not in body
    assert body[MM_PROCESSOR_KWARGS_FIELD] == {
        'fps': target_config.video.fps,
        'max_frames': target_config.video.max_frames,
    }


def test_request_includes_system_message_when_given(
    target_config: TargetConfig,
) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler)

    target.generate(TargetRequest(prompt='describe', system='You are an expert.'))

    body = _json_body(handler.requests[0])
    assert body['messages'][0] == {'role': 'system', 'content': 'You are an expert.'}
    assert body['messages'][1]['role'] == 'user'


def test_json_schema_mode_sets_response_format(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler)
    schema = {'type': 'object', 'properties': {'face': {'type': 'array'}}}

    target.generate(TargetRequest(prompt='describe', json_schema=schema))

    body = _json_body(handler.requests[0])
    # vLLM and the OpenAI API both want {name, schema}, never the bare schema.
    assert body['response_format'] == {
        'type': 'json_schema',
        'json_schema': {'name': JSON_SCHEMA_NAME, 'schema': schema},
    }


def test_openai_reasoning_flavor_spells_length_and_effort(
    target_config: TargetConfig,
) -> None:
    """GPT-5 rejects max_tokens and, above effort none, every sampling field."""
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)] * 2)
    quiet = target_config.model_copy(
        update={'api_flavor': 'openai_reasoning', 'reasoning_effort': 'none'}
    )
    thinking = quiet.model_copy(update={'reasoning_effort': 'high'})

    _target(quiet, handler).generate(TargetRequest(prompt='p', max_tokens=99, seed=7))
    _target(thinking, handler).generate(TargetRequest(prompt='p', max_tokens=99))

    first, second = (_json_body(r) for r in handler.requests)
    assert 'max_tokens' not in first and first['max_completion_tokens'] == 99
    assert first['reasoning_effort'] == 'none'
    assert first['temperature'] == 0.0 and first['seed'] == 7
    assert second['reasoning_effort'] == 'high'
    assert not {'temperature', 'top_p', 'seed', 'max_tokens'} & second.keys()


def test_frames_as_images_drops_the_video_part(
    target_config: TargetConfig,
    make_video: Callable[..., Path],
    make_image: Callable[..., Path],
) -> None:
    config = target_config.model_copy(
        update={
            'video': target_config.video.model_copy(update={'frames_as_images': True})
        }
    )
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(config, handler)

    target.generate(
        TargetRequest(
            prompt='describe',
            video_path=str(make_video()),
            image_paths=(str(make_image()),),
        )
    )

    body = _json_body(handler.requests[0])
    content = body['messages'][0]['content']
    assert [part['type'] for part in content] == ['text', 'image_url']
    assert MM_PROCESSOR_KWARGS_FIELD not in body


def test_an_api_key_env_sends_a_bearer_token(
    target_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_TARGET_KEY', 'sk-target')
    config = target_config.model_copy(
        update={'api_key_env': 'EMOHARNESS_TEST_TARGET_KEY'}
    )
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])

    _target(config, handler).generate(TargetRequest(prompt='describe'))

    assert handler.requests[0].headers['authorization'] == 'Bearer sk-target'


def test_a_missing_api_key_fails_before_the_call(
    target_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('EMOHARNESS_TEST_TARGET_KEY', raising=False)
    config = target_config.model_copy(
        update={'api_key_env': 'EMOHARNESS_TEST_TARGET_KEY'}
    )
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])

    with pytest.raises(TargetError, match='EMOHARNESS_TEST_TARGET_KEY'):
        _target(config, handler).generate(TargetRequest(prompt='describe'))

    assert handler.call_count == 0


def test_no_authorization_header_without_an_api_key_env(
    target_config: TargetConfig,
) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])

    _target(target_config, handler).generate(TargetRequest(prompt='describe'))

    assert 'authorization' not in handler.requests[0].headers


def test_a_teacher_row_bills_its_own_ledger_role(
    tmp_path: Path, target_config: TargetConfig
) -> None:
    ledger_path = tmp_path / 'ledger.jsonl'
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(
        target_config, handler, ledger_path=ledger_path, role='teacher_direct'
    )

    target.generate(TargetRequest(prompt='describe'))

    (entry,) = read_entries(ledger_path)
    assert entry.role == 'teacher_direct'


def test_video_mode_file_url_is_passed_through(
    target_config: TargetConfig, make_video: Callable[..., Path]
) -> None:
    video = make_video()
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler, video_mode='file_url')

    target.generate(TargetRequest(prompt='describe', video_path=str(video)))

    body = _json_body(handler.requests[0])
    content = body['messages'][0]['content']
    video_part = next(part for part in content if part['type'] == 'video_url')
    assert video_part['video_url']['url'].startswith('file://')


def test_retries_after_server_error_then_succeeds(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler(
        [httpx.Response(500), httpx.Response(200, json=_OK_BODY)]
    )
    target = _target(target_config, handler, max_retries=2)

    response = target.generate(TargetRequest(prompt='describe'))

    assert response.text == '<answer>joy</answer>'
    assert handler.call_count == 2


def test_retries_after_timeout_then_succeeds(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler(
        [httpx.ReadTimeout('slow'), httpx.Response(200, json=_OK_BODY)]
    )
    target = _target(target_config, handler, max_retries=2)

    response = target.generate(TargetRequest(prompt='describe'))

    assert response.text == '<answer>joy</answer>'
    assert handler.call_count == 2


def test_exhausts_retries_then_asks_health(target_config: TargetConfig) -> None:
    """After the retries a /health probe decides between a failed call and a dead server."""
    handler = _ScriptedHandler(
        [httpx.Response(500), httpx.Response(500), httpx.Response(200, json={})]
    )
    target = _target(target_config, handler, max_retries=1)

    with pytest.raises(TargetError) as caught:
        target.generate(TargetRequest(prompt='describe'))

    assert not isinstance(caught.value, TargetUnavailableError)
    assert handler.call_count == 3  # two attempts, then the health probe
    assert handler.requests[-1].url.path == '/health'


def test_dead_engine_is_unavailable_not_a_failed_call(
    target_config: TargetConfig,
) -> None:
    """vLLM keeps answering /v1 with 500s after its engine dies; /health says so."""
    handler = _ScriptedHandler([httpx.Response(500)])
    target = _target(target_config, handler, max_retries=1)

    with pytest.raises(TargetUnavailableError):
        target.generate(TargetRequest(prompt='describe'))


def test_client_error_fails_without_retry(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(400, text='bad request')])
    target = _target(target_config, handler, max_retries=3)

    with pytest.raises(TargetError):
        target.generate(TargetRequest(prompt='describe'))

    assert handler.call_count == 1


def test_malformed_response_raises_target_error(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json={'choices': []})])
    target = _target(target_config, handler)

    with pytest.raises(TargetError):
        target.generate(TargetRequest(prompt='describe'))


def test_ledger_entry_written_on_success(
    tmp_path: Path, target_config: TargetConfig
) -> None:
    ledger_path = tmp_path / 'ledger.jsonl'
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler, ledger_path=ledger_path)

    target.generate(TargetRequest(prompt='describe'))

    (entry,) = read_entries(ledger_path)
    assert entry.role == 'target'
    assert entry.model_id == target_config.model_id
    assert entry.experiment == 'exp1'
    assert entry.input_tokens == 42
    assert entry.output_tokens == 7
    expected_cost = (42 * target_config.pricing.input_per_million) + (
        7 * target_config.pricing.output_per_million
    )
    assert entry.cost_usd == pytest.approx(expected_cost / 1_000_000)


def test_no_ledger_written_when_path_is_none(target_config: TargetConfig) -> None:
    handler = _ScriptedHandler([httpx.Response(200, json=_OK_BODY)])
    target = _target(target_config, handler, ledger_path=None)

    response = target.generate(TargetRequest(prompt='describe'))

    assert response.text == '<answer>joy</answer>'


def _json_body(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content)
    return body
