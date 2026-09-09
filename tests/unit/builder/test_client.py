from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from emoharness.builder import client as client_module
from emoharness.builder.client import (
    Builder,
    FakeBuilder,
    GeminiBuilder,
    OpenAIBuilder,
    _mime_type_for,
    parse_proposal,
)
from emoharness.builder.config import BuilderConfig, BuilderPricing
from emoharness.builder.schema import BuilderProposal
from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.common.ledger import read_entries

Proposal = Callable[..., BuilderProposal]

PAYLOAD: dict[str, Any] = {
    'summary': 'one change',
    'candidates': [
        {
            'name': 'c0',
            'targeted_cluster': 'anger | sadness | agree',
            'rationale': 'the audible cues are ignored',
            'files': [{'path': 'prompts/extract_visual.md', 'content': 'new text\n'}],
        }
    ],
}


class _FakeParts:
    @staticmethod
    def from_bytes(data: bytes, mime_type: str) -> dict[str, Any]:
        return {'bytes': len(data), 'mime_type': mime_type}


class _FakeTypes:
    Part = _FakeParts

    @staticmethod
    def GenerateContentConfig(**kwargs: Any) -> dict[str, Any]:  # noqa: N802
        return kwargs


class _FakeUsage:
    prompt_token_count = 1_000_000
    candidates_token_count = 500_000


class _FakeResponse:
    def __init__(self, text: str, usage: object | None) -> None:
        self.text = text
        self.usage_metadata = usage


class _FakeModels:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeGenai:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self.models = _FakeModels(response)

    def Client(self) -> _FakeGenai:  # noqa: N802 - mirrors google.genai
        return self


def _install(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse | Exception
) -> _FakeGenai:
    genai = _FakeGenai(response)
    monkeypatch.setattr(client_module, '_load_genai', lambda: (genai, _FakeTypes))
    return genai


def _config(**overrides: Any) -> BuilderConfig:
    return BuilderConfig(
        model_id='fake-gemini',
        provider='gemini',
        pricing=BuilderPricing(input_usd_per_million=2.0, output_usd_per_million=12.0),
        **overrides,
    )


def _openai_config(**overrides: Any) -> BuilderConfig:
    return BuilderConfig(
        model_id='fake-gpt',
        provider='openai',
        api_key_env='EMOHARNESS_TEST_OPENAI_KEY',
        endpoint='https://api.example.com/v1',
        pricing=BuilderPricing(input_usd_per_million=2.0, output_usd_per_million=12.0),
        **overrides,
    )


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


def _completion(content: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {'choices': [{'message': {'content': content}}]}
    if usage is not None:
        body['usage'] = usage
    return body


def _openai_builder(
    handler: _ScriptedHandler,
    ledger: Path,
    config: BuilderConfig | None = None,
    max_retries: int = 3,
) -> OpenAIBuilder:
    return OpenAIBuilder(
        config if config is not None else _openai_config(),
        ledger,
        'stage2-r0',
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=max_retries,
        backoff_s=0.0,
    )


@pytest.fixture
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_OPENAI_KEY', 'sk-test')


def test_fake_builder_answers_in_order(make_proposal: Proposal) -> None:
    first, second = make_proposal('a'), make_proposal('b')
    builder = FakeBuilder([first, second])
    assert isinstance(builder, Builder)
    assert builder.model_id == 'fake-builder'
    assert builder.propose('packet one', ()) is first
    assert builder.propose('packet two', ('f.jpg',)) is second
    assert [call.packet_text for call in builder.calls] == ['packet one', 'packet two']
    assert builder.calls[1].image_paths == ('f.jpg',)


def test_fake_builder_raises_when_exhausted(make_proposal: Proposal) -> None:
    builder = FakeBuilder([make_proposal('a')])
    builder.propose('one', ())
    with pytest.raises(EmoHarnessError, match='ran out of proposals'):
        builder.propose('two', ())
    assert len(builder.calls) == 2


def test_parse_proposal_accepts_plain_json() -> None:
    proposal = parse_proposal(json.dumps(PAYLOAD))
    assert proposal.candidates[0].name == 'c0'
    assert proposal.summary == 'one change'


def test_parse_proposal_accepts_fences_and_prose() -> None:
    text = 'Sure, here is my answer.\n```json\n' + json.dumps(PAYLOAD) + '\n```\nDone.'
    assert parse_proposal(text).candidates[0].files[0].path.endswith('.md')


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        ('no json here', 'no JSON object'),
        ('}{', 'no JSON object'),
        ('{not json}', 'no JSON object'),
        ('{"candidates": []}', 'does not match the schema'),
    ],
)
def test_parse_proposal_rejects_bad_responses(text: str, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        parse_proposal(text)


def test_gemini_builder_sends_text_and_images_and_bills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    genai = _install(monkeypatch, _FakeResponse(json.dumps(PAYLOAD), _FakeUsage()))
    image = tmp_path / 'frame.png'
    image.write_bytes(b'\x89PNG')
    ledger = tmp_path / 'ledger.jsonl'
    builder = GeminiBuilder(_config(), ledger, 'stage2-r0')
    proposal = builder.propose('the packet', (str(image),))

    assert proposal.candidates[0].name == 'c0'
    assert builder.model_id == 'fake-gemini'
    sent = genai.models.calls[0]
    assert sent['model'] == 'fake-gemini'
    assert sent['contents'][0] == 'the packet'
    assert sent['contents'][1] == {'bytes': 4, 'mime_type': 'image/png'}
    entries = read_entries(ledger)
    assert len(entries) == 1
    assert entries[0].role == 'builder'
    assert entries[0].experiment == 'stage2-r0'
    assert entries[0].input_tokens == 1_000_000
    assert entries[0].cost_usd == pytest.approx(2.0 + 6.0)


def test_gemini_builder_estimates_tokens_without_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, _FakeResponse(json.dumps(PAYLOAD), None))
    ledger = tmp_path / 'ledger.jsonl'
    GeminiBuilder(_config(), ledger, 'exp').propose('a' * 400, ())
    assert read_entries(ledger)[0].input_tokens == 100


def test_gemini_builder_reports_provider_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, RuntimeError('503 unavailable'))
    with pytest.raises(EmoHarnessError, match='builder call failed'):
        GeminiBuilder(_config(), tmp_path / 'l.jsonl', 'exp').propose('packet', ())
    assert not (tmp_path / 'l.jsonl').exists()


def test_gemini_builder_rejects_an_empty_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, _FakeResponse('', None))
    with pytest.raises(EmoHarnessError, match='empty response'):
        GeminiBuilder(_config(), tmp_path / 'l.jsonl', 'exp').propose('packet', ())


def test_gemini_builder_rejects_an_unreadable_keyframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, _FakeResponse(json.dumps(PAYLOAD), None))
    builder = GeminiBuilder(_config(), tmp_path / 'l.jsonl', 'exp')
    with pytest.raises(ValidationError, match='cannot read keyframe'):
        builder.propose('packet', (str(tmp_path / 'missing.png'),))


@pytest.mark.parametrize(
    ('name', 'mime'),
    [
        ('a.png', 'image/png'),
        ('a.JPG', 'image/jpeg'),
        ('a.jpeg', 'image/jpeg'),
        ('a.webp', 'image/webp'),
        ('a.bin', 'image/jpeg'),
    ],
)
def test_mime_types_cover_the_keyframe_formats(name: str, mime: str) -> None:
    assert _mime_type_for(Path(name)) == mime


@pytest.mark.usefixtures('openai_key')
class TestOpenAIBuilder:
    def test_sends_the_packet_and_keyframes_and_bills(self, tmp_path: Path) -> None:
        image = tmp_path / 'frame.jpg'
        image.write_bytes(b'\xff\xd8\xff\xe0')
        ledger = tmp_path / 'ledger.jsonl'
        handler = _ScriptedHandler(
            [
                httpx.Response(
                    200,
                    json=_completion(
                        json.dumps(PAYLOAD),
                        {'prompt_tokens': 1_000_000, 'completion_tokens': 500_000},
                    ),
                )
            ]
        )
        builder = _openai_builder(handler, ledger)

        proposal = builder.propose('the packet', (str(image),))

        assert isinstance(builder, Builder)
        assert builder.model_id == 'fake-gpt'
        assert proposal.candidates[0].name == 'c0'
        sent = handler.requests[0]
        assert sent.url == httpx.URL('https://api.example.com/v1/chat/completions')
        assert sent.headers['authorization'] == 'Bearer sk-test'
        body = json.loads(sent.content)
        assert body['model'] == 'fake-gpt'
        assert body['temperature'] == 0.0
        assert body['response_format'] == {'type': 'json_object'}
        content = body['messages'][0]['content']
        assert body['messages'][0]['role'] == 'user'
        assert content[0] == {'type': 'text', 'text': 'the packet'}
        assert content[1]['type'] == 'image_url'
        assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
        entries = read_entries(ledger)
        assert len(entries) == 1
        assert entries[0].role == 'builder'
        assert entries[0].model_id == 'fake-gpt'
        assert entries[0].experiment == 'stage2-r0'
        assert entries[0].input_tokens == 1_000_000
        assert entries[0].cost_usd == pytest.approx(2.0 + 6.0)

    def test_estimates_tokens_without_usage(self, tmp_path: Path) -> None:
        ledger = tmp_path / 'ledger.jsonl'
        handler = _ScriptedHandler(
            [httpx.Response(200, json=_completion(json.dumps(PAYLOAD)))]
        )

        _openai_builder(handler, ledger).propose('a' * 400, ())

        assert read_entries(ledger)[0].input_tokens == 100

    def test_retries_a_server_error_then_succeeds(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler(
            [
                httpx.Response(500),
                httpx.Response(200, json=_completion(json.dumps(PAYLOAD))),
            ]
        )

        proposal = _openai_builder(handler, tmp_path / 'l.jsonl').propose('packet', ())

        assert proposal.summary == 'one change'
        assert handler.call_count == 2

    def test_retries_a_timeout_then_gives_up(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler([httpx.ReadTimeout('slow')])
        builder = _openai_builder(handler, tmp_path / 'l.jsonl', max_retries=1)

        with pytest.raises(EmoHarnessError, match='builder call failed after'):
            builder.propose('packet', ())

        assert handler.call_count == 2
        assert not (tmp_path / 'l.jsonl').exists()

    def test_a_client_error_fails_without_retry(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler([httpx.Response(400, text='bad request')])
        builder = _openai_builder(handler, tmp_path / 'l.jsonl')

        with pytest.raises(EmoHarnessError, match='status 400'):
            builder.propose('packet', ())

        assert handler.call_count == 1

    def test_rejects_an_empty_response(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler([httpx.Response(200, json=_completion(''))])

        with pytest.raises(EmoHarnessError, match='empty response'):
            _openai_builder(handler, tmp_path / 'l.jsonl').propose('packet', ())

    def test_rejects_a_malformed_response(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler([httpx.Response(200, json={'choices': []})])

        with pytest.raises(EmoHarnessError, match='malformed builder response'):
            _openai_builder(handler, tmp_path / 'l.jsonl').propose('packet', ())

    def test_rejects_an_unreadable_keyframe(self, tmp_path: Path) -> None:
        handler = _ScriptedHandler(
            [httpx.Response(200, json=_completion(json.dumps(PAYLOAD)))]
        )
        builder = _openai_builder(handler, tmp_path / 'l.jsonl')

        with pytest.raises(ValidationError, match='cannot read keyframe'):
            builder.propose('packet', (str(tmp_path / 'missing.jpg'),))

    def test_requires_the_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('EMOHARNESS_TEST_OPENAI_KEY', raising=False)
        handler = _ScriptedHandler(
            [httpx.Response(200, json=_completion(json.dumps(PAYLOAD)))]
        )
        builder = _openai_builder(handler, tmp_path / 'l.jsonl')

        with pytest.raises(EmoHarnessError, match='EMOHARNESS_TEST_OPENAI_KEY'):
            builder.propose('packet', ())

        assert handler.call_count == 0


def test_parse_proposal_trims_an_overlong_candidate_name() -> None:
    """Luna wrote a 70-character name on 2026-09-08; the patch is still usable."""
    long_name = 'Affect-sensitive visual and vocal cue weighting for happy-vs-neutral calibration'
    text = json.dumps(
        {
            'summary': 's',
            'candidates': [
                {
                    'name': long_name,
                    'targeted_cluster': 'c',
                    'rationale': 'r',
                    'files': [{'path': 'prompts/verify.md', 'content': 'x'}],
                }
            ],
        }
    )
    proposal = parse_proposal(text)
    assert len(proposal.candidates[0].name) <= 64
    assert proposal.candidates[0].name.startswith('Affect-sensitive')
