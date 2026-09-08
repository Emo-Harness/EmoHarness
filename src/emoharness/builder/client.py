"""Builder clients. The loop owns the control flow; a client only answers a packet.

Swapping the Builder model means swapping one class here, because the contract is a
JSON schema (:mod:`emoharness.builder.schema`) and nothing else. The default
teacher answers over the OpenAI chat-completions API (:class:`OpenAIBuilder`);
:class:`GeminiBuilder` is the alternative provider.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from emoharness.builder.config import BuilderConfig
from emoharness.builder.packet import estimate_tokens
from emoharness.builder.schema import BuilderProposal
from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.common.jsonish import extract_json_object
from emoharness.common.ledger import LedgerEntry, record
from emoharness.target.media import encode_image_data_uri

BUILDER_ROLE: Final = 'builder'
DEFAULT_MAX_RETRIES: Final = 3
DEFAULT_BACKOFF_S: Final = 1.0
_MIME_TYPES: Final[dict[str, str]] = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
}
_DEFAULT_MIME: Final = 'image/jpeg'


@runtime_checkable
class Builder(Protocol):
    """Anything that turns one diagnosis packet into one proposal."""

    @property
    def model_id(self) -> str: ...

    def propose(
        self, packet_text: str, image_paths: tuple[str, ...]
    ) -> BuilderProposal: ...


class BuilderCall(BaseModel):
    """What a client was asked, kept so tests can assert on the packet."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    packet_text: str
    image_paths: tuple[str, ...] = ()


class FakeBuilder:
    """Returns canned proposals in order and records every packet it was given."""

    def __init__(
        self,
        proposals: Sequence[BuilderProposal],
        model_id: str = 'fake-builder',
    ) -> None:
        self._proposals = tuple(proposals)
        self._model_id = model_id
        self._calls: list[BuilderCall] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def calls(self) -> tuple[BuilderCall, ...]:
        return tuple(self._calls)

    def propose(
        self, packet_text: str, image_paths: tuple[str, ...]
    ) -> BuilderProposal:
        index = len(self._calls)
        self._calls.append(
            BuilderCall(packet_text=packet_text, image_paths=tuple(image_paths))
        )
        if index >= len(self._proposals):
            raise EmoHarnessError(
                f'FakeBuilder ran out of proposals after {len(self._proposals)} calls'
            )
        return self._proposals[index]


def parse_proposal(text: str) -> BuilderProposal:
    """Parse a Builder response, tolerating code fences and leading prose."""
    payload = extract_json_object(text)
    if payload is None:
        raise ValidationError('builder response contains no JSON object')
    try:
        return BuilderProposal.model_validate(payload)
    except PydanticValidationError as exc:
        raise ValidationError(
            f'builder response does not match the schema: {exc}'
        ) from exc


class _RetryableError(Exception):
    """Internal signal that a call failed in a way that may be retried."""


class OpenAIBuilder:
    """OpenAI Builder client. One call per round, one ledger entry per call.

    The packet is the user message's text part and the keyframes are
    ``image_url`` data URIs beside it; the answer is asked for as a JSON object
    and parsed with :func:`parse_proposal`. Transient failures (5xx, timeouts)
    are retried with exponential backoff, as in
    :class:`emoharness.target.openai_compat.OpenAICompatTarget`.
    """

    def __init__(
        self,
        config: BuilderConfig,
        ledger_path: Path,
        experiment: str,
        client: httpx.Client | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
    ) -> None:
        self._config = config
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._client = client if client is not None else httpx.Client()
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._url = f'{config.endpoint.rstrip("/")}/chat/completions'

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def propose(
        self, packet_text: str, image_paths: tuple[str, ...]
    ) -> BuilderProposal:
        payload = self._payload(packet_text, image_paths)
        started = time.perf_counter()
        body = self._post_with_retries(payload)
        latency_s = time.perf_counter() - started
        text, usage_in, usage_out = _parse_completion(body)
        if not text:
            raise EmoHarnessError('builder returned an empty response')
        self._record(
            usage_in if usage_in else estimate_tokens(packet_text),
            usage_out if usage_out else estimate_tokens(text),
            latency_s,
        )
        return parse_proposal(text)

    def _payload(
        self, packet_text: str, image_paths: tuple[str, ...]
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{'type': 'text', 'text': packet_text}]
        content.extend(_image_url_part(path) for path in image_paths)
        return {
            'model': self._config.model_id,
            'messages': [{'role': 'user', 'content': content}],
            'temperature': self._config.temperature,
            'response_format': {'type': 'json_object'},
        }

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self._config.api_key_env, '')
        if not api_key:
            raise EmoHarnessError(
                f'environment variable {self._config.api_key_env} is not set'
            )
        return {'Authorization': f'Bearer {api_key}'}

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._max_retries + 1
        last_error: Exception = EmoHarnessError('builder call never attempted')
        for attempt in range(attempts):
            try:
                return self._post_once(payload)
            except _RetryableError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(self._backoff_s * (2**attempt))
        raise EmoHarnessError(
            f'builder call failed after {attempts} attempts: {last_error}'
        ) from last_error

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(
                self._url, json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise _RetryableError(str(exc)) from exc
        if response.status_code >= 500:
            raise _RetryableError(f'server error {response.status_code}')
        if response.status_code >= 400:
            raise EmoHarnessError(
                f'builder call failed with status {response.status_code}: '
                f'{response.text}'
            )
        body: dict[str, Any] = response.json()
        return body

    def _record(self, input_tokens: int, output_tokens: int, latency_s: float) -> None:
        record(
            self._ledger_path,
            LedgerEntry(
                role=BUILDER_ROLE,
                model_id=self._config.model_id,
                experiment=self._experiment,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency_s,
                cost_usd=self._config.pricing.cost_usd(input_tokens, output_tokens),
            ),
        )


def _image_url_part(path: str) -> dict[str, Any]:
    try:
        url = encode_image_data_uri(path)
    except OSError as exc:
        raise ValidationError(f'cannot read keyframe {path}: {exc}') from exc
    return {'type': 'image_url', 'image_url': {'url': url}}


def _parse_completion(body: dict[str, Any]) -> tuple[str, int, int]:
    """The answer text and the provider's token counts (0 when it reported none)."""
    try:
        text = str(body['choices'][0]['message']['content'] or '')
    except (KeyError, IndexError, TypeError) as exc:
        raise EmoHarnessError(f'malformed builder response: {exc}') from exc
    usage = body.get('usage') or {}
    return (
        text,
        int(usage.get('prompt_tokens', 0)),
        int(usage.get('completion_tokens', 0)),
    )


def _load_genai() -> tuple[Any, Any]:
    """Import ``google-genai`` lazily; the package is an optional dependency."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise EmoHarnessError(
            'google-genai is required for GeminiBuilder; '
            'install it with `poetry install --with api`'
        ) from exc
    return genai, types


class GeminiBuilder:
    """Gemini Builder client. One call per round, one ledger entry per call."""

    def __init__(
        self, config: BuilderConfig, ledger_path: Path, experiment: str
    ) -> None:
        self._config = config
        self._ledger_path = ledger_path
        self._experiment = experiment

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def propose(
        self, packet_text: str, image_paths: tuple[str, ...]
    ) -> BuilderProposal:
        genai, types = _load_genai()
        contents = [packet_text, *(_image_part(types, p) for p in image_paths)]
        started = time.perf_counter()
        response = self._generate(genai, types, contents)
        latency_s = time.perf_counter() - started
        text = getattr(response, 'text', None)
        if not text:
            raise EmoHarnessError('builder returned an empty response')
        self._record(response, packet_text, text, latency_s)
        return parse_proposal(text)

    def _generate(self, genai: Any, types: Any, contents: list[Any]) -> Any:
        client = genai.Client()
        try:
            return client.models.generate_content(
                model=self._config.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self._config.temperature,
                    response_mime_type='application/json',
                ),
            )
        except Exception as exc:  # provider errors are opaque; re-raise as ours
            raise EmoHarnessError(f'builder call failed: {exc}') from exc

    def _record(
        self, response: Any, packet_text: str, text: str, latency_s: float
    ) -> None:
        input_tokens, output_tokens = _token_counts(response, packet_text, text)
        record(
            self._ledger_path,
            LedgerEntry(
                role=BUILDER_ROLE,
                model_id=self._config.model_id,
                experiment=self._experiment,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency_s,
                cost_usd=self._config.pricing.cost_usd(input_tokens, output_tokens),
            ),
        )


def _token_counts(response: Any, packet_text: str, text: str) -> tuple[int, int]:
    """Provider counts when present, otherwise a four-characters-per-token estimate."""
    usage = getattr(response, 'usage_metadata', None)
    prompt_tokens = getattr(usage, 'prompt_token_count', None)
    output_tokens = getattr(usage, 'candidates_token_count', None)
    return (
        int(prompt_tokens) if prompt_tokens else estimate_tokens(packet_text),
        int(output_tokens) if output_tokens else estimate_tokens(text),
    )


def _image_part(types: Any, path: str) -> Any:
    file = Path(path)
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise ValidationError(f'cannot read keyframe {path}: {exc}') from exc
    return types.Part.from_bytes(data=data, mime_type=_mime_type_for(file))


def _mime_type_for(path: Path) -> str:
    return _MIME_TYPES.get(path.suffix.lower(), _DEFAULT_MIME)
