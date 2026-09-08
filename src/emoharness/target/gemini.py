"""Gemini teacher client: native-video and matched-protocol upper bounds (ADR-0008).

Gemini is the alternative teacher provider, and the only one that can take a
clip as native video; the default teacher answers over the OpenAI-compatible
client instead (:mod:`emoharness.target.openai_compat`). Lazily imports ``google.genai`` so the rest of the package works without the
optional ``api`` dependency group installed. Request bodies are built as plain
dicts matching the Gemini API shape, so tests exercise request building with a
fake client and never the real SDK (ADR-0008 point 4: responses are logged
with response ids because the API is not fully deterministic).
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.errors import EmoHarnessError, TargetError, ValidationError
from emoharness.common.ledger import LedgerEntry, record
from emoharness.common.logging import get_logger
from emoharness.target.base import TargetRequest, TargetResponse
from emoharness.target.config import TargetConfig
from emoharness.target.media import image_mime_type, video_mime_type
from emoharness.target.pricing import Pricing, estimate_cost

# Gemini's inline request payload limit; larger videos must go through the
# Files API instead of being base64-inlined into the request body.
_INLINE_MEDIA_MAX_BYTES: Final = 20 * 1024 * 1024

#: Ledger role every teacher Direct call is billed under, whatever the provider.
TEACHER_DIRECT_ROLE: Final = 'teacher_direct'

#: Environment variable a Gemini teacher config falls back to for its API key.
DEFAULT_GEMINI_KEY_ENV: Final = 'GEMINI_API_KEY'

_logger = get_logger(__name__)

GenAIClientFactory = Callable[[], Any]


class GeminiConfig(BaseModel):
    """Gemini teacher call parameters (ADR-0011 point 8: temperature 0)."""

    model_config = ConfigDict(frozen=True, extra='forbid', protected_namespaces=())

    model_id: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    pricing: Pricing
    api_key_env: str = Field(default=DEFAULT_GEMINI_KEY_ENV, min_length=1)


def gemini_config_from(config: TargetConfig) -> GeminiConfig:
    """The Gemini call parameters carried by a ``provider: gemini`` target config."""
    return GeminiConfig(
        model_id=config.model_id,
        temperature=config.decoding.temperature,
        pricing=config.pricing,
        api_key_env=config.api_key_env or DEFAULT_GEMINI_KEY_ENV,
    )


class GeminiDirectTarget:
    """Answers a ``TargetRequest`` directly with Gemini, bypassing the harness."""

    def __init__(
        self,
        config: GeminiConfig,
        ledger_path: Path | None,
        experiment: str,
        mode: Literal['native_video', 'matched_protocol'],
        client_factory: GenAIClientFactory | None = None,
    ) -> None:
        self._config = config
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._mode = mode
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def generate(self, request: TargetRequest) -> TargetResponse:
        """Send one Gemini call for the configured mode (native_video or matched_protocol)."""
        client = self._client_instance()
        contents = self._build_contents(request, client)
        started = time.perf_counter()
        raw_response = client.models.generate_content(
            model=self._config.model_id,
            contents=contents,
            config={'temperature': self._config.temperature},
        )
        latency_s = time.perf_counter() - started
        text, input_tokens, output_tokens, response_id = _parse_response(raw_response)
        response = TargetResponse(
            text=text,
            model_id=self._config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency_s,
        )
        self._log_and_record(response, response_id)
        return response

    def _client_instance(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _default_client_factory(self) -> Any:  # pragma: no cover - needs the real SDK
        try:
            from google import genai
        except ImportError as exc:
            raise EmoHarnessError(
                'google-genai is required for GeminiDirectTarget; install the '
                "'api' extra (poetry install --with api)"
            ) from exc
        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise EmoHarnessError(
                f'environment variable {self._config.api_key_env} is not set'
            )
        return genai.Client(api_key=api_key)

    def _build_contents(
        self, request: TargetRequest, client: Any
    ) -> list[dict[str, Any]]:
        if self._mode == 'native_video':
            parts = self._video_parts(request, client)
        else:
            parts = _image_parts(request)
        parts.append(_text_part(request.prompt))
        return [{'role': 'user', 'parts': parts}]

    def _video_parts(self, request: TargetRequest, client: Any) -> list[dict[str, Any]]:
        if not request.video_path:
            raise ValidationError('native_video mode requires request.video_path')
        path = Path(request.video_path)
        mime = video_mime_type(path)
        if path.stat().st_size <= _INLINE_MEDIA_MAX_BYTES:
            return [_inline_part(path, mime)]
        uploaded = client.files.upload(file=str(path))
        return [_file_part(uploaded.uri, uploaded.mime_type)]

    def _log_and_record(
        self, response: TargetResponse, response_id: str | None
    ) -> None:
        _logger.info(
            'teacher_direct response id=%s model=%s mode=%s tokens=%d/%d',
            response_id,
            self._config.model_id,
            self._mode,
            response.input_tokens,
            response.output_tokens,
        )
        if self._ledger_path is None:
            return
        entry = LedgerEntry(
            role=TEACHER_DIRECT_ROLE,
            model_id=self._config.model_id,
            experiment=self._experiment,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_s=response.latency_s,
            cost_usd=estimate_cost(
                self._config.pricing, response.input_tokens, response.output_tokens
            ),
        )
        record(self._ledger_path, entry)


def _text_part(text: str) -> dict[str, str]:
    return {'text': text}


def _inline_part(path: Path, mime: str) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return {'inline_data': {'mime_type': mime, 'data': encoded}}


def _file_part(uri: str, mime: str) -> dict[str, Any]:
    return {'file_data': {'file_uri': uri, 'mime_type': mime}}


def _image_parts(request: TargetRequest) -> list[dict[str, Any]]:
    return [
        _inline_part(Path(image_path), image_mime_type(image_path))
        for image_path in request.image_paths
    ]


def _parse_response(response: Any) -> tuple[str, int, int, str | None]:
    try:
        text = str(response.text)
        usage = response.usage_metadata
        input_tokens = int(usage.prompt_token_count)
        output_tokens = int(usage.candidates_token_count)
    except AttributeError as exc:
        raise TargetError(f'malformed gemini response: {exc}') from exc
    response_id = getattr(response, 'response_id', None)
    return text, input_tokens, output_tokens, response_id
