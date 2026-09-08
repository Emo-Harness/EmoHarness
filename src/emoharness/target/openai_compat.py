"""Target client for any OpenAI-compatible ``/chat/completions`` endpoint.

The same client serves locally-served vLLM Targets and the hosted OpenAI API
that answers the teacher row: an ``api_key_env`` on the config adds a bearer
token, and ``video.frames_as_images`` sends the sampled frames as images for an
endpoint that takes no video.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Final, Literal

import httpx

from emoharness.common.errors import TargetError
from emoharness.common.ledger import LedgerEntry, record
from emoharness.target.base import TargetRequest, TargetResponse
from emoharness.target.config import TargetConfig
from emoharness.target.media import content_parts
from emoharness.target.pricing import estimate_cost

# vLLM's Qwen video processor reads frame-sampling kwargs from this top-level
# request field (the openai client calls the same thing `extra_body`); posting
# raw JSON, we set it directly.
MM_PROCESSOR_KWARGS_FIELD: Final = 'mm_processor_kwargs'

#: Ledger role of an ordinary Target call; the teacher row bills its own role.
TARGET_ROLE: Final = 'target'


class _RetryableError(Exception):
    """Internal signal that a call failed in a way that may be retried."""


class OpenAICompatTarget:
    """Calls an OpenAI-compatible chat endpoint and records cost to the ledger."""

    def __init__(
        self,
        config: TargetConfig,
        ledger_path: Path | None,
        experiment: str,
        client: httpx.Client | None = None,
        video_mode: Literal['data_uri', 'file_url'] = 'data_uri',
        max_retries: int = 3,
        backoff_s: float = 1.0,
        role: str = TARGET_ROLE,
    ) -> None:
        self._config = config
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._client = client if client is not None else httpx.Client()
        self._video_mode = video_mode
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._role = role
        self._url = f'{config.endpoint.rstrip("/")}/chat/completions'

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def generate(self, request: TargetRequest) -> TargetResponse:
        """Send one chat completion request, retrying transient failures."""
        payload = self._build_payload(request)
        started = time.perf_counter()
        body = self._post_with_retries(payload)
        latency_s = time.perf_counter() - started
        text, input_tokens, output_tokens = _parse_response(body)
        response = TargetResponse(
            text=text,
            model_id=self._config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency_s,
        )
        self._record_ledger(response)
        return response

    def _build_payload(self, request: TargetRequest) -> dict[str, Any]:
        video = self._config.video
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({'role': 'system', 'content': request.system})
        messages.append(
            {
                'role': 'user',
                'content': content_parts(
                    request, self._video_mode, video.frames_as_images
                ),
            }
        )
        payload: dict[str, Any] = {
            'model': self._config.model_id,
            'messages': messages,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'seed': request.seed,
            'max_tokens': request.max_tokens,
        }
        if not video.frames_as_images:
            # vLLM-only frame-sampling knob: an endpoint that takes no video
            # rejects unknown top-level fields, and samples nothing anyway.
            payload[MM_PROCESSOR_KWARGS_FIELD] = {
                'fps': video.fps,
                'max_frames': video.max_frames,
            }
        if request.json_schema is not None:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': dict(request.json_schema),
            }
        return payload

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._max_retries + 1
        last_error: Exception = TargetError('target call never attempted')
        for attempt in range(attempts):
            try:
                return self._post_once(payload)
            except _RetryableError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(self._backoff_s * (2**attempt))
        raise TargetError(
            f'target call to {self._url} failed after {attempts} attempts: {last_error}'
        ) from last_error

    def _headers(self) -> dict[str, str]:
        """Bearer token for a hosted endpoint; a local vLLM config names no key."""
        key_env = self._config.api_key_env
        if key_env is None:
            return {}
        api_key = os.environ.get(key_env, '')
        if not api_key:
            raise TargetError(f'environment variable {key_env} is not set')
        return {'Authorization': f'Bearer {api_key}'}

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
            raise TargetError(
                f'target call to {self._url} failed with status '
                f'{response.status_code}: {response.text}'
            )
        body: dict[str, Any] = response.json()
        return body

    def _record_ledger(self, response: TargetResponse) -> None:
        if self._ledger_path is None:
            return
        entry = LedgerEntry(
            role=self._role,
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


def _parse_response(body: dict[str, Any]) -> tuple[str, int, int]:
    try:
        text = str(body['choices'][0]['message']['content'])
        usage = body['usage']
        input_tokens = int(usage['prompt_tokens'])
        output_tokens = int(usage['completion_tokens'])
    except (KeyError, IndexError, TypeError) as exc:
        raise TargetError(f'malformed target response: {exc}') from exc
    return text, input_tokens, output_tokens
