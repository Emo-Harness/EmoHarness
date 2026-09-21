"""Target client for any OpenAI-compatible ``/chat/completions`` endpoint.

The same client serves locally-served vLLM Targets and the hosted OpenAI API
that answers the teacher row: an ``api_key_env`` on the config adds a bearer
token, and ``video.frames_as_images`` sends the sampled frames as images for an
endpoint that takes no video.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Final, Literal

import httpx

from emoharness.common.errors import (
    TargetError,
    TargetUnavailableError,
    ValidationError,
)
from emoharness.common.ledger import LedgerEntry, record
from emoharness.target.base import TargetRequest, TargetResponse
from emoharness.target.config import TargetConfig
from emoharness.target.media import content_parts, default_remux_dir, playable_video
from emoharness.target.pricing import estimate_cost

# vLLM's Qwen video processor reads frame-sampling kwargs from this top-level
# request field (the openai client calls the same thing `extra_body`); posting
# raw JSON, we set it directly.
MM_PROCESSOR_KWARGS_FIELD: Final = 'mm_processor_kwargs'

#: Ledger role of an ordinary Target call; the teacher row bills its own role.
TARGET_ROLE: Final = 'target'

#: The name every JSON-schema response format is registered under. Both vLLM
#: and the OpenAI API require ``json_schema`` to be ``{name, schema}``, not the
#: bare schema.
JSON_SCHEMA_NAME: Final = 'response'

#: Sampling parameters the GPT-5 family only accepts at ``reasoning_effort: none``.
_NO_REASONING: Final = 'none'

#: Generation can take a minute on a busy server; httpx's 5 s default would turn
#: every long answer into a retry storm.
_DEFAULT_TIMEOUT: Final = httpx.Timeout(600.0, connect=30.0)

#: How many requests carrying a video may be in flight at once, process-wide.
#: The server decodes and tensorises every video in its frontend process, in
#: host RAM; on an 8 GB box a dozen at once is an OOM. Text-only requests (the
#: harness's later stages) are not gated.
VIDEO_WORKERS_ENV: Final = 'EMOHARNESS_VIDEO_WORKERS'
DEFAULT_VIDEO_WORKERS: Final = 4


def _video_gate() -> threading.BoundedSemaphore:
    raw = os.environ.get(VIDEO_WORKERS_ENV, '').strip()
    count = int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_VIDEO_WORKERS
    return threading.BoundedSemaphore(count)


_VIDEO_GATE = _video_gate()


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
        media_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._media_dir = media_dir if media_dir is not None else default_remux_dir()
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._client = (
            client if client is not None else httpx.Client(timeout=_DEFAULT_TIMEOUT)
        )
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
        carries_video = (
            bool(request.video_path) and not self._config.video.frames_as_images
        )
        started = time.perf_counter()
        with _VIDEO_GATE if carries_video else nullcontext():
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
        if request.video_path and not video.frames_as_images:
            try:
                playable = playable_video(request.video_path, self._media_dir)
            except ValidationError as exc:
                # Surface as a call failure so the per-item fallbacks apply.
                raise TargetError(str(exc)) from exc
            request = request.model_copy(update={'video_path': str(playable)})
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
            **self._generation_params(request),
        }
        if not video.frames_as_images:
            # vLLM-only frame-sampling knob: an endpoint that takes no video
            # rejects unknown top-level fields, and samples nothing anyway.
            payload[MM_PROCESSOR_KWARGS_FIELD] = {
                'fps': video.fps,
                'max_frames': video.max_frames,
            }
        if self._config.chat_template_kwargs:
            # Thinking models otherwise spend the whole token budget
            # deliberating and never reach the answer tag: Qwen3.5-4B hit
            # max_tokens on 15 of 24 smoke items with no </answer> at all.
            payload['chat_template_kwargs'] = dict(self._config.chat_template_kwargs)
        if request.json_schema is not None:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': JSON_SCHEMA_NAME,
                    'schema': dict(request.json_schema),
                },
            }
        return payload

    def _generation_params(self, request: TargetRequest) -> dict[str, Any]:
        """Length and sampling fields, spelled the way this endpoint's family takes them."""
        sampling = {
            'temperature': request.temperature,
            'top_p': request.top_p,
            'seed': request.seed,
        }
        if self._config.api_flavor == 'standard':
            return {**sampling, 'max_tokens': request.max_tokens}
        effort = self._config.reasoning_effort
        params: dict[str, Any] = {
            'max_completion_tokens': request.max_tokens,
            'reasoning_effort': effort,
        }
        if effort == _NO_REASONING:
            params.update(sampling)
        return params

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
        if not self._healthy():
            raise TargetUnavailableError(
                f'target endpoint {self._url} is down or its engine is dead: {last_error}'
            ) from last_error
        raise TargetError(
            f'target call to {self._url} failed after {attempts} attempts: {last_error}'
        ) from last_error

    def _healthy(self) -> bool:
        """vLLM's ``/health`` fails once its engine has died; a hosted API has no such route and counts as healthy."""
        base = self._config.endpoint.rstrip('/')
        root = base[: -len('/v1')] if base.endswith('/v1') else base
        try:
            response = self._client.get(f'{root}/health', timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 500

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
