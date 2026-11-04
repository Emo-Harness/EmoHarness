"""Deterministic Targets for tests and ``--fake`` runs, priced like real ones.

``FakeTarget`` answers and records; ``RecordingTarget`` wraps any Target so the
calls it makes reach the cost ledger; ``scripted_target`` composes the two into
the Target a ``--fake`` run uses. What the answers say is the responder's
business, and a responder belongs to the seam it exercises - the harness ships
its own in :mod:`emoharness.harness.fake_responder`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from emoharness.common.ledger import LedgerEntry, record
from emoharness.target.base import TargetModel, TargetRequest, TargetResponse
from emoharness.target.pricing import Pricing, estimate_cost

Responder = Callable[[TargetRequest], str]

#: Nominal price list, so ``--fake`` runs exercise the ledger and the cost table.
FAKE_PRICING: Final = Pricing(input_per_million=0.5, output_per_million=1.5)
FAKE_MODEL_ID: Final = 'fake-target'
FAKE_TEACHER_ID: Final = 'fake-teacher'
FAKE_EXPERIMENT: Final = 'fake'


class FakeTarget:
    """Answers with ``responder(request)`` or a fixed text; records every request.

    Token counts are derived from text length so budget accounting can be tested.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        default_text: str = '<answer>neutral</answer>',
        model_id: str = FAKE_MODEL_ID,
    ) -> None:
        self._responder = responder
        self._default_text = default_text
        self._model_id = model_id
        self._requests: list[TargetRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def requests(self) -> tuple[TargetRequest, ...]:
        return tuple(self._requests)

    @property
    def call_count(self) -> int:
        return len(self._requests)

    def generate(self, request: TargetRequest) -> TargetResponse:
        started = time.perf_counter()
        self._requests.append(request)
        text = self._responder(request) if self._responder else self._default_text
        return TargetResponse(
            text=text,
            model_id=self._model_id,
            input_tokens=max(1, len(request.prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_s=time.perf_counter() - started,
        )


class RecordingTarget:
    """Wraps a Target so every call it actually makes lands in the cost ledger."""

    def __init__(
        self,
        inner: TargetModel,
        ledger_path: Path | None,
        experiment: str,
        pricing: Pricing = FAKE_PRICING,
    ) -> None:
        self._inner = inner
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._pricing = pricing

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def generate(self, request: TargetRequest) -> TargetResponse:
        """Call the wrapped Target and append one ledger entry."""
        response = self._inner.generate(request)
        if self._ledger_path is not None:
            record(
                self._ledger_path,
                LedgerEntry(
                    role='target',
                    model_id=response.model_id,
                    experiment=self._experiment,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_s=response.latency_s,
                    cost_usd=estimate_cost(
                        self._pricing, response.input_tokens, response.output_tokens
                    ),
                ),
            )
        return response


def scripted_target(
    responder: Responder,
    ledger_path: Path | None = None,
    experiment: str = FAKE_EXPERIMENT,
    model_id: str = FAKE_MODEL_ID,
) -> TargetModel:
    """A Target that answers with ``responder`` and records what its calls cost."""
    inner = FakeTarget(responder=responder, model_id=model_id)
    return RecordingTarget(inner, ledger_path, experiment)
