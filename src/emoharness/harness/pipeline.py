"""The four-stage inference pipeline (README §4.3, ADR-0011).

One item costs four Target calls when nothing goes wrong and never more than the
hard cap, retries included. When the cap is reached the remaining model stages are
skipped and the controller completes the answer from whatever exists; it never
refuses. No branch here is keyed on a task name: the only thing the pipeline knows
about an item is what ``parse_prompt`` reads from its prompt.
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import (
    CALL_BUDGET_HARD_CAP,
    LABEL_SEPARATOR,
    MULTILABEL_CAP,
    RETRIES_PER_STAGE,
    STEP_TITLES,
    VIDEO_MAX_FRAMES,
)
from emoharness.common.errors import ValidationError
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example
from emoharness.data.prompt_template import ParsedPrompt, parse_prompt
from emoharness.harness._controller import MODEL_WRITTEN_STEPS, build_output
from emoharness.harness._evidence import build_record
from emoharness.harness._stages import (
    StageOutcome,
    agreement_from_payload,
    hypothesis_set_from_payload,
    run_json_stage,
    step_texts_from_payload,
    uniform_hypothesis_set,
    visual_from_payload,
)
from emoharness.harness._verification import apply_bundle_verification
from emoharness.harness.bundle_models import STAGE_NAMES
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.schema import (
    EvidenceRecord,
    HarnessOutput,
    HypothesisSet,
    PerceivedInputs,
    StageTrace,
    VerificationResult,
    VisualEvidence,
)
from emoharness.target.base import TargetModel, TargetRequest

__all__ = [
    'HarnessPipeline',
    'PipelineConfig',
]

DEFAULT_MAX_TOKENS: Final = 1024

STAGE_VISUAL: Final = STAGE_NAMES[0]
STAGE_HYPOTHESES: Final = STAGE_NAMES[1]
STAGE_VERIFY: Final = STAGE_NAMES[2]
STAGE_SYNTHESIZE: Final = STAGE_NAMES[3]

STEP_TITLES_BLOCK: Final = '\n'.join(
    f'{number}. {title}'
    for number, title in enumerate(STEP_TITLES[:MODEL_WRITTEN_STEPS], start=1)
)


class PipelineConfig(BaseModel):
    """Decoding and budget settings shared by every stage (ADR-0011)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    call_hard_cap: int = Field(default=CALL_BUDGET_HARD_CAP, ge=1)
    retries_per_stage: int = Field(default=RETRIES_PER_STAGE, ge=0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = Field(default=0, ge=0)


class _Placeholders(dict[str, str]):
    """A prompt's render mapping. An unfilled placeholder renders as nothing."""

    def __missing__(self, key: str) -> str:
        return ''


class HarnessPipeline:
    """Runs one frozen harness bundle against one frozen Target."""

    def __init__(
        self,
        bundle: HarnessBundle,
        target: TargetModel,
        config: PipelineConfig | None = None,
    ) -> None:
        self._bundle = bundle
        self._target = target
        self._config = config or PipelineConfig()

    def run(self, example: Example, inputs: PerceivedInputs) -> HarnessOutput:
        """Produce one item's labels, five steps, and response text."""
        parsed = self._parsed(example, inputs)
        values = _base_values(example, inputs, parsed)

        visual = self._call(STAGE_VISUAL, self._visual_request(inputs, values), ())
        traces: tuple[StageTrace, ...] = (visual.trace,)
        record = self._record(parsed, visual.payload, inputs)

        request = self._text_request(STAGE_HYPOTHESES, values, record)
        proposed = self._call(STAGE_HYPOTHESES, request, traces)
        traces += (proposed.trace,)
        hypotheses = self._hypotheses(proposed.payload, parsed)

        request = self._text_request(STAGE_VERIFY, values, record, hypotheses)
        checked = self._call(STAGE_VERIFY, request, traces)
        traces += (checked.trace,)
        verification = self._verification(checked.payload, record, hypotheses)

        ranked = hypotheses.model_copy(update={'hypotheses': verification.ranked})
        request = self._text_request(STAGE_SYNTHESIZE, values, record, ranked)
        written = self._call(STAGE_SYNTHESIZE, request, traces)
        traces += (written.trace,)

        outcomes = (visual, proposed, checked, written)
        return build_output(
            example.example_id,
            parsed,
            record,
            hypotheses,
            verification,
            self._step_texts(written),
            cap=MULTILABEL_CAP,
            traces=traces,
            fallback_used=any(outcome.payload is None for outcome in outcomes),
        )

    def _parsed(self, example: Example, inputs: PerceivedInputs) -> ParsedPrompt:
        if example.example_id != inputs.example_id:
            raise ValidationError(
                f'inputs {inputs.example_id!r} do not belong to {example.example_id!r}'
            )
        return parse_prompt(example.prompt)

    def _call(
        self, stage: str, request: TargetRequest, traces: tuple[StageTrace, ...]
    ) -> StageOutcome:
        """Run one stage inside what is left of the per-item call budget."""
        if stage not in self._bundle.routing.stages:
            return _skipped(stage)
        used = sum(trace.calls for trace in traces)
        remaining = self._config.call_hard_cap - used
        allowed = min(1 + self._config.retries_per_stage, remaining)
        if allowed <= 0:
            return _skipped(stage)
        return run_json_stage(stage, self._target, request, allowed)

    def _record(
        self,
        parsed: ParsedPrompt,
        payload: dict[str, Any] | None,
        inputs: PerceivedInputs,
    ) -> EvidenceRecord:
        visual = visual_from_payload(payload) if payload else VisualEvidence()
        return build_record(
            parsed,
            visual,
            inputs,
            self._bundle.audio_pipeline.transcript_max_chars,
        )

    def _hypotheses(
        self, payload: dict[str, Any] | None, parsed: ParsedPrompt
    ) -> HypothesisSet:
        label_set = LabelSet.from_parsed(parsed)
        proposed = hypothesis_set_from_payload(payload, label_set) if payload else None
        return proposed or uniform_hypothesis_set(label_set)

    def _verification(
        self,
        payload: dict[str, Any] | None,
        record: EvidenceRecord,
        hypotheses: HypothesisSet,
    ) -> VerificationResult:
        reranked = hypotheses
        if payload is not None:
            label_set = LabelSet.from_parsed(record.text)
            reranked = hypothesis_set_from_payload(payload, label_set) or hypotheses
        result = apply_bundle_verification(self._bundle, record, reranked)
        stated = agreement_from_payload(payload) if payload else None
        if stated is None or stated == result.modalities_agree:
            return result
        note = f'target reported modalities_agree={stated}'
        return result.model_copy(update={'notes': _joined(result.notes, note)})

    def _step_texts(self, outcome: StageOutcome) -> tuple[str, ...]:
        return step_texts_from_payload(outcome.payload) if outcome.payload else ()

    def _visual_request(
        self, inputs: PerceivedInputs, values: _Placeholders
    ) -> TargetRequest:
        return TargetRequest(
            prompt=self._render(STAGE_VISUAL, values),
            tag=STAGE_VISUAL,
            video_path=inputs.video_path,
            image_paths=self._extra_images(inputs),
            json_schema=VisualEvidence.model_json_schema(),
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            seed=self._config.seed,
        )

    def _text_request(
        self,
        stage: str,
        values: _Placeholders,
        record: EvidenceRecord,
        hypotheses: HypothesisSet | None = None,
    ) -> TargetRequest:
        extra = {'evidence_json': _as_json(record)}
        if hypotheses is not None:
            extra['hypotheses_json'] = _as_json(hypotheses)
        return TargetRequest(
            prompt=self._render(stage, _Placeholders({**values, **extra})),
            tag=stage,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            seed=self._config.seed,
        )

    def _render(self, stage: str, values: _Placeholders) -> str:
        return self._bundle.prompt(stage).format_map(values)

    def _extra_images(self, inputs: PerceivedInputs) -> tuple[str, ...]:
        sampling = self._bundle.sampling
        images: tuple[str, ...] = ()
        if sampling.face_crops.enabled:
            images += tuple(inputs.face_crop_paths[: sampling.face_crops.max_crops])
        if sampling.extra_images.enabled:
            images += tuple(inputs.frame_paths[:VIDEO_MAX_FRAMES])
        return images


def _base_values(
    example: Example, inputs: PerceivedInputs, parsed: ParsedPrompt
) -> _Placeholders:
    """The placeholder values every stage of one item shares."""
    return _Placeholders(
        prompt=example.prompt,
        quantifier=parsed.quantifier.value,
        noun=parsed.noun,
        label_set=LABEL_SEPARATOR.join(parsed.label_set),
        audio_text=inputs.audio_text(),
        step_titles=STEP_TITLES_BLOCK,
    )


def _skipped(stage: str) -> StageOutcome:
    """A stage that never ran: no calls, no tokens, no payload."""
    return StageOutcome(
        payload=None,
        trace=StageTrace(
            stage=stage,
            calls=0,
            parse_failures=0,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
        ),
    )


def _as_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode='json'), indent=2, ensure_ascii=False)


def _joined(first: str, second: str) -> str:
    return '; '.join(part for part in (first, second) if part)
