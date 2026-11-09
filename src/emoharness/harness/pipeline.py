"""The four-stage inference pipeline.

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
from emoharness.harness._mechanism_screen import (
    RAW_FRAMES_CONTRACT,
    REVIEW_VIDEO_CONTRACT,
    TOP_TWO_CONTRACT,
    confidence_margin,
    decision_frames,
    source_table,
    top_two,
)
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
    """Decoding and budget settings shared by every stage."""

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
        self.mechanism_trace: dict[str, Any] = {}

    def run(self, example: Example, inputs: PerceivedInputs) -> HarnessOutput:
        """Produce one item's labels, five steps, and response text."""
        parsed = self._parsed(example, inputs)
        values = _base_values(example, inputs, parsed)
        mechanism = self._bundle.routing.mechanism_screen
        self.mechanism_trace = {'enabled': mechanism.enabled}
        if self._bundle.routing.workflow == 'direct_same_tools':
            return self._direct_same_tools(example, inputs, parsed, values)

        visual = self._call(STAGE_VISUAL, self._visual_request(inputs, values), ())
        traces: tuple[StageTrace, ...] = (visual.trace,)
        record = self._record(parsed, visual.payload, inputs)

        request = self._text_request(STAGE_HYPOTHESES, values, record)
        if mechanism.enabled and mechanism.decision_media == 'cached_frames':
            frames = decision_frames(inputs, mechanism.decision_frame_limit)
            if not frames:
                raise ValidationError(
                    'v5 decision_media requires existing cached frames'
                )
            request = request.model_copy(
                update={
                    'image_paths': frames,
                    'prompt': request.prompt + '\n\n' + RAW_FRAMES_CONTRACT,
                }
            )
            self.mechanism_trace['decision_frame_paths'] = list(frames)
        proposed = self._call(STAGE_HYPOTHESES, request, traces)
        traces += (proposed.trace,)
        hypotheses = self._hypotheses(proposed.payload, parsed)

        preliminary = apply_bundle_verification(self._bundle, record, hypotheses)
        should_verify = (
            self._bundle.routing.verify_on == 'always'
            or not preliminary.modalities_agree
            or bool(preliminary.notes)
            or proposed.payload is None
            or visual.payload is None
        )
        verify_hypotheses = hypotheses
        if mechanism.enabled:
            sanitized = hypotheses.model_copy(update={'hypotheses': preliminary.ranked})
            margin = confidence_margin(sanitized)
            invalid_hypotheses = (
                proposed.payload is None
                or hypothesis_set_from_payload(
                    proposed.payload, LabelSet.from_parsed(parsed)
                )
                is None
            )
            if mechanism.verify_trigger == 'conflict_or_low_margin':
                should_verify = (
                    not preliminary.modalities_agree
                    or margin < mechanism.margin_threshold
                    or bool(preliminary.notes)
                    or invalid_hypotheses
                    or visual.payload is None
                )
            if mechanism.verify_style == 'top2_contrast':
                verify_hypotheses = top_two(sanitized)
            self.mechanism_trace.update(
                {
                    'margin': margin,
                    'modalities_agree': preliminary.modalities_agree,
                    'verify_trigger': mechanism.verify_trigger,
                    'verify_executed': should_verify,
                    'verify_style': mechanism.verify_style,
                    'verify_input_labels': [
                        h.label for h in verify_hypotheses.hypotheses
                    ],
                    'visual_payload_missing': visual.payload is None,
                    'hypothesis_payload_missing': invalid_hypotheses,
                }
            )
        verify_record = preliminary
        if mechanism.enabled and mechanism.verify_style == 'top2_contrast':
            verify_record = preliminary.model_copy(
                update={'ranked': verify_hypotheses.hypotheses}
            )
        request = self._text_request(
            STAGE_VERIFY, values, record, verify_hypotheses, verify_record
        )
        if mechanism.enabled and mechanism.verify_style == 'top2_contrast':
            request = request.model_copy(
                update={'prompt': request.prompt + '\n\n' + TOP_TWO_CONTRACT}
            )
        if (
            mechanism.enabled
            and mechanism.review_media == 'video_on_trigger'
            and should_verify
        ):
            request = request.model_copy(
                update={
                    'video_path': inputs.video_path,
                    'prompt': request.prompt + '\n\n' + REVIEW_VIDEO_CONTRACT,
                }
            )
            self.mechanism_trace['review_video_path'] = inputs.video_path
        checked = (
            self._call(STAGE_VERIFY, request, traces)
            if should_verify
            else _skipped(STAGE_VERIFY)
        )
        traces += (checked.trace,)
        verification = (
            self._verification(checked.payload, record, hypotheses)
            if should_verify
            else preliminary
        )

        ranked = hypotheses.model_copy(update={'hypotheses': verification.ranked})
        request = self._text_request(
            STAGE_SYNTHESIZE, values, record, ranked, verification
        )
        written = self._call(STAGE_SYNTHESIZE, request, traces)
        traces += (written.trace,)

        outcomes = (visual, proposed, written) + ((checked,) if should_verify else ())
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

    def _direct_same_tools(
        self,
        example: Example,
        inputs: PerceivedInputs,
        parsed: ParsedPrompt,
        values: _Placeholders,
    ) -> HarnessOutput:
        # One real stage trace, not fake four-stage traces. The distinct request
        # tag + full contract + video/schema form a separate content-cache key.
        audio_record = self._record(parsed, None, inputs)
        request = self._text_request(STAGE_HYPOTHESES, values, audio_record)
        request = request.model_copy(
            update={
                'prompt': request.prompt + '\n\n' + _DIRECT_CONTRACT,
                'tag': 'direct_same_tools_v1',
                'video_path': inputs.video_path,
                'image_paths': self._extra_images(inputs),
            }
        )
        allowed = min(1 + self._config.retries_per_stage, self._config.call_hard_cap)
        outcome = run_json_stage(
            'direct_same_tools',
            self._target,
            request,
            allowed,
            payload_check=lambda payload: _valid_direct_payload(payload, parsed),
        )
        record = self._record(parsed, outcome.payload, inputs)
        hypotheses = self._hypotheses(outcome.payload, parsed)
        verification = apply_bundle_verification(self._bundle, record, hypotheses)
        valid_hypotheses = (
            hypothesis_set_from_payload(outcome.payload, LabelSet.from_parsed(parsed))
            if outcome.payload
            else None
        )
        return build_output(
            example.example_id,
            parsed,
            record,
            hypotheses,
            verification,
            self._step_texts(outcome),
            cap=MULTILABEL_CAP,
            traces=(outcome.trace,),
            fallback_used=valid_hypotheses is None,
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
            video_max_frames=self._video_frames(inputs),
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
        verification: VerificationResult | None = None,
    ) -> TargetRequest:
        extra = {
            'evidence_json': _as_json(record),
            'verification_json': _as_json(verification) if verification else '{}',
        }
        mechanism = self._bundle.routing.mechanism_screen
        if mechanism.enabled and mechanism.evidence_layout == 'source_table':
            extra['evidence_json'] = source_table(record)
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

    def _video_frames(self, inputs: PerceivedInputs) -> int | None:
        video = self._bundle.sampling.video
        return video.frames_for(inputs.duration_seconds) if video else None

    def _extra_images(self, inputs: PerceivedInputs) -> tuple[str, ...]:
        sampling = self._bundle.sampling
        images: tuple[str, ...] = ()
        if sampling.face_crops.enabled:
            if not inputs.face_crop_paths or sampling.face_crops.max_crops == 0:
                raise ValidationError(
                    'face crops requested but no usable crops are available'
                )
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


_DIRECT_CONTRACT = """DIRECT SAME-TOOLS WORKFLOW — output contract v1.
You also receive the original video. Complete observation, label hypotheses and
explanation in this one answer, using video plus the cached audio evidence above.
For this workflow the following JSON shape extends/supersedes any earlier output
shape. Return one object with:
face: list of visible face cues; body: list of body cues; scene: list of scene cues;
temporal_change: string or null; coarse: string;
hypotheses: list of {label, confidence, support, contradiction};
steps: exactly four grounded explanation strings.
Use only labels from the request. Existing a1..an audio IDs remain unchanged.
Your visual cues get v1..vn in face, body, scene, temporal_change order; cite those
and audio IDs in hypotheses, and [v1] or [a1] style citations in steps. t1 is only
the question and is not supporting evidence. Do not invent missing audio, extra
model calls, or tools. Give hypotheses for plausible alternatives, then distinguish
them with observed evidence. The fixed controller will emit the final answer.
"""


def _valid_direct_payload(payload: dict[str, Any], parsed: ParsedPrompt) -> bool:
    """A JSON object alone is not a successful direct-workflow answer."""
    hypotheses = hypothesis_set_from_payload(payload, LabelSet.from_parsed(parsed))
    steps = payload.get('steps')
    return (
        hypotheses is not None
        and isinstance(steps, list)
        and len(steps) == MODEL_WRITTEN_STEPS
        and all(isinstance(step, str) and bool(step.strip()) for step in steps)
    )
