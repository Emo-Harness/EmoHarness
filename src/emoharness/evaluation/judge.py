"""Proxy-judge prompts and scorer implementations, calibrated against the official judge.

The prompt constants below mirror the official `extract_step.py` and
`eval_cot_gpt4o.py` scripts (QwenAudio/MME-Emotion, MIT) verbatim in wording,
numbers, and examples. The scripts build these prompts as an f-string nested
several indentation levels deep, which leaves stray runs of leading whitespace
at every line break in the text actually sent to the API; that incidental
whitespace (not meaningful content) is normalized here to single spaces and
clean line breaks at the paragraph boundaries the source already implies.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import JUDGE_MAX_FRAMES
from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.common.ledger import LedgerEntry, record
from emoharness.evaluation.official import parse_score_string
from emoharness.evaluation.schema import StepScores

#: How a judge prompt spells a multi-label gold answer.
GOLD_SEPARATOR: Final = ','

JUDGE_SCORE_SYSTEM_PROMPT: Final = (
    'You are an expert in affective computing and very good at handling tasks '
    'related to emotion recognition.'
)

# Mirrors extract_step.py (GPT-4.1 step extraction). Not called by any StepScorer
# below (JudgeInput carries already-extracted steps_text); kept here, verbatim,
# for a future step-extraction proxy call and for provenance alongside the
# scoring prompts it feeds.
STEP_EXTRACTION_PROMPT_TEMPLATE: Final = """Given the following answer about the emotion conveyed in a video, please help me extract several key steps from the answer. Each step should be as concise as possible and prefixed with Step X. In the last step, showcase the predicted emotion. If there are no reasoning steps, please showcase the predicted emotion at Step 1. Enclose your result within <step></step> tags.
answer: '{model_response}'
Example 1:
answer: In the video, an elderly man wearing a green shirt is in an outdoor nighttime setting. He appears focused and serious, with slightly furrowed brows and a serious expression. His eyes are scanning the other person, suggesting he wants to convey something important or ask about a certain thing. As time passes, his gaze shifts from scanning to direct engagement, eventually relaxing and showing a hint of happiness, indicating fluctuations in his emotions. Overall, the elderly man experiences a moderate intensity of neutral emotion, mixed with slight positive fluctuations.
result: <step>Step 1: Observe the setting and characters in the video, noting any relevant details such as location and participants' appearance. Step 2: Analyze the facial expressions and movements, such as furrowed brows and downturned eyes, looking for indicators of specific emotions. Step 3: Consider verbal and non-verbal cues, including mouth movements and gaze direction, that signify communication and emotional responses. Step 4: Synthesize observations to understand the overall emotional state conveyed by the participant. Step 5: Based on the analysis, determine that the predicted emotion is neutral.</step>
Example 2:
answer: happy
result: <step>Step 1: The predicted emotion is happy.</step>"""

# Mirrors eval_cot_gpt4o.py for a single-label task (e.g. ER-Wild, ER-Lab, SA, IR,
# Noise-ER); those scripts differ from each other only in the noun ("emotion" /
# "sentiment" / "intent"), which this constant does not parameterize.
JUDGE_SINGLE_LABEL_SCORE_PROMPT: Final = """I will first give you some ground truth information about the emotion in a video: visual clue, audio clue, and emotion label. I will also give you a model prediction. Please help me rate the performance of the prediction.

Rating Requirements:
1. Rate each step using 0 or 1 (0=wrong, 1=correct)
2. For the last step, your rating should be 1 if the predicted emotion matches the ground truth emotion label, otherwise 0. Don't consider visual/audio cues at this step
3. For each other step, only consider predictions clearly contradicting visual/audio cues as incorrect
4. Ensure the number of steps in your rating is equal to that in the model prediction
5. Output format: <score>Step 1: 0/1, Step 2: 1/1,...</score>

Input Data:
- Visual clue: The video frames (images)
- Audio clue: '{audio_clue}'
- Emotion label: '{label}'
- Model prediction: '{steps}'

Example Output:
<score>Step 1: 1/1, Step 2: 0/1, Step 3: 0/1</score>"""

# Mirrors eval_cot_gpt4o.py for ML-ER (last step scored C/K). FG-ER's own script
# carries a differently-worded C/K prompt with its own examples; this constant
# uses the ML-ER wording, which docs/benchmark-facts.md also quotes.
JUDGE_MULTI_LABEL_SCORE_PROMPT: Final = """I will first give you some ground truth information about the emotions in a video: visual clue, audio clue, and emotion labels. I will also give you a model prediction. Please help me rate the performance of the prediction.

Rating Requirements:
1. Rate each step (except the last step) using 0/1 or 1/1 (0/1=wrong, 1/1=correct)
2. Rate the last step using C/K (C=the number of correct emotions in the prediction, K=the number of emotions in the emotion labels). The denominator of the last step should be equal to the number of emotions in the emotion labels. Don't consider visual/audio cues at this step. For example: if predicted emotions are "<step>Step N: The predicted emotion is neutral.</step>" and emotion labels are "anger,fear", then the rating should be 0/2. If predicted emotions are "<step>Step N: The predicted emotion is happy.</step>" and emotion labels are "happy,anger,disgust", then the rating should be 1/3. If predicted emotions are "<step>Step N: The predicted emotions are happy,disgust,sad.</step>" and emotion labels are "sad,anger,surprise,disgust", then the rating should be 2/4. If predicted emotions are "<step>Step N: The predicted emotions are disappointment, fear, contempt, disgust, sadness, anger, anxiety, and helplessness.</step>" and emotion labels are "fear,sadness,anxiety", then the rating should be 3/3.
3. For each other step (except the last step, if any), only consider predictions clearly contradicting visual/audio cues as incorrect
4. Ensure the number of steps in your rating is equal to that in the model prediction
5. Output format: <score>Step 1: 0/1, Step 2: 1/1,..., Step N: 1/4</score>

Input Data:
- Visual clue: The video frames (images)
- Audio clue: '{audio_clue}'
- Emotion labels: '{label}'
- Model prediction: '{steps}'

Example Output:
<score>Step 1: 1/1, Step 2: 0/1, Step 3: 2/3</score> or <score>Step 1: 0/2 </score>"""


class JudgeInput(BaseModel):
    """Everything one judge scoring call needs for one item."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str = Field(min_length=1)
    frame_paths: tuple[str, ...] = Field(default=(), max_length=JUDGE_MAX_FRAMES)
    audio_clue: str = ''
    gold: str = Field(min_length=1)
    steps_text: str = Field(min_length=1)


def judge_input_for(
    example_id: str,
    frames: Sequence[str],
    gold: Sequence[str],
    steps_text: str,
) -> JudgeInput:
    """One item's judge call: the first ``JUDGE_MAX_FRAMES`` frames, no audio clue.

    The clue is empty on purpose: only the official-judge wrapper may read the
    benchmark's judge-side clues, so every proxy call scores the steps without them.
    """
    return JudgeInput(
        example_id=example_id,
        frame_paths=tuple(frames[:JUDGE_MAX_FRAMES]),
        audio_clue='',
        gold=GOLD_SEPARATOR.join(gold),
        steps_text=steps_text,
    )


@runtime_checkable
class StepScorer(Protocol):
    """Anything that rates one item's extracted steps against its gold label(s)."""

    def score(self, inp: JudgeInput) -> StepScores: ...


class FakeStepScorer:
    """Deterministic scorer for tests: returns pre-registered scores by example_id."""

    def __init__(self, scores_by_id: Mapping[str, StepScores]) -> None:
        self._scores_by_id = dict(scores_by_id)

    def score(self, inp: JudgeInput) -> StepScores:
        try:
            return self._scores_by_id[inp.example_id]
        except KeyError as exc:
            raise ValidationError(
                f'no fake score registered for {inp.example_id}'
            ) from exc


def _render_score_prompt(inp: JudgeInput, multi_label: bool) -> str:
    template = (
        JUDGE_MULTI_LABEL_SCORE_PROMPT
        if multi_label
        else JUDGE_SINGLE_LABEL_SCORE_PROMPT
    )
    return template.format(
        audio_clue=inp.audio_clue, label=inp.gold, steps=inp.steps_text
    )


def _frame_data_uri(path: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


def build_score_messages(inp: JudgeInput, multi_label: bool) -> list[dict[str, Any]]:
    """OpenAI-style chat messages for one scoring call, mirroring eval_cot_gpt4o.py."""
    content: list[dict[str, Any]] = [
        {'type': 'text', 'text': _render_score_prompt(inp, multi_label)}
    ]
    for path in inp.frame_paths:
        content.append(
            {'type': 'image_url', 'image_url': {'url': _frame_data_uri(path)}}
        )
    return [
        {'role': 'system', 'content': JUDGE_SCORE_SYSTEM_PROMPT},
        {'role': 'user', 'content': content},
    ]


def _judge_cost(
    input_tokens: int, output_tokens: int, pricing: Mapping[str, float]
) -> float:
    input_cost = input_tokens * pricing.get('input', 0.0)
    output_cost = output_tokens * pricing.get('output', 0.0)
    return (input_cost + output_cost) / 1_000_000


class OpenAIStepScorer:
    """Scores one item through an OpenAI-compatible chat endpoint; logs a `LedgerEntry`.

    Used for the proxy judge (ADR-0007) or for ad hoc GPT-4o scoring outside the
    official scripts. `multi_label` picks the prompt template for every call
    made by this instance; the ADR-0007 gate only ever scores D_val, which is
    single-label, so its scorer instances never need to set it.
    `openai` is imported lazily so importing this module never requires it, and
    the real network call is never exercised by tests (engineering.md).
    """

    def __init__(
        self,
        model_id: str,
        api_key_env: str,
        endpoint: str | None,
        ledger_path: Path,
        experiment: str,
        pricing: Mapping[str, float],
        multi_label: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._endpoint = endpoint
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._pricing = pricing
        self._multi_label = multi_label
        self._reasoning_effort = reasoning_effort

    def generation_params(self) -> dict[str, Any]:
        """Sampling fields for this scorer's model family.

        The GPT-5 family takes ``reasoning_effort`` and rejects a temperature at
        any effort other than ``none``; every other model gets temperature 0.
        """
        effort = self._reasoning_effort
        params: dict[str, Any] = {}
        if effort is not None:
            params['reasoning_effort'] = effort
        if effort is None or effort == 'none':
            params['temperature'] = 0.0
        return params

    def score(self, inp: JudgeInput) -> StepScores:  # pragma: no cover
        from openai import OpenAI  # pragma: no cover

        api_key = os.environ.get(self._api_key_env, '')
        if not api_key:
            raise EmoHarnessError(
                f'environment variable {self._api_key_env} is not set'
            )
        client = OpenAI(api_key=api_key, base_url=self._endpoint)
        messages: Any = build_score_messages(inp, self._multi_label)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=self._model_id, messages=messages, **self.generation_params()
        )
        latency = time.perf_counter() - started
        text = response.choices[0].message.content or ''
        usage = response.usage
        input_tokens = int(usage.prompt_tokens) if usage else 0
        output_tokens = int(usage.completion_tokens) if usage else 0
        scores = parse_score_string(text)
        if scores is None:
            raise ValidationError(
                f'judge response for {inp.example_id} has no <score> block'
            )
        record(
            self._ledger_path,
            LedgerEntry(
                role='judge_proxy',
                model_id=self._model_id,
                experiment=self._experiment,
                example_id=inp.example_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
                cost_usd=_judge_cost(input_tokens, output_tokens, self._pricing),
            ),
        )
        return StepScores(example_id=inp.example_id, scores=scores)


class GeminiStepScorer:
    """Scores one item through Gemini (google-genai); mirrors `OpenAIStepScorer`'s contract.

    `google.genai` is imported lazily; the real network call is never exercised
    by tests (engineering.md).
    """

    def __init__(
        self,
        model_id: str,
        api_key_env: str,
        ledger_path: Path,
        experiment: str,
        pricing: Mapping[str, float],
        multi_label: bool = False,
    ) -> None:
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._ledger_path = ledger_path
        self._experiment = experiment
        self._pricing = pricing
        self._multi_label = multi_label

    def score(self, inp: JudgeInput) -> StepScores:  # pragma: no cover
        from google import genai  # pragma: no cover
        from google.genai import types  # pragma: no cover

        api_key = os.environ.get(self._api_key_env, '')
        if not api_key:
            raise EmoHarnessError(
                f'environment variable {self._api_key_env} is not set'
            )
        client = genai.Client(api_key=api_key)
        parts: list[Any] = [
            types.Part.from_text(text=_render_score_prompt(inp, self._multi_label))
        ]
        for path in inp.frame_paths:
            data = Path(path).read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type='image/jpeg'))
        started = time.perf_counter()
        response = client.models.generate_content(model=self._model_id, contents=parts)
        latency = time.perf_counter() - started
        text = str(response.text or '')
        usage = getattr(response, 'usage_metadata', None)
        input_tokens = int(getattr(usage, 'prompt_token_count', 0) or 0)
        output_tokens = int(getattr(usage, 'candidates_token_count', 0) or 0)
        scores = parse_score_string(text)
        if scores is None:
            raise ValidationError(
                f'judge response for {inp.example_id} has no <score> block'
            )
        record(
            self._ledger_path,
            LedgerEntry(
                role='judge_proxy',
                model_id=self._model_id,
                experiment=self._experiment,
                example_id=inp.example_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
                cost_usd=_judge_cost(input_tokens, output_tokens, self._pricing),
            ),
        )
        return StepScores(example_id=inp.example_id, scores=scores)


class CalibrationReport(BaseModel):
    """Proxy-vs-reference judge agreement summary (ADR-0007 §3)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    n: int = Field(ge=0)
    step_agreement: float = Field(ge=0.0, le=1.0)
    recognition_agreement: float = Field(ge=0.0, le=1.0)
    reasoning_offset: float
    reasoning_std: float = Field(ge=0.0)


def calibrate(
    proxy: Iterable[StepScores], reference: Iterable[StepScores]
) -> CalibrationReport:
    """Compare a proxy judge against the reference judge, aligned by `example_id`.

    Step agreement (the fraction of matching per-step 0/1 ratings) is computed
    only over items where both judges emitted the same number of steps; a
    mismatched step count still contributes to recognition agreement and to the
    Rea-S offset. `n` is the number of items present in both inputs.
    """
    proxy_by_id = {s.example_id: s for s in proxy}
    reference_by_id = {s.example_id: s for s in reference}
    shared_ids = tuple(i for i in proxy_by_id if i in reference_by_id)
    if not shared_ids:
        return CalibrationReport(
            n=0,
            step_agreement=0.0,
            recognition_agreement=0.0,
            reasoning_offset=0.0,
            reasoning_std=0.0,
        )

    step_matches, step_total, recognition_matches = 0, 0, 0
    reasoning_diffs: list[float] = []
    for item_id in shared_ids:
        proxy_score, reference_score = proxy_by_id[item_id], reference_by_id[item_id]
        if proxy_score.num_steps == reference_score.num_steps:
            step_matches += sum(
                1
                for x, y in zip(proxy_score.scores, reference_score.scores, strict=True)
                if x == y
            )
            step_total += proxy_score.num_steps
        if proxy_score.recognition == reference_score.recognition:
            recognition_matches += 1
        reasoning_diffs.append(proxy_score.reasoning - reference_score.reasoning)

    return CalibrationReport(
        n=len(shared_ids),
        step_agreement=(step_matches / step_total) if step_total else 0.0,
        recognition_agreement=recognition_matches / len(shared_ids),
        reasoning_offset=float(np.mean(reasoning_diffs)),
        reasoning_std=float(np.std(reasoning_diffs)),
    )
