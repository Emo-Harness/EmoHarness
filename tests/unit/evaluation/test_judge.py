from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.evaluation.judge import (
    JUDGE_MULTI_LABEL_SCORE_PROMPT,
    JUDGE_SCORE_SYSTEM_PROMPT,
    JUDGE_SINGLE_LABEL_SCORE_PROMPT,
    STEP_EXTRACTION_PROMPT_TEMPLATE,
    CalibrationReport,
    FakeStepScorer,
    GeminiStepScorer,
    JudgeInput,
    OpenAIStepScorer,
    _judge_cost,
    build_score_messages,
    calibrate,
)
from emoharness.evaluation.schema import StepScores


def _tiny_jpeg(path: Path) -> Path:
    Image.new('RGB', (2, 2), color=(10, 20, 30)).save(path, format='JPEG')
    return path


def test_prompt_constants_carry_the_official_placeholders_and_examples() -> None:
    assert '{model_response}' in STEP_EXTRACTION_PROMPT_TEMPLATE
    for template in (JUDGE_SINGLE_LABEL_SCORE_PROMPT, JUDGE_MULTI_LABEL_SCORE_PROMPT):
        assert '{audio_clue}' in template
        assert '{label}' in template
        assert '{steps}' in template
    # The ML-ER prompt's own worked example (docs/benchmark-facts.md).
    assert 'the rating should be 3/3' in JUDGE_MULTI_LABEL_SCORE_PROMPT


def test_judge_input_caps_frame_paths_at_ten() -> None:
    with pytest.raises(PydanticValidationError):
        JudgeInput(
            example_id='e1',
            frame_paths=tuple(f'f{i}.jpg' for i in range(11)),
            gold='anger',
            steps_text='Step 1: The predicted emotion is anger.',
        )


def test_build_score_messages_structure_single_label(tmp_path: Path) -> None:
    frame1 = _tiny_jpeg(tmp_path / 'f1.jpg')
    frame2 = _tiny_jpeg(tmp_path / 'f2.jpg')
    inp = JudgeInput(
        example_id='e1',
        frame_paths=(str(frame1), str(frame2)),
        audio_clue='speech present, calm tone',
        gold='anger',
        steps_text='Step 1: The predicted emotion is anger.',
    )
    messages = build_score_messages(inp, multi_label=False)
    assert len(messages) == 2
    assert messages[0] == {'role': 'system', 'content': JUDGE_SCORE_SYSTEM_PROMPT}
    assert messages[1]['role'] == 'user'
    content = messages[1]['content']
    assert len(content) == 3  # 1 text part + 2 image parts
    assert content[0]['type'] == 'text'
    assert "'anger'" in content[0]['text']
    assert "'speech present, calm tone'" in content[0]['text']
    assert "'Step 1: The predicted emotion is anger.'" in content[0]['text']
    for image_part in content[1:]:
        assert image_part['type'] == 'image_url'
        assert image_part['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_build_score_messages_no_frames_is_text_only() -> None:
    inp = JudgeInput(example_id='e2', gold='anger,fear', steps_text='Step 1: x.')
    messages = build_score_messages(inp, multi_label=True)
    content = messages[1]['content']
    assert len(content) == 1
    assert 'C=the number of correct emotions' in content[0]['text']


def test_fake_step_scorer_returns_registered_and_raises_for_unknown() -> None:
    expected = StepScores(example_id='e1', scores=(1.0,))
    scorer = FakeStepScorer({'e1': expected})
    inp = JudgeInput(example_id='e1', gold='anger', steps_text='Step 1: anger.')
    assert scorer.score(inp) is expected
    with pytest.raises(ValidationError):
        scorer.score(JudgeInput(example_id='missing', gold='anger', steps_text='x'))


def test_openai_and_gemini_scorers_construct_without_network(tmp_path: Path) -> None:
    # Constructing must never touch the network or require the lazy-imported SDKs.
    ledger_path = tmp_path / 'ledger.jsonl'
    openai_scorer = OpenAIStepScorer(
        model_id='gpt-4o-2024-11-20',
        api_key_env='OPENAI_API_KEY_TEST_UNSET',
        endpoint=None,
        ledger_path=ledger_path,
        experiment='exp1',
        pricing={'input': 2.5, 'output': 10.0},
    )
    gemini_scorer = GeminiStepScorer(
        model_id='gemini-3-flash',
        api_key_env='GEMINI_API_KEY_TEST_UNSET',
        ledger_path=ledger_path,
        experiment='exp1',
        pricing={'input': 0.3, 'output': 2.5},
    )
    assert isinstance(openai_scorer, OpenAIStepScorer)
    assert isinstance(gemini_scorer, GeminiStepScorer)


def test_calibrate_partial_overlap_and_step_count_mismatch() -> None:
    proxy = (
        StepScores(example_id='a', scores=(1.0, 0.0, 1.0)),
        StepScores(example_id='b', scores=(1.0, 1.0)),
        StepScores(example_id='only_proxy', scores=(1.0,)),
    )
    reference = (
        StepScores(example_id='a', scores=(1.0, 0.0, 1.0)),
        StepScores(example_id='b', scores=(0.0, 1.0, 1.0)),
        StepScores(example_id='only_reference', scores=(0.0,)),
    )
    report = calibrate(proxy, reference)
    assert report.n == 2
    assert report.step_agreement == 1.0  # only 'a' has matching step counts (3 == 3)
    assert report.recognition_agreement == 1.0  # both items' last steps agree
    assert report.reasoning_offset == pytest.approx(0.25)
    assert report.reasoning_std == pytest.approx(0.25)


def test_judge_cost_uses_input_and_output_pricing() -> None:
    cost = _judge_cost(1_000_000, 1_000_000, {'input': 0.3, 'output': 2.5})
    assert cost == pytest.approx(0.3 + 2.5)
    assert _judge_cost(0, 0, {'input': 0.3, 'output': 2.5}) == 0.0
    assert _judge_cost(1_000_000, 0, {}) == 0.0  # missing pricing keys default to 0


def test_calibrate_empty_overlap_is_a_zero_report() -> None:
    report = calibrate(
        (StepScores(example_id='x', scores=(1.0,)),),
        (StepScores(example_id='y', scores=(1.0,)),),
    )
    assert report == CalibrationReport(
        n=0,
        step_agreement=0.0,
        recognition_agreement=0.0,
        reasoning_offset=0.0,
        reasoning_std=0.0,
    )
