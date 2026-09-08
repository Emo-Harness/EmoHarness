"""The harness's own scripted Target: what it answers, and how it chooses."""

from __future__ import annotations

import json
from typing import Final

from emoharness.common.constants import NUM_STEPS
from emoharness.harness._stages import STRICT_JSON_REMINDER
from emoharness.harness.bundle_models import STAGE_NAMES
from emoharness.harness.fake_responder import (
    example_key,
    label_chooser,
    label_set_of,
    scripted_responder,
)
from emoharness.harness.schema import HypothesisSet, VisualEvidence
from emoharness.target.base import TargetRequest

LABELS: Final[tuple[str, ...]] = ('anger', 'happiness', 'sadness', 'neutral')
OFFICIAL_PROMPT: Final = (
    'As an emotion recognition expert, identify the most prominent emotion '
    'displayed in the video. Candidate labels: anger, happiness, sadness, neutral. '
    'Please think step by step. Enclose your final answer within '
    '<answer></answer> tags.'
)
RECORD_PROMPT: Final = 'record: {"label_set": ["anger", "happiness", "sadness"]}'


def answer(prompt: str, tag: str | None = None, **fields: object) -> str:
    request = TargetRequest.model_validate({'prompt': prompt, 'tag': tag, **fields})
    return scripted_responder(label_chooser())(request)


class TestStageAnswers:
    def test_the_visual_stage_answers_with_a_visual_evidence_record(self) -> None:
        payload = json.loads(answer('describe the clip', STAGE_NAMES[0]))

        assert VisualEvidence.model_validate(payload).face

    def test_the_hypothesis_stage_answers_with_a_hypothesis_set(self) -> None:
        payload = json.loads(answer(RECORD_PROMPT, STAGE_NAMES[1]))
        proposed = HypothesisSet.model_validate(payload)

        assert {item.label for item in proposed.hypotheses} <= set(LABELS)

    def test_the_verify_stage_answers_with_a_ranking_and_a_flag(self) -> None:
        payload = json.loads(answer(RECORD_PROMPT, STAGE_NAMES[2]))

        assert isinstance(payload['modalities_agree'], bool)
        assert len(payload['hypotheses']) > 1

    def test_the_synthesis_stage_answers_with_step_texts(self) -> None:
        payload = json.loads(answer(RECORD_PROMPT, STAGE_NAMES[3]))

        assert len(payload['steps']) == NUM_STEPS - 1
        assert payload['justification']

    def test_hypothesis_payloads_cite_ids_the_record_will_have(self) -> None:
        payload = json.loads(answer(RECORD_PROMPT, STAGE_NAMES[1]))
        supports = {i for row in payload['hypotheses'] for i in row['support']}

        assert supports <= {'v1', 'v2', 'v3', 'a1'}

    def test_the_chosen_label_leads_the_ranking(self) -> None:
        request = TargetRequest(prompt=RECORD_PROMPT, tag=STAGE_NAMES[1])
        respond = scripted_responder(lambda _request: 'sadness')

        payload = json.loads(respond(request))

        assert payload['hypotheses'][0]['label'] == 'sadness'

    def test_a_failing_stage_answers_with_prose_once(self) -> None:
        respond = scripted_responder(
            label_chooser(), fail_stages=frozenset({STAGE_NAMES[1]})
        )
        request = TargetRequest(prompt=RECORD_PROMPT, tag=STAGE_NAMES[1])
        retried = request.model_copy(
            update={'prompt': f'{RECORD_PROMPT}\n\n{STRICT_JSON_REMINDER}'}
        )

        assert 'hypotheses' not in respond(request)
        assert 'hypotheses' in respond(retried)

    def test_a_script_replaces_one_stage_outright(self) -> None:
        respond = scripted_responder(
            label_chooser(), script={STAGE_NAMES[0]: 'not JSON at all'}
        )

        assert (
            respond(TargetRequest(prompt='p', tag=STAGE_NAMES[0])) == 'not JSON at all'
        )


class TestSingleCallAnswers:
    def test_an_untagged_row_gets_five_steps_and_a_candidate(self) -> None:
        text = answer(OFFICIAL_PROMPT, None, video_path='/clip.mp4')

        assert text.count('Step ') == NUM_STEPS
        assert any(f'<answer>{label}</answer>' in text for label in LABELS)

    def test_a_baseline_tag_is_answered_the_same_way(self) -> None:
        tagged = answer(OFFICIAL_PROMPT, 'direct', video_path='/clip.mp4')
        untagged = answer(OFFICIAL_PROMPT, None, video_path='/clip.mp4')

        assert tagged == untagged

    def test_different_clips_get_different_answers(self) -> None:
        respond = scripted_responder(label_chooser())
        answers = {
            respond(
                TargetRequest(prompt=OFFICIAL_PROMPT, video_path=f'/clip_{index}.mp4')
            )
            for index in range(12)
        }

        assert len(answers) > 1

    def test_the_same_clip_always_gets_the_same_answer(self) -> None:
        request = TargetRequest(prompt=OFFICIAL_PROMPT, video_path='/clip.mp4')
        respond = scripted_responder(label_chooser())

        assert respond(request) == respond(request)

    def test_two_variants_answer_the_same_request_differently(self) -> None:
        request = TargetRequest(prompt=OFFICIAL_PROMPT, video_path='/clip.mp4')

        first = scripted_responder(label_chooser('a'))(request)
        second = scripted_responder(label_chooser('b'))(request)

        assert first != second


class TestExampleKey:
    def test_the_framing_note_makes_a_prompt_edit_change_some_answers(self) -> None:
        base = 'record {"transcript": "the speaker says one at minute %d."}'
        old, new = 'body: framing note 1', 'body: framing note 4'

        changed = sum(
            example_key(TargetRequest(prompt=f'{base % index} {old}'))
            != example_key(TargetRequest(prompt=f'{base % index} {new}'))
            for index in range(64)
        )

        assert 0 < changed < 64

    def test_temperature_makes_samples_differ_by_seed(self) -> None:
        first = TargetRequest(prompt='p', video_path='/c.mp4', temperature=0.7, seed=0)
        second = first.model_copy(update={'seed': 1})

        assert example_key(first) != example_key(second)

    def test_a_variant_keys_the_same_clip_apart(self) -> None:
        request = TargetRequest(prompt='p', video_path='/c.mp4')

        assert example_key(request, 'a') != example_key(request, 'b')


class TestLabelChoice:
    def test_the_choice_is_stable_and_within_the_candidate_list(self) -> None:
        choose = label_chooser()
        request = TargetRequest(prompt=OFFICIAL_PROMPT, video_path='/clip.mp4')

        assert choose(request) in LABELS
        assert choose(request) == choose(request)

    def test_the_label_set_comes_from_the_official_prompt_when_there_is_one(
        self,
    ) -> None:
        assert label_set_of(TargetRequest(prompt=OFFICIAL_PROMPT)) == LABELS

    def test_the_label_set_falls_back_to_the_evidence_record(self) -> None:
        request = TargetRequest(prompt='x "label_set": ["calm", "tense"] y')

        assert label_set_of(request) == ('calm', 'tense')

    def test_the_label_set_has_a_last_resort(self) -> None:
        assert len(label_set_of(TargetRequest(prompt='nothing here'))) >= 2
