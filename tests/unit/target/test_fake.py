"""The Target doubles: what they answer, what they record, and what they cost."""

from __future__ import annotations

from pathlib import Path

from emoharness.common.ledger import read_entries
from emoharness.target.base import TargetRequest
from emoharness.target.fake import (
    FAKE_MODEL_ID,
    FakeTarget,
    RecordingTarget,
    scripted_target,
)


class TestFakeTarget:
    def test_a_default_target_answers_with_its_default_text(self) -> None:
        target = FakeTarget(default_text='<answer>joy</answer>')

        assert target.generate(TargetRequest(prompt='p')).text == '<answer>joy</answer>'

    def test_a_responder_sees_the_whole_request(self) -> None:
        target = FakeTarget(responder=lambda request: request.tag or 'no tag')

        assert target.generate(TargetRequest(prompt='p', tag='verify')).text == 'verify'

    def test_every_request_is_recorded_in_order(self) -> None:
        target = FakeTarget()

        target.generate(TargetRequest(prompt='one'))
        target.generate(TargetRequest(prompt='two'))

        assert [request.prompt for request in target.requests] == ['one', 'two']
        assert target.call_count == 2

    def test_token_counts_follow_the_text_length(self) -> None:
        target = FakeTarget(default_text='x' * 40)

        response = target.generate(TargetRequest(prompt='p' * 80))

        assert response.input_tokens == 20
        assert response.output_tokens == 10
        assert response.model_id == FAKE_MODEL_ID


class TestRecordingTarget:
    def test_one_ledger_entry_per_call(self, tmp_path: Path) -> None:
        ledger = tmp_path / 'ledger.jsonl'
        target = RecordingTarget(FakeTarget(), ledger, 'test')

        target.generate(TargetRequest(prompt='identify'))
        target.generate(TargetRequest(prompt='identify again'))

        entries = read_entries(ledger)
        assert len(entries) == 2
        assert all(entry.role == 'target' for entry in entries)
        assert all(entry.cost_usd > 0 for entry in entries)

    def test_no_ledger_path_records_nothing(self, tmp_path: Path) -> None:
        target = RecordingTarget(FakeTarget(), None, 'test')

        assert target.generate(TargetRequest(prompt='p')).text
        assert not list(tmp_path.iterdir())

    def test_the_model_id_is_the_wrapped_target_id(self) -> None:
        target = RecordingTarget(FakeTarget(model_id='inner'), None, 'test')

        assert target.model_id == 'inner'


class TestScriptedTarget:
    def test_the_target_answers_with_its_responder(self) -> None:
        target = scripted_target(lambda request: f'saw {request.prompt}')

        assert target.generate(TargetRequest(prompt='a clip')).text == 'saw a clip'

    def test_the_target_reports_the_model_id_it_was_given(self) -> None:
        assert scripted_target(lambda _r: 'x', None, 'test', 'my-model').model_id == (
            'my-model'
        )

    def test_the_target_records_one_ledger_entry_per_call(self, tmp_path: Path) -> None:
        ledger = tmp_path / 'ledger.jsonl'
        target = scripted_target(lambda _r: 'x', ledger, 'test')

        target.generate(TargetRequest(prompt='identify', video_path='/c.mp4'))

        assert ledger.read_text(encoding='utf-8').count('\n') == 1
