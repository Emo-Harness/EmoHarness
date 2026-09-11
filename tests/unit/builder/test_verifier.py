"""The per-round verifier: what it is shown, what it answers, and how it bills."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from emoharness.builder.config import BuilderConfig, BuilderPricing, VerifierConfig
from emoharness.builder.schema import BuilderProposal, CandidatePatch
from emoharness.builder.verifier import (
    VERIFIER_ROLE,
    FakeVerifier,
    OpenAIVerifier,
    ReviewRequest,
    Verdict,
    VerificationReport,
    Verifier,
    parse_report,
    render_review_text,
    review_request,
    verifier_for,
)
from emoharness.common.errors import EmoHarnessError, ValidationError
from emoharness.common.ledger import read_entries

FILES = {
    'prompts/extract_visual.md': 'Describe only what can be seen.\n',
    'prompts/verify.md': 'Re-examine the earlier reading.\nKeep every label.\n',
    'routing.yaml': 'stages: [extract_visual, verify]\n',
}


def _request(*candidates: CandidatePatch) -> ReviewRequest:
    return review_request(
        stage=2,
        round_index=1,
        bundle_files=FILES,
        editable_paths=('prompts/extract_visual.md', 'prompts/verify.md'),
        clusters=(('happy | neutral | agree', 12), ('anger | sadness | agree', 3)),
        proposal=BuilderProposal(candidates=candidates),
    )


def _report(*rows: tuple[str, bool, tuple[str, ...]]) -> dict[str, Any]:
    return {
        'summary': 'reviewed',
        'verdicts': [
            {'name': name, 'approved': approved, 'issues': list(issues), 'notes': ''}
            for name, approved, issues in rows
        ],
    }


class TestReviewText:
    def test_shows_rules_files_clusters_and_a_diff_per_candidate(
        self, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        patch = make_patch(
            name='verify_more',
            path='prompts/verify.md',
            content='Re-examine the earlier reading.\nKeep every label.\nWeigh the voice.\n',
        )
        text = render_review_text(_request(patch))

        assert 'Reject a candidate when' in text
        assert '- prompts/verify.md' in text
        assert 'happy | neutral | agree: 12 items' in text
        assert '### prompts/routing.yaml' not in text
        assert '### routing.yaml' in text
        assert '### Candidate 1: verify_more' in text
        assert '+Weigh the voice.' in text
        assert '--- prompts/verify.md (current)' in text

    def test_an_unchanged_file_says_so(
        self, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        patch = make_patch(path='prompts/verify.md', content=FILES['prompts/verify.md'])
        assert '(no textual change)' in render_review_text(_request(patch))

    def test_the_request_records_what_the_proposer_was_told(
        self, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        request = _request(make_patch(name='a'), make_patch(name='b'))
        assert [c.name for c in request.candidates] == ['a', 'b']
        assert request.clusters[1] == 'anger | sadness | agree: 3 items'
        assert request.editable_paths == (
            'prompts/extract_visual.md',
            'prompts/verify.md',
        )


class TestParseReport:
    def test_accepts_plain_json_and_fences(self) -> None:
        payload = _report(('a', True, ()), ('b', False, ('2. label forcing',)))
        plain = parse_report(json.dumps(payload))
        fenced = parse_report('Here:\n```json\n' + json.dumps(payload) + '\n```')

        assert plain == fenced
        assert plain.verdict_for('a').approved is True
        assert plain.verdict_for('b').issues == ('2. label forcing',)

    def test_names_match_across_case_and_punctuation(self) -> None:
        report = parse_report(json.dumps(_report(('promote-smiles', True, ()))))
        assert report.verdict_for('Promote smiles').approved is True
        assert report.verdict_for('Promote smiles 2').approved is False

    def test_a_skipped_candidate_is_not_approved(self) -> None:
        report = parse_report(json.dumps(_report(('a', True, ()))))
        verdict = report.verdict_for('missing')
        assert verdict.approved is False
        assert 'no verdict' in verdict.issues[0]

    def test_bent_fields_are_coerced(self) -> None:
        text = json.dumps(
            {
                'verdicts': [
                    {'name': 'a', 'approved': False, 'issues': '1. overfit'},
                    {'name': 'b', 'approved': True, 'notes': None},
                ]
            }
        )
        report = parse_report(text)
        assert report.verdict_for('a').issues == ('1. overfit',)
        assert report.verdict_for('b').notes == ''
        assert report.summary == ''

    @pytest.mark.parametrize(
        ('text', 'match'),
        [
            ('no json', 'no JSON object'),
            ('{"verdicts": [{"name": ""}]}', 'does not match the schema'),
        ],
    )
    def test_rejects_bad_responses(self, text: str, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            parse_report(text)


class TestFakeVerifier:
    def test_approves_everything_by_default_and_records_the_request(
        self, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        verifier = FakeVerifier()
        request = _request(make_patch(name='a'), make_patch(name='b'))

        report = verifier.review(request)

        assert isinstance(verifier, Verifier)
        assert [v.approved for v in report.verdicts] == [True, True]
        assert verifier.requests == (request,)

    def test_answers_from_its_script_first(
        self, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        scripted = VerificationReport(
            verdicts=(Verdict(name='a', approved=False, issues=('1. overfit',)),)
        )
        verifier = FakeVerifier([scripted])

        first = verifier.review(_request(make_patch(name='a')))
        second = verifier.review(_request(make_patch(name='a')))

        assert first is scripted
        assert second.verdict_for('a').approved is True


def _config(**overrides: Any) -> BuilderConfig:
    return BuilderConfig(
        model_id='fake-gpt',
        provider='openai',
        api_key_env='EMOHARNESS_TEST_OPENAI_KEY',
        endpoint='https://api.example.com/v1',
        reasoning_effort='high',
        pricing=BuilderPricing(input_usd_per_million=2.0, output_usd_per_million=12.0),
        **overrides,
    )


@pytest.mark.usefixtures('openai_key')
class TestOpenAIVerifier:
    def test_posts_the_review_bills_as_verifier_and_uses_its_own_effort(
        self, tmp_path: Path, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = {
                'choices': [
                    {'message': {'content': json.dumps(_report(('a', True, ())))}}
                ],
                'usage': {'prompt_tokens': 1_000_000, 'completion_tokens': 500_000},
            }
            return httpx.Response(200, json=body)

        ledger = tmp_path / 'ledger.jsonl'
        verifier = OpenAIVerifier(
            _config(verifier=VerifierConfig(reasoning_effort='low')),
            ledger,
            'stage2-r0',
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        report = verifier.review(_request(make_patch(name='a')))

        assert isinstance(verifier, Verifier)
        assert verifier.model_id == 'fake-gpt'
        assert report.verdict_for('a').approved is True
        sent = json.loads(requests[0].content)
        assert sent['reasoning_effort'] == 'low'
        assert 'temperature' not in sent
        assert 'Reject a candidate when' in sent['messages'][0]['content'][0]['text']
        entries = read_entries(ledger)
        assert [e.role for e in entries] == [VERIFIER_ROLE]
        assert entries[0].cost_usd == pytest.approx(2.0 + 6.0)

    def test_inherits_the_builder_effort_when_unset(
        self, tmp_path: Path, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = json.dumps(_report(('a', True, ())))
            return httpx.Response(
                200, json={'choices': [{'message': {'content': content}}]}
            )

        verifier = OpenAIVerifier(
            _config(),
            tmp_path / 'l.jsonl',
            'exp',
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        verifier.review(_request(make_patch(name='a')))
        assert json.loads(requests[0].content)['reasoning_effort'] == 'high'

    def test_failures_name_the_verifier(
        self, tmp_path: Path, make_patch: Callable[..., CandidatePatch]
    ) -> None:
        verifier = OpenAIVerifier(
            _config(),
            tmp_path / 'l.jsonl',
            'exp',
            client=httpx.Client(
                transport=httpx.MockTransport(lambda _r: httpx.Response(500))
            ),
            max_retries=0,
        )
        with pytest.raises(EmoHarnessError, match='verifier call failed after'):
            verifier.review(_request(make_patch(name='a')))


def test_verifier_for_honours_the_switch(tmp_path: Path) -> None:
    assert verifier_for(_config(), tmp_path / 'l', 'e') is not None
    off = _config(verifier=VerifierConfig(enabled=False))
    assert verifier_for(off, tmp_path / 'l', 'e') is None
