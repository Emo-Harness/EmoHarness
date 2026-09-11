from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from emoharness.builder.client import FakeBuilder
from emoharness.builder.clusters import FailureRecord
from emoharness.builder.config import BuilderConfig, PacketLimits
from emoharness.builder.gates import GateConfig
from emoharness.builder.loop import (
    Diagnose,
    Evaluate,
    LoopResult,
    RoundCost,
    RoundRecord,
    run_stage,
    safe_name,
    write_loop_result,
)
from emoharness.builder.packet import DiagnosisPacket, build_packet
from emoharness.builder.patch import read_bundle_files
from emoharness.builder.schema import BuilderProposal, CandidatePatch, CandidateScores
from emoharness.builder.verifier import FakeVerifier, Verdict, VerificationReport
from emoharness.common.errors import LeakageError, ValidationError
from emoharness.common.ledger import LedgerEntry, record

Proposal = Callable[..., BuilderProposal]
PROMPT = 'prompts/extract_visual.md'
GATES = GateConfig(max_p95_latency_s=10.0)


def _config(rounds: int = 3, early_stop: int = 2, candidates: int = 4) -> BuilderConfig:
    return BuilderConfig(
        max_rounds_per_stage=rounds,
        early_stop_rounds=early_stop,
        max_candidates_per_round=candidates,
        packet=PacketLimits(max_keyframe_samples=0),
    )


def _diagnose(
    stage: int = 2, example_id: str = 'ext_0001', keyframes: tuple[str, ...] = ()
) -> Diagnose:
    def diagnose(bundle: Path, round_index: int) -> DiagnosisPacket:
        record = FailureRecord(
            example_id=example_id,
            gold=('anger',),
            predicted=('sadness',),
            evidence_json='{"cue":"lowered gaze"}',
            keyframe_paths=keyframes,
        )
        return build_packet(
            stage=stage,
            round_index=round_index,
            bundle_files=read_bundle_files(bundle),
            build_recognition=0.4,
            records=(record,),
            limits=PacketLimits(max_keyframe_samples=len(keyframes)),
        )

    return diagnose


def _evaluate(recognition_by_name: Mapping[str, float]) -> Evaluate:
    def evaluate(bundle: Path) -> CandidateScores:
        return CandidateScores(
            name=bundle.name,
            recognition=recognition_by_name.get(bundle.name, 0.5),
            reasoning=0.5,
            regression_rate=0.0,
            mean_calls=3.0,
            p95_latency_s=1.0,
        )

    return evaluate


def _patch(name: str, path: str = PROMPT) -> CandidatePatch:
    return CandidatePatch(
        name=name,
        targeted_cluster='anger | sadness | agree',
        rationale='the audible cues are ignored',
        files=({'path': path, 'content': f'Prompt revision {name}.\n'},),
    )


def _proposal(*patches: CandidatePatch) -> BuilderProposal:
    return BuilderProposal(candidates=patches, summary='round summary')


def test_loop_accepts_then_stops_early(bundle_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / 'construction'
    builder = FakeBuilder(
        [
            _proposal(
                _patch('good'), _patch('off_limits', 'verification/cross_modal.py')
            ),
            _proposal(_patch('weak')),
            _proposal(_patch('weaker')),
        ]
    )
    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=builder,
        cfg=_config(),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'good': 0.7, 'weak': 0.3, 'weaker': 0.3}),
    )

    assert result.stopped_reason == 'early_stop'
    assert [r.accepted for r in result.rounds] == ['good', None, None]
    assert result.rounds[0].candidates[0].recognition == 0.7
    assert result.final_bundle == out_dir / 'final'
    assert (out_dir / 'final' / PROMPT).read_text(encoding='utf-8') == (
        'Prompt revision good.\n'
    )
    assert (out_dir / 'final' / 'routing.yaml').exists()


def test_loop_writes_provenance_for_every_round(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    builder = FakeBuilder(
        [
            _proposal(
                _patch('good'), _patch('off_limits', 'verification/cross_modal.py')
            ),
            _proposal(_patch('weak')),
            _proposal(_patch('weaker')),
        ]
    )
    result = run_stage(
        stage=2,
        replicate=1,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=builder,
        cfg=_config(),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'good': 0.7, 'weak': 0.3, 'weaker': 0.3}),
    )

    for index in range(len(result.rounds)):
        round_dir = out_dir / f'round_{index:02d}'
        for name in ('packet.txt', 'packet.json', 'proposal.json'):
            assert (round_dir / name).exists()
    first = out_dir / 'round_00'
    assert (first / 'packet.txt').read_text(encoding='utf-8') == builder.calls[
        0
    ].packet_text
    candidates = json.loads((first / 'candidates.json').read_text(encoding='utf-8'))
    assert [row['name'] for row in candidates['evaluated']] == ['good']
    assert candidates['rejected'][0]['name'] == 'off_limits'
    assert 'not editable at stage 2' in candidates['rejected'][0]['violations'][0]
    accepted = json.loads((first / 'accepted.json').read_text(encoding='utf-8'))
    assert accepted['accepted'] == 'good'
    assert accepted['scores']['recognition'] == 0.7
    second = json.loads(
        (out_dir / 'round_01' / 'accepted.json').read_text(encoding='utf-8')
    )
    assert second == {'accepted': None}
    written = json.loads((out_dir / 'loop_result.json').read_text(encoding='utf-8'))
    assert written['stage'] == 2
    assert written['replicate'] == 1
    assert written['stopped_reason'] == 'early_stop'


def test_accepted_patches_accumulate_until_the_round_budget_runs_out(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    builder = FakeBuilder(
        [_proposal(_patch('first')), _proposal(_patch('second', 'routing.yaml'))]
    )
    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=builder,
        cfg=_config(rounds=2),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'first': 0.6, 'second': 0.8}),
    )

    assert result.stopped_reason == 'max_rounds'
    assert [r.accepted for r in result.rounds] == ['first', 'second']
    assert result.rounds[0].bundle_hash != result.rounds[1].bundle_hash
    final = out_dir / 'final'
    assert (final / PROMPT).read_text(encoding='utf-8') == 'Prompt revision first.\n'
    assert (final / 'routing.yaml').read_text(encoding='utf-8') == (
        'Prompt revision second.\n'
    )


def test_the_firewall_stops_the_loop_before_the_builder_call(
    bundle_root: Path, tmp_path: Path
) -> None:
    builder = FakeBuilder([_proposal(_patch('good'))])
    with pytest.raises(LeakageError, match='ext_0001'):
        run_stage(
            stage=2,
            replicate=0,
            bundle_root=bundle_root,
            out_dir=tmp_path / 'construction',
            builder=builder,
            cfg=_config(),
            gate_cfg=GATES,
            diagnose=_diagnose(),
            evaluate=_evaluate({}),
            forbidden_ids=('ext_0001',),
        )
    assert builder.calls == ()


def test_extra_and_duplicate_candidates_are_rejected(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    builder = FakeBuilder(
        [_proposal(_patch('good'), _patch('good', 'routing.yaml'), _patch('third'))]
    )
    run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=builder,
        cfg=_config(rounds=1, candidates=2),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'good': 0.7}),
    )
    rejected = json.loads(
        (out_dir / 'round_00' / 'candidates.json').read_text(encoding='utf-8')
    )['rejected']
    assert [row['name'] for row in rejected] == ['good', 'third']
    assert 'duplicate candidate name' in rejected[0]['violations'][0]
    assert 'exceeds max_candidates_per_round' in rejected[1]['violations'][0]


def test_the_code_checker_reaches_verification_patches(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    builder = FakeBuilder([_proposal(_patch('risky', 'verification/cross_modal.py'))])

    def checker(source: str, path: str) -> tuple[str, ...]:
        return (f'{path}: file access is not allowed',) if 'Prompt' in source else ()

    result = run_stage(
        stage=3,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=builder,
        cfg=_config(rounds=1),
        gate_cfg=GATES,
        diagnose=_diagnose(stage=3),
        evaluate=_evaluate({}),
        code_checker=checker,
    )
    assert result.rounds[0].accepted is None
    rejected = json.loads(
        (out_dir / 'round_00' / 'candidates.json').read_text(encoding='utf-8')
    )['rejected']
    assert 'file access is not allowed' in rejected[0]['violations'][0]


def test_run_stage_refuses_to_overwrite_a_final_bundle(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    (out_dir / 'final').mkdir(parents=True)
    with pytest.raises(ValidationError, match='already exists'):
        run_stage(
            stage=2,
            replicate=0,
            bundle_root=bundle_root,
            out_dir=out_dir,
            builder=FakeBuilder([]),
            cfg=_config(),
            gate_cfg=GATES,
            diagnose=_diagnose(),
            evaluate=_evaluate({}),
        )


@pytest.mark.parametrize(
    ('name', 'expected'),
    [('good one', 'good_one'), ('../../evil', '_.._evil'), ('...', 'candidate')],
)
def test_candidate_names_never_escape_their_directory(name: str, expected: str) -> None:
    assert safe_name(name) == expected
    assert '/' not in safe_name(name)


def test_write_loop_result_is_atomic(tmp_path: Path) -> None:
    result = LoopResult(
        stage=2,
        replicate=0,
        rounds=(RoundRecord(round_index=0, packet_hash='p', bundle_hash='b'),),
        final_bundle=tmp_path / 'final',
        stopped_reason='max_rounds',
    )
    write_loop_result(tmp_path, result)
    document = json.loads((tmp_path / 'loop_result.json').read_text(encoding='utf-8'))
    assert document['rounds'][0]['packet_hash'] == 'p'
    assert document['final_bundle'].endswith('final')


def test_the_firewall_also_covers_keyframe_paths(
    bundle_root: Path, tmp_path: Path
) -> None:
    builder = FakeBuilder([_proposal(_patch('good'))])
    with pytest.raises(LeakageError, match='mme_er_wild_000001'):
        run_stage(
            stage=2,
            replicate=0,
            bundle_root=bundle_root,
            out_dir=tmp_path / 'construction',
            builder=builder,
            cfg=_config(),
            gate_cfg=GATES,
            diagnose=_diagnose(keyframes=('frames/mme_er_wild_000001/f0.jpg',)),
            evaluate=_evaluate({}),
            forbidden_ids=('mme_er_wild_000001',),
        )
    assert builder.calls == ()


# --- the verifier, the load check, and the cost record (ADR-0006 §8) -------------


def _vetoing(approved: tuple[str, ...], vetoed: tuple[str, ...]) -> FakeVerifier:
    """A verifier whose round-0 verdicts are scripted; later rounds approve all."""
    verdicts = tuple(Verdict(name=name, approved=True) for name in approved) + tuple(
        Verdict(name=name, approved=False, issues=('2. label forcing',))
        for name in vetoed
    )
    return FakeVerifier(
        [VerificationReport(verdicts=verdicts, summary='round 0 review')]
    )


def test_a_vetoed_candidate_is_never_written_or_scored(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'
    evaluated: list[str] = []

    def evaluate(bundle: Path) -> CandidateScores:
        evaluated.append(bundle.name)
        return _evaluate({'good': 0.7, 'forced': 0.9})(bundle)

    verifier = _vetoing(approved=('good',), vetoed=('forced',))
    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=FakeBuilder(
            [
                _proposal(
                    _patch('good'),
                    _patch('forced', 'routing.yaml'),
                    _patch('off_limits', 'verification/cross_modal.py'),
                )
            ]
        ),
        cfg=_config(rounds=1),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=evaluate,
        verifier=verifier,
    )

    # The verifier only sees candidates that passed the deterministic checks.
    assert [c.name for c in verifier.requests[0].candidates] == ['good', 'forced']
    assert evaluated == ['harness', 'good']
    assert not (out_dir / 'round_00' / 'candidates' / 'forced').exists()
    assert result.rounds[0].accepted == 'good'
    assert result.rounds[0].vetoed == ('forced',)
    assert result.rounds[0].incumbent_recognition == 0.5
    round_dir = out_dir / 'round_00'
    document = json.loads((round_dir / 'candidates.json').read_text(encoding='utf-8'))
    assert document['vetoed'] == [{'name': 'forced', 'issues': ['2. label forcing']}]
    assert [row['name'] for row in document['rejected']] == ['off_limits']
    verification = json.loads(
        (round_dir / 'verification.json').read_text(encoding='utf-8')
    )
    assert verification['model_id'] == 'fake-verifier'
    assert verification['report']['summary'] == 'round 0 review'
    assert [c['name'] for c in verification['request']['candidates']] == [
        'good',
        'forced',
    ]
    review = (round_dir / 'review.txt').read_text(encoding='utf-8')
    assert '### Candidate 2: forced' in review
    assert 'Prompt revision forced.' in review


def test_the_verifier_is_not_called_when_nothing_survives_validation(
    bundle_root: Path, tmp_path: Path
) -> None:
    verifier = FakeVerifier()
    run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=tmp_path / 'construction',
        builder=FakeBuilder([_proposal(_patch('bad', 'verification/cross_modal.py'))]),
        cfg=_config(rounds=1),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({}),
        verifier=verifier,
    )
    assert verifier.requests == ()
    assert not (tmp_path / 'construction' / 'round_00' / 'verification.json').exists()


def test_the_firewall_covers_the_review_text_too(
    bundle_root: Path, tmp_path: Path
) -> None:
    leaking = _patch('good')
    leaking = leaking.model_copy(update={'rationale': 'fixes mme_er_wild_000002'})
    with pytest.raises(LeakageError, match='mme_er_wild_000002'):
        run_stage(
            stage=2,
            replicate=0,
            bundle_root=bundle_root,
            out_dir=tmp_path / 'construction',
            builder=FakeBuilder([_proposal(leaking)]),
            cfg=_config(rounds=1),
            gate_cfg=GATES,
            diagnose=_diagnose(),
            evaluate=_evaluate({}),
            verifier=FakeVerifier(),
            forbidden_ids=('mme_er_wild_000002',),
        )


def test_a_candidate_the_harness_cannot_load_is_rejected_after_writing(
    bundle_root: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'construction'

    def bundle_check(directory: Path) -> tuple[str, ...]:
        return ('missing {noun}',) if directory.name == 'broken' else ()

    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=FakeBuilder([_proposal(_patch('broken'), _patch('good'))]),
        cfg=_config(rounds=1),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'good': 0.7, 'broken': 0.9}),
        bundle_check=bundle_check,
    )

    assert result.rounds[0].accepted == 'good'
    assert [c.name for c in result.rounds[0].candidates] == ['good']
    rejected = json.loads(
        (out_dir / 'round_00' / 'candidates.json').read_text(encoding='utf-8')
    )['rejected']
    assert rejected == [{'name': 'broken', 'violations': ['missing {noun}']}]


def test_every_round_records_what_it_cost(bundle_root: Path, tmp_path: Path) -> None:
    ledger = tmp_path / 'ledger.jsonl'
    out_dir = tmp_path / 'construction'

    def bill(role: str, cost: float) -> None:
        record(
            ledger,
            LedgerEntry(
                role=role,
                model_id='m',
                experiment='e',
                input_tokens=10,
                output_tokens=5,
                latency_s=0.1,
                cost_usd=cost,
            ),
        )

    def evaluate(bundle: Path) -> CandidateScores:
        bill('target', 0.0)
        bill('judge_proxy', 0.25)
        return _evaluate({'first': 0.6, 'second': 0.4})(bundle)

    class BillingBuilder(FakeBuilder):
        def propose(self, packet_text: str, image_paths: tuple[str, ...]):
            bill('builder', 1.0)
            return super().propose(packet_text, image_paths)

    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=out_dir,
        builder=BillingBuilder(
            [_proposal(_patch('first')), _proposal(_patch('second'))]
        ),
        cfg=_config(rounds=2, early_stop=5),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=evaluate,
        ledger=ledger,
    )

    assert result.baseline_cost.total_usd == pytest.approx(0.25)
    assert result.baseline_cost.by_role['target'].calls == 1
    first = result.rounds[0].cost
    assert first.total_usd == pytest.approx(1.25)
    assert first.by_role['builder'].input_tokens == 10
    assert first.by_role['judge_proxy'].cost_usd == pytest.approx(0.25)
    assert result.rounds[1].cost.total_usd == pytest.approx(1.25)
    assert result.total_cost.total_usd == pytest.approx(0.25 + 1.25 + 1.25)
    cost = json.loads((out_dir / 'round_01' / 'cost.json').read_text(encoding='utf-8'))
    assert cost['total_usd'] == pytest.approx(1.25)
    written = json.loads((out_dir / 'loop_result.json').read_text(encoding='utf-8'))
    assert written['rounds'][0]['cost']['by_role']['builder']['calls'] == 1
    assert written['total_cost']['total_usd'] == pytest.approx(2.75)


def test_rounds_without_a_ledger_record_an_empty_cost(
    bundle_root: Path, tmp_path: Path
) -> None:
    result = run_stage(
        stage=2,
        replicate=0,
        bundle_root=bundle_root,
        out_dir=tmp_path / 'construction',
        builder=FakeBuilder([_proposal(_patch('good'))]),
        cfg=_config(rounds=1),
        gate_cfg=GATES,
        diagnose=_diagnose(),
        evaluate=_evaluate({'good': 0.7}),
    )
    assert result.rounds[0].cost == RoundCost.empty()
    assert result.total_cost.by_role == {}
