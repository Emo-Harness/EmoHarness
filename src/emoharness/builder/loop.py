"""The controlled construction loop (ADR-0006 §1, §4, §7, §8).

One Builder call per round, every candidate validated before it is written, the
survivors reviewed by the verifier (the Builder's own model, asked whether each
patch is a general rule or an overfit), every approved candidate scored by an
injected evaluator, and the whole round recorded on disk with what it cost.
``diagnose`` and ``evaluate`` are parameters so the loop runs without a GPU, a
harness runtime, or a network.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.builder.client import Builder
from emoharness.builder.config import BuilderConfig
from emoharness.builder.gates import GateConfig, GateResult, select_candidate
from emoharness.builder.packet import (
    DiagnosisPacket,
    assert_no_forbidden_ids,
    render_packet_text,
)
from emoharness.builder.patch import (
    IGNORED_DIRS,
    CodeChecker,
    apply_patch,
    patch_hash,
    validate_patch,
)
from emoharness.builder.schema import BuilderProposal, CandidatePatch, CandidateScores
from emoharness.builder.verifier import (
    ReviewRequest,
    VerificationReport,
    Verifier,
    render_review_text,
    review_request,
)
from emoharness.common.errors import ValidationError
from emoharness.common.hashing import directory_hash, sha256_text
from emoharness.common.io import write_json_atomic
from emoharness.common.ledger import LedgerSummary, read_entries, summarize_by
from emoharness.common.logging import get_logger

Diagnose = Callable[[Path, int], DiagnosisPacket]
Evaluate = Callable[[Path], CandidateScores]
#: Reasons a written candidate bundle cannot run; empty means it loads.
BundleCheck = Callable[[Path], tuple[str, ...]]

FINAL_DIR: Final = 'final'
CANDIDATES_DIR: Final = 'candidates'
PACKET_TEXT_FILE: Final = 'packet.txt'
PACKET_JSON_FILE: Final = 'packet.json'
PROPOSAL_FILE: Final = 'proposal.json'
REVIEW_TEXT_FILE: Final = 'review.txt'
VERIFICATION_FILE: Final = 'verification.json'
CANDIDATES_FILE: Final = 'candidates.json'
ACCEPTED_FILE: Final = 'accepted.json'
COST_FILE: Final = 'cost.json'
LOOP_RESULT_FILE: Final = 'loop_result.json'
STOP_EARLY: Final = 'early_stop'
STOP_MAX_ROUNDS: Final = 'max_rounds'
_SAFE_CHARS: Final = frozenset(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
)

_LOGGER = get_logger(__name__)


class RoundCost(BaseModel):
    """What one round spent, from the ledger rows it appended, grouped by role.

    ``target`` rows are the student's calls (local, so free, but the token
    counts show the work); ``builder``, ``verifier``, and ``judge_proxy`` rows
    are the paid ones. ``total_usd`` sums every role.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    by_role: Mapping[str, LedgerSummary] = Field(default_factory=dict)
    total_usd: float = Field(default=0.0, ge=0.0)

    @classmethod
    def empty(cls) -> RoundCost:
        return cls()

    @classmethod
    def from_ledger(cls, ledger: Path | None, first_row: int) -> RoundCost:
        """The cost of the rows appended to ``ledger`` from index ``first_row`` on."""
        if ledger is None:
            return cls.empty()
        rows = read_entries(ledger)[first_row:]
        by_role = summarize_by(rows, 'role')
        return cls(
            by_role=by_role,
            total_usd=sum(summary.cost_usd for summary in by_role.values()),
        )


class RoundRecord(BaseModel):
    """What one round did, in the order the loop did it."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    round_index: int = Field(ge=0)
    packet_hash: str
    incumbent_recognition: float | None = Field(default=None, ge=0.0, le=1.0)
    candidates: tuple[CandidateScores, ...] = ()
    vetoed: tuple[str, ...] = Field(
        default=(), description='candidate names the verifier did not approve'
    )
    gate_results: tuple[GateResult, ...] = ()
    accepted: str | None = None
    bundle_hash: str
    cost: RoundCost = Field(default_factory=RoundCost.empty)


class LoopResult(BaseModel):
    """The outcome of one stage for one replicate."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stage: int = Field(ge=1)
    replicate: int = Field(ge=0)
    rounds: tuple[RoundRecord, ...] = ()
    final_bundle: Path
    stopped_reason: str
    baseline_cost: RoundCost = Field(
        default_factory=RoundCost.empty,
        description='scoring the parent bundle before round 0',
    )
    total_cost: RoundCost = Field(default_factory=RoundCost.empty)


@dataclass(frozen=True)
class _Scored:
    patch: CandidatePatch
    directory: Path
    scores: CandidateScores


@dataclass(frozen=True)
class _Review:
    request: ReviewRequest
    text: str
    report: VerificationReport
    model_id: str


@dataclass(frozen=True)
class _Outcome:
    packet: DiagnosisPacket
    packet_text: str
    proposal: BuilderProposal
    review: _Review | None
    scored: tuple[_Scored, ...]
    rejections: tuple[tuple[str, tuple[str, ...]], ...]
    vetoed: tuple[tuple[str, tuple[str, ...]], ...]
    gate_results: tuple[GateResult, ...]
    accepted: _Scored | None
    cost: RoundCost


@dataclass(frozen=True)
class _Context:
    stage: int
    out_dir: Path
    builder: Builder
    cfg: BuilderConfig
    gate_cfg: GateConfig
    diagnose: Diagnose
    evaluate: Evaluate
    code_checker: CodeChecker | None
    forbidden_ids: tuple[str, ...]
    verifier: Verifier | None
    bundle_check: BundleCheck | None
    ledger: Path | None


@dataclass(frozen=True)
class _State:
    bundle: Path
    scores: CandidateScores
    streak: int
    rounds: tuple[RoundRecord, ...]


def run_stage(
    stage: int,
    replicate: int,
    bundle_root: Path,
    out_dir: Path,
    builder: Builder,
    cfg: BuilderConfig,
    gate_cfg: GateConfig,
    diagnose: Diagnose,
    evaluate: Evaluate,
    code_checker: CodeChecker | None = None,
    forbidden_ids: Iterable[str] = (),
    verifier: Verifier | None = None,
    bundle_check: BundleCheck | None = None,
    ledger: Path | None = None,
) -> LoopResult:
    """Run one stage's construction loop and write its provenance under ``out_dir``.

    ``verifier`` reviews each round's valid candidates before any is applied;
    ``bundle_check`` refuses a written candidate the harness cannot load; and
    ``ledger`` is the file whose new rows are summed into each round's cost.
    """
    final_dir = out_dir / FINAL_DIR
    if final_dir.exists():
        raise ValidationError(f'destination already exists: {final_dir}')
    context = _Context(
        stage=stage,
        out_dir=out_dir,
        builder=builder,
        cfg=cfg,
        gate_cfg=gate_cfg,
        diagnose=diagnose,
        evaluate=evaluate,
        code_checker=code_checker,
        forbidden_ids=tuple(forbidden_ids),
        verifier=verifier,
        bundle_check=bundle_check,
        ledger=ledger,
    )
    first_row = _ledger_length(ledger)
    state = _State(
        bundle=bundle_root,
        scores=evaluate(bundle_root),
        streak=0,
        rounds=(),
    )
    baseline_cost = RoundCost.from_ledger(ledger, first_row)
    state, reason = _run_rounds(state, context)
    result = LoopResult(
        stage=stage,
        replicate=replicate,
        rounds=state.rounds,
        final_bundle=_copy_final(state.bundle, final_dir),
        stopped_reason=reason,
        baseline_cost=baseline_cost,
        total_cost=RoundCost.from_ledger(ledger, first_row),
    )
    write_loop_result(out_dir, result)
    return result


def _ledger_length(ledger: Path | None) -> int:
    """How many rows the ledger holds now; new rows from here on belong to us."""
    if ledger is None or not ledger.exists():
        return 0
    with ledger.open('rb') as handle:
        return sum(1 for line in handle if line.strip())


def _run_rounds(state: _State, ctx: _Context) -> tuple[_State, str]:
    for round_index in range(ctx.cfg.max_rounds_per_stage):
        state = _run_round(round_index, state, ctx)
        if state.streak >= ctx.cfg.early_stop_rounds:
            _LOGGER.info(
                'stage %s stopped early after round %s', ctx.stage, round_index
            )
            return state, STOP_EARLY
    return state, STOP_MAX_ROUNDS


def _run_round(round_index: int, state: _State, ctx: _Context) -> _State:
    round_dir = ctx.out_dir / f'round_{round_index:02d}'
    first_row = _ledger_length(ctx.ledger)
    packet = ctx.diagnose(state.bundle, round_index)
    packet_text = render_packet_text(packet, ctx.cfg.packet.target_input_tokens)
    images = packet.image_paths()
    # The firewall covers the media paths too: they are recorded in packet.json.
    assert_no_forbidden_ids('\n'.join((packet_text, *images)), ctx.forbidden_ids)
    proposal = ctx.builder.propose(packet_text, images)
    valid, rejections = _validated(proposal, packet, ctx)
    review, approved, vetoed = _reviewed(valid, packet, ctx)
    scored, unloadable = _score_candidates(approved, state.bundle, round_dir, ctx)
    best, gate_results = select_candidate(
        (item.scores for item in scored), state.scores, ctx.gate_cfg
    )
    accepted = next((item for item in scored if item.scores is best), None)
    outcome = _Outcome(
        packet=packet,
        packet_text=packet_text,
        proposal=proposal,
        review=review,
        scored=scored,
        rejections=rejections + unloadable,
        vetoed=vetoed,
        gate_results=gate_results,
        accepted=accepted,
        cost=RoundCost.from_ledger(ctx.ledger, first_row),
    )
    _write_round_provenance(round_dir, outcome)
    return _advance(round_index, state, outcome)


def _advance(round_index: int, state: _State, outcome: _Outcome) -> _State:
    accepted = outcome.accepted
    bundle = accepted.directory if accepted else state.bundle
    record = RoundRecord(
        round_index=round_index,
        packet_hash=sha256_text(outcome.packet_text),
        incumbent_recognition=state.scores.recognition,
        candidates=tuple(item.scores for item in outcome.scored),
        vetoed=tuple(name for name, _ in outcome.vetoed),
        gate_results=outcome.gate_results,
        accepted=accepted.patch.name if accepted else None,
        bundle_hash=directory_hash(bundle),
        cost=outcome.cost,
    )
    return replace(
        state,
        bundle=bundle,
        scores=accepted.scores if accepted else state.scores,
        streak=0 if accepted else state.streak + 1,
        rounds=(*state.rounds, record),
    )


Rejections = tuple[tuple[str, tuple[str, ...]], ...]


def _validated(
    proposal: BuilderProposal, packet: DiagnosisPacket, ctx: _Context
) -> tuple[tuple[CandidatePatch, ...], Rejections]:
    """The deterministic checks: count, names, paths, and the code allowlist."""
    valid: list[CandidatePatch] = []
    rejections: list[tuple[str, tuple[str, ...]]] = []
    used: set[str] = set()
    for index, patch in enumerate(proposal.candidates):
        violations = _violations(index, patch, packet, used, ctx)
        if violations:
            rejections.append((patch.name, violations))
            _LOGGER.info('rejected candidate %s: %s', patch.name, '; '.join(violations))
            continue
        used.add(safe_name(patch.name))
        valid.append(patch)
    return tuple(valid), tuple(rejections)


def _reviewed(
    valid: tuple[CandidatePatch, ...], packet: DiagnosisPacket, ctx: _Context
) -> tuple[_Review | None, tuple[CandidatePatch, ...], Rejections]:
    """One verifier call over the valid candidates; unapproved ones are vetoed."""
    if ctx.verifier is None or not valid:
        return None, valid, ()
    request = review_request(
        packet.stage,
        packet.round_index,
        packet.bundle_files,
        packet.editable_paths,
        tuple((cluster.key, cluster.count) for cluster in packet.clusters),
        BuilderProposal(candidates=valid),
    )
    text = render_review_text(request)
    assert_no_forbidden_ids(text, ctx.forbidden_ids)
    report = ctx.verifier.review(request)
    approved: list[CandidatePatch] = []
    vetoed: list[tuple[str, tuple[str, ...]]] = []
    for position, patch in enumerate(valid):
        verdict = report.verdict_for(patch.name, position, len(valid))
        if verdict.approved:
            approved.append(patch)
            continue
        vetoed.append((patch.name, verdict.issues or ('not approved',)))
        _LOGGER.info('vetoed candidate %s: %s', patch.name, '; '.join(verdict.issues))
    review = _Review(
        request=request, text=text, report=report, model_id=ctx.verifier.model_id
    )
    return review, tuple(approved), tuple(vetoed)


def _score_candidates(
    approved: tuple[CandidatePatch, ...],
    bundle: Path,
    round_dir: Path,
    ctx: _Context,
) -> tuple[tuple[_Scored, ...], Rejections]:
    """Write each approved candidate, refuse one the harness cannot load, score the rest."""
    scored: list[_Scored] = []
    unloadable: list[tuple[str, tuple[str, ...]]] = []
    for patch in approved:
        directory = round_dir / CANDIDATES_DIR / safe_name(patch.name)
        apply_patch(bundle, patch, directory)
        problems = ctx.bundle_check(directory) if ctx.bundle_check else ()
        if problems:
            unloadable.append((patch.name, tuple(problems)))
            _LOGGER.info(
                'candidate %s does not load: %s', patch.name, '; '.join(problems)
            )
            continue
        scored.append(
            _Scored(patch=patch, directory=directory, scores=ctx.evaluate(directory))
        )
    return tuple(scored), tuple(unloadable)


def _violations(
    index: int,
    patch: CandidatePatch,
    packet: DiagnosisPacket,
    used: set[str],
    ctx: _Context,
) -> tuple[str, ...]:
    if index >= ctx.cfg.max_candidates_per_round:
        return (f'candidate {index} exceeds max_candidates_per_round',)
    if safe_name(patch.name) in used:
        return (f'{patch.name}: duplicate candidate name',)
    return validate_patch(
        patch, ctx.stage, packet.bundle_files.keys(), ctx.code_checker
    )


def safe_name(name: str) -> str:
    """A candidate name reduced to a directory name; model output is never trusted."""
    cleaned = ''.join(char if char in _SAFE_CHARS else '_' for char in name).strip('.')
    return cleaned if cleaned else 'candidate'


def _write_round_provenance(round_dir: Path, outcome: _Outcome) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / PACKET_TEXT_FILE).write_text(outcome.packet_text, encoding='utf-8')
    write_json_atomic(
        round_dir / PACKET_JSON_FILE, outcome.packet.model_dump(mode='json')
    )
    write_json_atomic(
        round_dir / PROPOSAL_FILE, outcome.proposal.model_dump(mode='json')
    )
    if outcome.review is not None:
        (round_dir / REVIEW_TEXT_FILE).write_text(outcome.review.text, encoding='utf-8')
        write_json_atomic(
            round_dir / VERIFICATION_FILE,
            {
                'model_id': outcome.review.model_id,
                'request': outcome.review.request.model_dump(mode='json'),
                'report': outcome.review.report.model_dump(mode='json'),
            },
        )
    write_json_atomic(round_dir / CANDIDATES_FILE, _candidates_document(outcome))
    write_json_atomic(round_dir / ACCEPTED_FILE, _accepted_document(outcome))
    write_json_atomic(round_dir / COST_FILE, outcome.cost.model_dump(mode='json'))


def _candidates_document(outcome: _Outcome) -> Mapping[str, Any]:
    return {
        'evaluated': [
            {
                # ``name`` is the directory-safe form the gate entries use;
                # ``patch_name`` is the Builder's own wording.
                'name': item.scores.name,
                'patch_name': item.patch.name,
                'targeted_cluster': item.patch.targeted_cluster,
                'patch_hash': patch_hash(item.patch),
                'directory': str(item.directory),
                'scores': item.scores.model_dump(mode='json'),
            }
            for item in outcome.scored
        ],
        'gates': [result.model_dump(mode='json') for result in outcome.gate_results],
        'rejected': [
            {'name': name, 'violations': list(violations)}
            for name, violations in outcome.rejections
        ],
        'vetoed': [
            {'name': name, 'issues': list(issues)} for name, issues in outcome.vetoed
        ],
    }


def _accepted_document(outcome: _Outcome) -> Mapping[str, Any]:
    accepted = outcome.accepted
    if accepted is None:
        return {'accepted': None}
    return {
        'accepted': accepted.patch.name,
        'targeted_cluster': accepted.patch.targeted_cluster,
        'rationale': accepted.patch.rationale,
        'patch_hash': patch_hash(accepted.patch),
        'bundle': str(accepted.directory),
        'scores': accepted.scores.model_dump(mode='json'),
    }


def _copy_final(bundle: Path, final_dir: Path) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, final_dir, ignore=shutil.ignore_patterns(*IGNORED_DIRS))
    return final_dir


def write_loop_result(out_dir: Path, result: LoopResult) -> None:
    """Write ``loop_result.json`` for one stage and replicate."""
    write_json_atomic(out_dir / LOOP_RESULT_FILE, result.model_dump(mode='json'))
