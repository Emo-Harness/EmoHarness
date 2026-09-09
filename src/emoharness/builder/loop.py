"""The controlled construction loop (ADR-0006 §1, §4, §7).

One Builder call per round, every candidate validated before it is written, every
candidate scored by an injected evaluator, and the whole round recorded on disk.
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
from emoharness.common.errors import ValidationError
from emoharness.common.hashing import directory_hash, sha256_text
from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger

Diagnose = Callable[[Path, int], DiagnosisPacket]
Evaluate = Callable[[Path], CandidateScores]

FINAL_DIR: Final = 'final'
CANDIDATES_DIR: Final = 'candidates'
PACKET_TEXT_FILE: Final = 'packet.txt'
PACKET_JSON_FILE: Final = 'packet.json'
PROPOSAL_FILE: Final = 'proposal.json'
CANDIDATES_FILE: Final = 'candidates.json'
ACCEPTED_FILE: Final = 'accepted.json'
LOOP_RESULT_FILE: Final = 'loop_result.json'
STOP_EARLY: Final = 'early_stop'
STOP_MAX_ROUNDS: Final = 'max_rounds'
_SAFE_CHARS: Final = frozenset(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
)

_LOGGER = get_logger(__name__)


class RoundRecord(BaseModel):
    """What one round did, in the order the loop did it."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    round_index: int = Field(ge=0)
    packet_hash: str
    candidates: tuple[CandidateScores, ...] = ()
    gate_results: tuple[GateResult, ...] = ()
    accepted: str | None = None
    bundle_hash: str


class LoopResult(BaseModel):
    """The outcome of one stage for one replicate."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stage: int = Field(ge=1)
    replicate: int = Field(ge=0)
    rounds: tuple[RoundRecord, ...] = ()
    final_bundle: Path
    stopped_reason: str


@dataclass(frozen=True)
class _Scored:
    patch: CandidatePatch
    directory: Path
    scores: CandidateScores


@dataclass(frozen=True)
class _Outcome:
    packet: DiagnosisPacket
    packet_text: str
    proposal: BuilderProposal
    scored: tuple[_Scored, ...]
    rejections: tuple[tuple[str, tuple[str, ...]], ...]
    gate_results: tuple[GateResult, ...]
    accepted: _Scored | None


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
) -> LoopResult:
    """Run one stage's construction loop and write its provenance under ``out_dir``."""
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
    )
    state = _State(
        bundle=bundle_root,
        scores=evaluate(bundle_root),
        streak=0,
        rounds=(),
    )
    state, reason = _run_rounds(state, context)
    result = LoopResult(
        stage=stage,
        replicate=replicate,
        rounds=state.rounds,
        final_bundle=_copy_final(state.bundle, final_dir),
        stopped_reason=reason,
    )
    write_loop_result(out_dir, result)
    return result


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
    packet = ctx.diagnose(state.bundle, round_index)
    packet_text = render_packet_text(packet, ctx.cfg.packet.target_input_tokens)
    images = packet.image_paths()
    # The firewall covers the media paths too: they are recorded in packet.json.
    assert_no_forbidden_ids('\n'.join((packet_text, *images)), ctx.forbidden_ids)
    proposal = ctx.builder.propose(packet_text, images)
    scored, rejections = _score_candidates(
        proposal, packet, state.bundle, round_dir, ctx
    )
    best, gate_results = select_candidate(
        (item.scores for item in scored), state.scores, ctx.gate_cfg
    )
    accepted = next((item for item in scored if item.scores is best), None)
    outcome = _Outcome(
        packet=packet,
        packet_text=packet_text,
        proposal=proposal,
        scored=scored,
        rejections=rejections,
        gate_results=gate_results,
        accepted=accepted,
    )
    _write_round_provenance(round_dir, outcome)
    return _advance(round_index, state, outcome)


def _advance(round_index: int, state: _State, outcome: _Outcome) -> _State:
    accepted = outcome.accepted
    bundle = accepted.directory if accepted else state.bundle
    record = RoundRecord(
        round_index=round_index,
        packet_hash=sha256_text(outcome.packet_text),
        candidates=tuple(item.scores for item in outcome.scored),
        gate_results=outcome.gate_results,
        accepted=accepted.patch.name if accepted else None,
        bundle_hash=directory_hash(bundle),
    )
    return replace(
        state,
        bundle=bundle,
        scores=accepted.scores if accepted else state.scores,
        streak=0 if accepted else state.streak + 1,
        rounds=(*state.rounds, record),
    )


def _score_candidates(
    proposal: BuilderProposal,
    packet: DiagnosisPacket,
    bundle: Path,
    round_dir: Path,
    ctx: _Context,
) -> tuple[tuple[_Scored, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    scored: list[_Scored] = []
    rejections: list[tuple[str, tuple[str, ...]]] = []
    used: set[str] = set()
    for index, patch in enumerate(proposal.candidates):
        directory = round_dir / CANDIDATES_DIR / safe_name(patch.name)
        violations = _violations(index, patch, packet, used, ctx)
        if violations:
            rejections.append((patch.name, violations))
            _LOGGER.info('rejected candidate %s: %s', patch.name, '; '.join(violations))
            continue
        used.add(safe_name(patch.name))
        apply_patch(bundle, patch, directory)
        scored.append(
            _Scored(patch=patch, directory=directory, scores=ctx.evaluate(directory))
        )
    return tuple(scored), tuple(rejections)


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
    write_json_atomic(round_dir / CANDIDATES_FILE, _candidates_document(outcome))
    write_json_atomic(round_dir / ACCEPTED_FILE, _accepted_document(outcome))


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
