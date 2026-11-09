"""Wiring the Builder loop: what a round diagnoses and how a candidate is scored.

``emoharness.builder.loop`` owns the control flow and takes ``diagnose`` and
``evaluate`` as parameters. This module supplies them: the harness runs on
D_build to produce the failure clusters a packet is built from, and on D_val to
produce the Recognition objective, the proxy Reasoning gate, the regression rate,
and the call and latency budgets the other gates read.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import ValidationError as PydanticValidationError

from emoharness.builder.clusters import FailureRecord, failure_records_from_outputs
from emoharness.builder.loop import (
    AssessBuild,
    CompleteEvaluation,
    Diagnose,
    Evaluate,
    LoopResult,
    run_stage,
)
from emoharness.builder.packet import DiagnosisPacket, build_packet
from emoharness.builder.patch import read_bundle_files
from emoharness.builder.regression import regression_rate, update_regression_set
from emoharness.builder.schema import BuildAssessment, CandidateScores, SelectionMetrics
from emoharness.common.constants import GATE_SAMPLE_SIZE, NUM_STEPS, SUBSET_SEED
from emoharness.common.errors import ValidationError
from emoharness.common.hashing import directory_hash, sha256_file
from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.common.parallel import map_ordered
from emoharness.common.stats import mean, percentile
from emoharness.data.isolation import assert_isolated
from emoharness.data.label_set import normalize_label
from emoharness.data.manifest import Example
from emoharness.data.prompt_template import parse_prompt
from emoharness.evaluation.judge import JudgeInput, judge_input_for
from emoharness.evaluation.quality import context_quality
from emoharness.evaluation.schema import MethodScores
from emoharness.evaluation.scoreboard import Scoreboard
from emoharness.harness._verification import apply_bundle_verification
from emoharness.harness.allowlist import check_verification_source
from emoharness.harness.loader import HarnessBundle, load_bundle
from emoharness.harness.schema import (
    AudioEvidence,
    EvidenceItem,
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    VisualEvidence,
)
from emoharness.runner.context import RunContext
from emoharness.runner.items import by_id, gold_of, load_benchmark, select_items
from emoharness.runner.judging import ScoreSpec
from emoharness.runner.paths import ProjectPaths, guard_write, source_root
from emoharness.runner.perception import judge_frames_by_id
from emoharness.runner.rows import inputs_for, run_harness

BUILD_ITEMS: Final = 'external:build'
VAL_ITEMS: Final = 'external:val'
EXT_TEST_ITEMS: Final = 'external:ext_test'
STAGE_TWO_PARENT_REPLICATE: Final = 0
MIN_FORBIDDEN_ID_LENGTH: Final = 4
_P95: Final = 95
_GIT_TIMEOUT_S: Final = 10.0

_LOGGER = get_logger(__name__)


class RegressionTracker:
    """The regression set: items an accepted harness once solved. It only grows.

    ``rate`` remembers what each evaluated bundle solved; ``promote`` folds the
    incumbent's solved items in at the start of the next round, which is the only
    moment the loop tells us which bundle was accepted.
    """

    def __init__(self) -> None:
        self._regression: frozenset[str] = frozenset()
        self.incumbent_bundle: Path | None = None
        self._solved: Mapping[str, frozenset[str]] = {}

    @property
    def regression_set(self) -> frozenset[str]:
        return self._regression

    def rate(self, bundle: Path, solved: frozenset[str]) -> float:
        """Share of the regression set ``bundle`` no longer solves."""
        self._solved = {**self._solved, str(bundle): solved}
        return regression_rate(self._regression, solved)

    def promote(self, bundle: Path) -> None:
        """Fold an accepted bundle's solved items into the regression set."""
        self.incumbent_bundle = bundle
        solved = self._solved.get(str(bundle))
        if solved is not None:
            self._regression = update_regression_set(self._regression, solved)


@dataclass(frozen=True)
class ConstructionData:
    """The two construction splits and the fixed proxy-judge gate sample."""

    build: tuple[Example, ...]
    val: tuple[Example, ...]
    gate_ids: frozenset[str]


def gate_sample_ids(
    examples: Sequence[Example], size: int = GATE_SAMPLE_SIZE
) -> frozenset[str]:
    """A fixed sample of D_val ids, drawn once with the pre-registered seed."""
    pool = sorted(example.example_id for example in examples)
    rng = random.Random(SUBSET_SEED)  # noqa: S311 -- reproducible sampling, not crypto
    return frozenset(rng.sample(pool, min(size, len(pool))))


def load_construction_data(ctx: RunContext) -> ConstructionData:
    """D_build, D_val, and the gate sample drawn deterministically from D_val."""
    from emoharness.runner.items import load_external

    external = load_external(ctx.paths)
    if ctx.builder.require_isolated_splits:
        assert_isolated(external)
    build = select_items(external, BUILD_ITEMS, ctx.paths)
    val = select_items(external, VAL_ITEMS, ctx.paths)
    if not build or not val:
        raise ValidationError('construction needs a non-empty D_build and D_val')
    return ConstructionData(
        build=build,
        val=val,
        gate_ids=gate_sample_ids(val, ctx.builder.gates.gate_sample_size),
    )


def _scoreboard(
    outputs: Sequence[HarnessOutput], examples: Sequence[Example]
) -> Scoreboard:
    """Local scores for one bundle's run over one construction split."""
    return Scoreboard.from_outputs(outputs, by_id(examples))


def _require_mean(board: Scoreboard, context: str) -> float:
    """The flat mean Recognition, or a refusal: a candidate with no items has no score."""
    value = board.mean()
    if value is None:
        raise ValidationError(f'{context}: no items were scored')
    return value


def make_diagnose(
    ctx: RunContext, data: ConstructionData, stage: int, tracker: RegressionTracker
) -> Diagnose:
    """A round's diagnosis: run the incumbent on D_build and cluster its failures."""
    target = ctx.target()
    perceived = inputs_for(data.build, ctx)
    gold = gold_of(data.build)
    keyframes = judge_frames_by_id(data.build, ctx.clip_cache(), ctx.preprocess)

    def diagnose(bundle_dir: Path, round_index: int) -> DiagnosisPacket:
        tracker.promote(bundle_dir)
        outputs = run_harness(data.build, bundle_dir, target, ctx, perceived)
        records: tuple[FailureRecord, ...] = failure_records_from_outputs(
            outputs, gold, keyframes, normalize_label
        )
        board = _scoreboard(outputs, data.build)
        recognition = _require_mean(board, f'stage {stage} round {round_index}')
        packet = build_packet(
            stage,
            round_index,
            read_bundle_files(bundle_dir),
            recognition,
            records,
            ctx.builder.packet,
            max_candidates=ctx.builder.max_candidates_per_round,
        )
        correct = failure_records_from_outputs(
            outputs, gold, {}, normalize_label, correct=True
        )
        # Prefer boundary examples with labels involved in the remaining failures.
        boundary_labels = {
            label for record in records for label in (*record.gold, *record.predicted)
        }
        groups: dict[tuple[str, ...], list[FailureRecord]] = {}
        for item in sorted(correct, key=lambda item: item.example_id):
            groups.setdefault(item.gold, []).append(item)
        ordered = sorted(
            groups, key=lambda labels: (not bool(set(labels) & boundary_labels), labels)
        )
        balanced = [
            groups[labels][depth]
            for depth in range(ctx.builder.packet.max_correct_examples)
            for labels in ordered
            if depth < len(groups[labels])
        ]
        correct = tuple(balanced[: ctx.builder.packet.max_correct_examples])
        tools = {
            'face_crops': 'available'
            if all(item.face_crop_paths for item in perceived.values())
            else 'unavailable; do not enable sampling.face_crops',
            'extra_images': 'cached frames; bounded by the image budget',
            'audio': f'ASR backend={ctx.preprocess.asr.backend}; prosody=fixed openSMILE rules; no model analysis or opinion',
            'video_frames': 'sampling.yaml may add `video: {min_fps, min_frames, '
            'max_frames}`: the visual call then samples clamp(ceil(duration * '
            'min_fps), min_frames, max_frames) frames (max_frames <= 64) instead of '
            'the fixed target cap',
        }
        if ctx.preprocess.utterances is not None:
            tools['utterances'] = (
                'speech evidence items are timed utterances: '
                '"speech: [start-end s] S<k>: text (prosody phrases)", one per '
                'Whisper segment; S<k> is a voice-cluster id from speaker '
                'embeddings, prosody is per utterance against utterance-level '
                'reference statistics; no model judges affect'
            )
        update: dict[str, object] = {
            'correct_examples': correct,
            'tool_availability': tools,
        }
        if ctx.builder.selection_protocol == 'context_quality':
            update['build_quality'] = context_quality(outputs, by_id(data.build))
        return packet.model_copy(update=update)

    return diagnose


def _judge_inputs(
    outputs: Sequence[HarnessOutput],
    data: ConstructionData,
    keyframes: Mapping[str, tuple[str, ...]],
    gold: Mapping[str, tuple[str, ...]],
) -> tuple[JudgeInput, ...]:
    return tuple(
        judge_input_for(
            output.example_id,
            keyframes.get(output.example_id, ()),
            gold[output.example_id],
            output.response_text,
        )
        for output in outputs
        if output.example_id in data.gate_ids
    )


def _score_specs(
    board: Scoreboard, gold: Mapping[str, tuple[str, ...]]
) -> Mapping[str, ScoreSpec]:
    """What a stand-in proxy judge needs per item, from the local scoreboard."""
    return {
        example_id: (
            round(score.recognition * len(gold[example_id])),
            len(gold[example_id]),
            NUM_STEPS,
        )
        for example_id, score in board.items.items()
    }


def proxy_reasoning(
    outputs: Sequence[HarnessOutput],
    board: Scoreboard,
    data: ConstructionData,
    ctx: RunContext,
    keyframes: Mapping[str, tuple[str, ...]],
    gold: Mapping[str, tuple[str, ...]],
) -> float | None:
    """Mean proxy Rea-S over the fixed gate sample, or None when it is empty."""
    inputs = _judge_inputs(outputs, data, keyframes, gold)
    if not inputs:
        return None
    scorer = ctx.step_scorer(ctx.step_score_table(_score_specs(board, gold)))
    rated = map_ordered(scorer.score, inputs)
    return mean(item.reasoning for item in rated)


def make_evaluate(
    ctx: RunContext,
    data: ConstructionData,
    tracker: RegressionTracker,
    *,
    include_reasoning: bool = True,
    output_cache: dict[Path, tuple[HarnessOutput, ...]] | None = None,
) -> Evaluate:
    """Score one bundle on D_val: Recognition, proxy Reasoning, regression, budgets."""
    target = ctx.target()
    perceived = inputs_for(data.val, ctx)
    gold = gold_of(data.val)
    keyframes = judge_frames_by_id(data.val, ctx.clip_cache(), ctx.preprocess)

    selection_boards: dict[Path, Scoreboard] = {}
    scaffold_per_task: dict[str, float] | None = None

    def evaluate(bundle_dir: Path) -> CandidateScores:
        nonlocal scaffold_per_task
        outputs = run_harness(data.val, bundle_dir, target, ctx, perceived)
        if output_cache is not None:
            output_cache.clear()
            output_cache[bundle_dir] = tuple(outputs)
        board = _scoreboard(outputs, data.val)
        solved = frozenset(
            i for i, score in board.items.items() if score.recognition >= 1.0
        )
        latencies = [sum(t.latency_s for t in o.traces) for o in outputs]
        recognition = _require_mean(board, bundle_dir.name)
        quality = None
        if ctx.builder.selection_protocol == 'context_quality':
            quality = context_quality(outputs, by_id(data.val))
            recognition = quality.objective
        selection = None
        if ctx.builder.selection_protocol == 'macro_feedback':
            per_task = {task.value: value for task, value in board.by_task().items()}
            if set(per_task) != {'ER_Lab', 'ER_Wild', 'SA'}:
                raise ValidationError(
                    'macro selection requires exactly three seen tasks'
                )
            if scaffold_per_task is None:
                scaffold_per_task = dict(per_task)
            before = (
                selection_boards[tracker.incumbent_bundle]
                if tracker.incumbent_bundle is not None
                else board
            )
            before_tasks = {
                task.value: value for task, value in before.by_task().items()
            }
            selection = SelectionMetrics(
                macro=mean(per_task.values()),
                pooled=recognition,
                per_task=per_task,
                scaffold_per_task=scaffold_per_task,
                incumbent_per_task=before_tasks,
                per_task_delta={
                    task: value - before_tasks[task] for task, value in per_task.items()
                },
                macro_delta=mean(per_task.values()) - mean(before_tasks.values()),
                pooled_delta=recognition
                - _require_mean(before, 'selection-dev incumbent'),
                improved_count=sum(
                    s.recognition > before.items[i].recognition
                    for i, s in board.items.items()
                ),
                regressed_count=sum(
                    s.recognition < before.items[i].recognition
                    for i, s in board.items.items()
                ),
            )
            selection_boards[bundle_dir] = board
            recognition = selection.macro
        return CandidateScores(
            name=bundle_dir.name,
            recognition=recognition,
            selection=selection,
            reasoning=proxy_reasoning(outputs, board, data, ctx, keyframes, gold)
            if include_reasoning
            else None,
            regression_rate=tracker.rate(bundle_dir, solved),
            mean_calls=mean(float(o.target_calls) for o in outputs),
            p95_latency_s=percentile(latencies, _P95),
            quality=quality,
        )

    return evaluate


def forbidden_ids(ctx: RunContext) -> tuple[str, ...]:
    """Every benchmark example id and video id, for the packet firewall."""
    try:
        examples = load_benchmark(ctx.paths)
    except ValidationError:
        _LOGGER.warning(
            'no benchmark manifest; the packet firewall has nothing to check'
        )
        return ()
    found: set[str] = set()
    for example in examples:
        media = PurePosixPath(example.media_path)
        found.update({example.example_id, media.name, media.stem})
    # A recording the construction set legitimately contains is not a leak when
    # the benchmark reuses the same file under another task (MME's seen and
    # held-out tasks share media); its example ids stay forbidden.
    found -= _construction_media_names(ctx)
    # A bare number is not an identifier: ER_Wild stems like ``2670`` would match
    # any count or timestamp in a packet, and every id that contains them.
    return tuple(
        sorted(
            i for i in found if len(i) >= MIN_FORBIDDEN_ID_LENGTH and not i.isdigit()
        )
    )


def _construction_media_names(ctx: RunContext) -> set[str]:
    """File names and stems of every construction clip; empty without a manifest."""
    from emoharness.runner.items import load_external

    try:
        external = load_external(ctx.paths)
    except ValidationError:
        return set()
    names: set[str] = set()
    for example in external:
        media = PurePosixPath(example.media_path)
        names.update({media.name, media.stem})
    return names


def base_bundle_for(stage: int, ctx: RunContext) -> Path:
    """Where a stage's loop starts: the Stage 1 scaffold, then the first Stage 2 run."""
    if ctx.bundle_override is not None:
        return ctx.bundle_override
    if stage <= 2:
        return ctx.paths.bundle
    return ctx.paths.stage_dir(2, STAGE_TWO_PARENT_REPLICATE) / 'final'


def bundle_problems(bundle_dir: Path) -> tuple[str, ...]:
    """Why a written candidate cannot run, as the loader would say it; empty if it loads."""
    try:
        load_bundle(bundle_dir)
    except (ValidationError, PydanticValidationError) as exc:
        return (str(exc),)
    return ()


def run_construction(stage: int, replicate: int, ctx: RunContext) -> LoopResult:
    """Run one stage's construction loop and write its provenance."""
    if stage < 2:
        raise ValidationError(
            'stage 1 is the human scaffold; the Builder may not edit it'
        )
    base = base_bundle_for(stage, ctx)
    if not base.is_dir():
        raise ValidationError(f'no parent harness bundle at {base}')
    out_dir = ctx.paths.stage_dir(stage, replicate)
    if (out_dir / 'final').exists() and not ctx.force:
        raise ValidationError(f'destination already exists: {out_dir / "final"}')
    if ctx.force and out_dir.exists():
        shutil.rmtree(out_dir)
    data = load_construction_data(ctx)
    tracker = RegressionTracker()
    _LOGGER.info(
        'stage %s replicate %s: %d build items, %d val items, %d gate items',
        stage,
        replicate,
        len(data.build),
        len(data.val),
        len(data.gate_ids),
    )
    builder_client = ctx.builder_client(stage)
    output_cache: dict[Path, tuple[HarnessOutput, ...]] = {}
    write_json_atomic(
        out_dir / 'run_config.json',
        {
            'runtime_protocol': {
                'macro_feedback': 'v4-macro-feedback-workflow-v1',
                'context_quality': 'v5-context-quality',
            }.get(ctx.builder.selection_protocol, 'v3-feedback-and-decision-seams'),
            'runtime_hash': directory_hash(source_root() / 'src' / 'emoharness'),
            'parent_bundle_hash': load_bundle(base).content_hash(),
            'manifest': str(ctx.paths.external_manifest),
            'manifest_hash': sha256_file(ctx.paths.external_manifest),
            'target': ctx.target_model_config().model_dump(mode='json'),
            'builder': ctx.builder.model_dump(mode='json'),
            'preprocess': ctx.preprocess.model_dump(mode='json'),
            'preprocess_hash': ctx.preprocess.config_hash(),
            'gates': ctx.gate_config().model_dump(mode='json'),
            'build_size': len(data.build),
            'val_size': len(data.val),
        },
    )
    result = run_stage(
        stage,
        replicate,
        base,
        out_dir,
        builder_client,
        ctx.builder,
        ctx.gate_config(),
        make_diagnose(ctx, data, stage, tracker),
        make_evaluate(
            ctx, data, tracker, include_reasoning=False, output_cache=output_cache
        ),
        check_verification_source,
        forbidden_ids(ctx),
        verifier=ctx.builder_verifier(stage),
        bundle_check=make_bundle_check(ctx, data),
        ledger=ctx.ledger,
        assess_build=make_build_assessment(ctx, data),
        complete_evaluation=make_complete_evaluation(ctx, data, output_cache),
    )
    stamp_manifest(result, load_bundle(base).content_hash(), builder_client.model_id)
    return result


def stamp_manifest(result: LoopResult, parent_hash: str, builder_id: str) -> Path:
    """Rewrite the final bundle's manifest with the construction-stage provenance.

    The Builder may not edit ``manifest.json`` (it is outside every stage's globs),
    so the loop records stage, parent hash, and Builder here.
    """
    accepted = tuple(r.accepted for r in result.rounds if r.accepted is not None)
    notes = (
        f'{len(accepted)} accepted patch(es) over {len(result.rounds)} round(s); '
        f'stopped: {result.stopped_reason}'
    )
    payload = {
        'version': f'stage{result.stage}-r{result.replicate}',
        'stage': result.stage,
        'parent': parent_hash,
        'builder': builder_id,
        'created': datetime.now(timezone.utc).date().isoformat(),
        'notes': notes,
    }
    path = result.final_bundle / 'manifest.json'
    write_json_atomic(path, payload)
    return path


def evaluate_external_test(bundle_dir: Path, ctx: RunContext) -> MethodScores:
    """The once-per-frozen-stage D_ext-test number."""
    from emoharness.runner.items import load_external

    examples = select_items(load_external(ctx.paths), EXT_TEST_ITEMS, ctx.paths)
    if not examples:
        raise ValidationError('D_ext-test is empty; run prepare_external first')
    outputs = run_harness(examples, bundle_dir, ctx.target(), ctx)
    board = _scoreboard(outputs, examples)
    return board.method_scores('harness', EXT_TEST_ITEMS)


def git_head(*candidates: Path) -> str | None:
    """The first candidate directory that is a git checkout, as a commit id."""
    for directory in candidates:
        try:
            result = subprocess.run(
                ['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_S,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        head = result.stdout.strip()
        if head:
            return head
    return None


def write_freeze_record(
    bundle_dir: Path, paths: ProjectPaths, force: bool = False
) -> Path:
    """Record what was frozen: the bundle's content hash, the commit, and when."""
    bundle = load_bundle(bundle_dir)
    payload = {
        'bundle': str(bundle_dir),
        'manifest_stage': bundle.manifest.stage,
        'manifest_version': bundle.manifest.version,
        'content_hash': bundle.content_hash(),
        'git_head': git_head(paths.root, source_root()),
        'frozen_at': datetime.now(timezone.utc).isoformat(),
    }
    path = guard_write(paths.freeze_record, force)
    write_json_atomic(path, payload)
    return path


def make_complete_evaluation(
    ctx: RunContext,
    data: ConstructionData,
    output_cache: dict[Path, tuple[HarnessOutput, ...]],
) -> CompleteEvaluation:
    """Judge the exact outputs cheap gates examined, without rerunning the Target."""
    keyframes = judge_frames_by_id(data.val, ctx.clip_cache(), ctx.preprocess)
    gold = gold_of(data.val)

    def complete(bundle: Path, scores: CandidateScores) -> CandidateScores:
        outputs = output_cache.pop(bundle)
        reasoning = proxy_reasoning(
            outputs, _scoreboard(outputs, data.val), data, ctx, keyframes, gold
        )
        if reasoning is None:
            raise ValidationError('proxy reasoning gate returned no score')
        return scores.model_copy(update={'reasoning': reasoning})

    return complete


def make_build_assessment(ctx: RunContext, data: ConstructionData) -> AssessBuild:
    """Evaluate improvements and collateral damage only on visible D_build."""
    target = ctx.target()
    perceived = inputs_for(data.build, ctx)
    # Only the incumbent board is retained; candidate output does not accumulate.
    incumbent_boards: dict[Path, Scoreboard] = {}

    def board_for(bundle: Path) -> Scoreboard:
        return _scoreboard(
            run_harness(data.build, bundle, target, ctx, perceived), data.build
        )

    def assess(incumbent: Path, candidate: Path) -> BuildAssessment:
        if incumbent not in incumbent_boards:
            incumbent_boards.clear()
            incumbent_boards[incumbent] = board_for(incumbent)
        before = incumbent_boards[incumbent]
        after = board_for(candidate)
        improved = tuple(
            sorted(
                i
                for i in before.items
                if after.items[i].recognition > before.items[i].recognition
            )
        )
        regressed = tuple(
            sorted(
                i
                for i in before.items
                if after.items[i].recognition < before.items[i].recognition
            )
        )
        return BuildAssessment(
            before=_require_mean(before, 'D_build incumbent'),
            after=_require_mean(after, 'D_build candidate'),
            improved_count=len(improved),
            regressed_count=len(regressed),
            improved_examples=improved[:8],
            regressed_examples=regressed[:8],
        )

    return assess


def make_bundle_check(
    ctx: RunContext, data: ConstructionData
) -> Callable[[Path], tuple[str, ...]]:
    """Refuse valid YAML that requests a capability the inputs do not provide."""
    perceived = inputs_for((*data.build, *data.val), ctx)
    crops_available = bool(perceived) and all(
        item.face_crop_paths for item in perceived.values()
    )

    def check(directory: Path) -> tuple[str, ...]:
        problems = bundle_problems(directory)
        if problems:
            return problems
        bundle = load_bundle(directory)
        if not ctx.builder.workflow_search and (
            bundle.routing.workflow != 'multistep'
            or bundle.routing.rank_policy != 'bundle'
            or bundle.routing.audio_weight != 1.0
            or bundle.routing.visual_weight != 1.0
        ):
            return (
                'workflow/rank policy switches are available only in workflow_search',
            )
        if bundle.sampling.face_crops.enabled and (
            not crops_available or bundle.sampling.face_crops.max_crops == 0
        ):
            return (
                'sampling.face_crops requested but no usable crop inputs are available',
            )
        return _probe_verification(bundle, data.build[0])

    return check


def _probe_verification(bundle: HarnessBundle, example: Example) -> tuple[str, ...]:
    """Execute pure functions on boundary inputs before reviewing or scoring a patch."""
    parsed = parse_prompt(example.prompt)
    labels = parsed.label_set[:2]
    observations = (
        EvidenceItem(id='v1', modality='visual', cue='visible movement'),
        EvidenceItem(id='a1', modality='audio', cue='measured pitch variation'),
    )
    for items in ((), observations):
        record = EvidenceRecord(
            visual=VisualEvidence(),
            audio=AudioEvidence(speech_present=None),
            text=parsed,
            items=items,
        )
        hypotheses = HypothesisSet(
            coarse='unspecified',
            hypotheses=tuple(
                Hypothesis(
                    label=label,
                    confidence=0.6 - index * 0.1,
                    support=(observations[index].id,) if items else (),
                )
                for index, label in enumerate(labels)
            ),
        )
        result = apply_bundle_verification(bundle, record, hypotheses)
        if result.notes:
            return (f'verification contract probe: {result.notes}',)
    return ()
