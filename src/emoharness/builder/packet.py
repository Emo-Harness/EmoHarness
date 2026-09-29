"""The diagnosis packet: what the Builder sees in one round.

Text first. The packet carries the harness files verbatim, the confusion matrix and
cluster counts measured on the construction set, and a bounded number of failure
records; only a sampled subset carries keyframes, and videos are never sent.

The wording never names a held-out capability: the Builder is
told the seen request wording and nothing else about the benchmark.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.builder.clusters import (
    FailureCluster,
    FailureRecord,
    cluster_failures,
    confusion_matrix,
)
from emoharness.builder.config import PacketLimits
from emoharness.builder.patch import editable_paths_for
from emoharness.builder.schema import CandidateFeedback, SelectionMetrics
from emoharness.common.constants import MAX_CANDIDATES_PER_ROUND
from emoharness.common.errors import LeakageError
from emoharness.common.text import truncate

#: Directory the Stage 3 verification functions live in, relative to the bundle.
VERIFICATION_DIR: Final = 'verification'
CHARS_PER_TOKEN: Final = 4
TRUNCATION_MARKER: Final = ' ...[truncated]'
_FULL: Final[None] = None
# Progressively smaller renderings; the first one that fits the budget is used.
_TRUNCATION_STEPS: Final[tuple[tuple[int | None, int | None], ...]] = (
    (_FULL, _FULL),
    (1200, _FULL),
    (400, 1200),
    (0, 400),
    (0, 0),
)

# Substituted with str.replace, not str.format: the template contains JSON braces.
_CANDIDATE_SLOT: Final = '{max_candidates}'
_INSTRUCTIONS: Final = """You improve a harness that runs a frozen video model.

The harness answers one kind of request: "As an emotion recognition expert, identify
the most prominent emotion displayed in the video. Candidate labels: ... ." The same
wording occurs with "sentiment" in place of "emotion". Answers must use labels copied
verbatim from the candidate list of the request the harness is given.

Your objective, in priority order:
1. raise the share of construction items whose emitted label matches the gold label;
2. keep every item the current harness already answers correctly;
3. keep the mean model calls per item and the p95 latency within budget.

Rules:
- Reply with one JSON object and nothing else, of exactly this shape:
  {"summary": "...", "continue_search": true, "stop_reason": "",
  "candidates": [{"name": "...", "targeted_cluster": "...",
  "rationale": "...", "files": [{"path": "...", "content": "..."}]}]}
- "continue_search" is yours to decide: answer false, with a one-sentence
  "stop_reason", when another round would only repeat directions you have
  already tried and you would rather stop than propose filler. Answer true
  when you still have an untried direction. Consult the search memory for
  earlier proposals, executable-check failures, reviewer objections, and
  D_build effects. Validation scores and gate decisions are never shown.
  When memory is disabled, do not claim to remember previous attempts.
- Every "content" is the complete new text of that file. Diffs and partial files are
  rejected. A file you do not list stays as it is.
- Each candidate targets exactly one cluster and repeats that cluster's key verbatim
  in "targeted_cluster".
- Propose at most {max_candidates} candidates; each is evaluated on its own.
  When stopping you may return an empty candidates list; do not invent filler.
- You may replace only the paths listed under "Editable paths" below. A candidate that
  touches any other path is rejected before it is evaluated.
- The five-step answer format and the output controller that emits the final
  answer are fixed. Anything listed under "Editable paths" is not."""


class DiagnosisPacket(BaseModel):
    """One round's diagnosis, assembled before any Builder call."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    selection_protocol: str = 'current'
    workflow_search: bool = False
    max_task_drop: float = 0.03
    incumbent_selection: SelectionMetrics | None = None
    stage: int = Field(ge=1)
    round_index: int = Field(ge=0)
    bundle_files: Mapping[str, str] = Field(default_factory=dict)
    build_recognition: float = Field(ge=0.0, le=1.0)
    confusion: Mapping[str, Mapping[str, int]] = Field(default_factory=dict)
    clusters: tuple[FailureCluster, ...] = ()
    keyframe_samples: tuple[tuple[str, tuple[str, ...]], ...] = ()
    correct_examples: tuple[FailureRecord, ...] = ()
    history: tuple[CandidateFeedback, ...] = ()
    tool_availability: Mapping[str, str] = Field(default_factory=dict)
    editable_paths: tuple[str, ...] = ()
    max_candidates: int = Field(default=MAX_CANDIDATES_PER_ROUND, ge=1)

    def image_paths(self) -> tuple[str, ...]:
        """Every keyframe path in packet order, for the Builder call."""
        return tuple(path for _, paths in self.keyframe_samples for path in paths)


def build_packet(
    stage: int,
    round_index: int,
    bundle_files: Mapping[str, str],
    build_recognition: float,
    records: Iterable[FailureRecord],
    limits: PacketLimits,
    confusion: Mapping[str, Mapping[str, int]] | None = None,
    max_candidates: int = MAX_CANDIDATES_PER_ROUND,
) -> DiagnosisPacket:
    """Cluster the failures and pick the keyframe samples for one round."""
    rows = tuple(records)
    clusters = cluster_failures(rows, limits.max_failures_per_cluster)
    return DiagnosisPacket(
        stage=stage,
        round_index=round_index,
        bundle_files=dict(bundle_files),
        build_recognition=build_recognition,
        confusion=confusion if confusion is not None else confusion_matrix(rows),
        clusters=clusters,
        keyframe_samples=_keyframe_samples(clusters, limits),
        editable_paths=editable_paths_for(stage, bundle_files.keys()),
        max_candidates=max_candidates,
    )


def _keyframe_samples(
    clusters: tuple[FailureCluster, ...], limits: PacketLimits
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Round-robin across clusters so no single cluster spends the frame budget."""
    picked: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    depth = max((len(cluster.samples) for cluster in clusters), default=0)
    for index in range(depth):
        for cluster in clusters:
            if len(picked) >= limits.max_keyframe_samples:
                return tuple(picked)
            record = _sample_at(cluster, index, seen)
            if record is None:
                continue
            seen.add(record.example_id)
            picked.append(
                (
                    record.example_id,
                    record.keyframe_paths[: limits.max_frames_per_sample],
                )
            )
    return tuple(picked)


def _sample_at(
    cluster: FailureCluster, index: int, seen: set[str]
) -> FailureRecord | None:
    if index >= len(cluster.samples):
        return None
    record = cluster.samples[index]
    if not record.keyframe_paths or record.example_id in seen:
        return None
    return record


def estimate_tokens(text: str) -> int:
    """Rough input-token count: four characters per token, rounded up."""
    return -(-len(text) // CHARS_PER_TOKEN)


def render_packet_text(packet: DiagnosisPacket, target_input_tokens: int) -> str:
    """Render the Builder's text input, shrinking samples until it fits the budget."""
    text = ''
    for evidence_chars, stage_chars in _TRUNCATION_STEPS:
        text = _render(packet, evidence_chars, stage_chars)
        if estimate_tokens(text) <= target_input_tokens:
            return text
    return text


def _render(
    packet: DiagnosisPacket, evidence_chars: int | None, stage_chars: int | None
) -> str:
    blocks = (
        _render_instructions(packet),
        _render_files(packet),
        _render_confusion(packet),
        _render_clusters(packet, evidence_chars, stage_chars),
        _render_correct(packet, evidence_chars, stage_chars),
        _render_history(packet),
    )
    return '\n\n'.join(blocks) + '\n'


def _render_instructions(packet: DiagnosisPacket) -> str:
    editable = '\n'.join(f'- {path}' for path in packet.editable_paths)
    accuracy = f'{packet.build_recognition:.4f}'
    instructions = _INSTRUCTIONS
    if packet.selection_protocol == 'macro_feedback':
        instructions = instructions.replace(
            '1. raise the share of construction items whose emitted label matches the gold label;\n'
            '2. keep every item the current harness already answers correctly;',
            '1. raise task-macro Recognition across ER_Lab, ER_Wild, SA on selection-dev;\n'
            f'2. allow net improvements while each task stays within {packet.max_task_drop * 100:.1f} '
            'percentage points below the ORIGINAL scaffold task score;',
        ).replace(
            'Validation scores and gate decisions are never shown.',
            'Aggregate selection-dev scores, costs, gate reasons and promotion status '
            'are shown; selection-dev is adaptive development data, not a clean test.',
        )
    blocks = [
        instructions.replace(_CANDIDATE_SLOT, str(packet.max_candidates)),
        '',
        f'Stage: {packet.stage}. Round: {packet.round_index}.',
        f'Current construction-set accuracy: {accuracy}.',
        '',
        'Editable paths:',
        editable if editable else '- (none at this stage)',
    ]
    if any(path.startswith(f'{VERIFICATION_DIR}/') for path in packet.editable_paths):
        blocks += ['', _VERIFICATION_NOTE.rstrip()]
    if packet.selection_protocol == 'macro_feedback':
        blocks += [
            '',
            'Selection-dev objective: strict positive macro gain over incumbent; '
            'tie-break fewer calls, then candidate name. Mean calls <= 4.5; '
            'proxy reasoning no more than 2 points below incumbent. '
            'D_build diagnostics remain pooled and are not the selection objective.',
        ]
        if packet.incumbent_selection is not None:
            blocks += [
                'Current selection-dev aggregate: '
                + packet.incumbent_selection.model_dump_json()
            ]
    if packet.workflow_search:
        blocks += ['', _WORKFLOW_NOTE]
    if packet.tool_availability:
        blocks += ['', 'Available tools (measured on D_build):']
        blocks += [
            f'- {name}: {status}'
            for name, status in sorted(packet.tool_availability.items())
        ]
    return '\n'.join(blocks)


def _render_files(packet: DiagnosisPacket) -> str:
    parts = ['## Current harness files']
    for path, content in sorted(packet.bundle_files.items()):
        parts.append(f'### {path}\n```\n{content}\n```')
    return '\n\n'.join(parts)


def _render_confusion(packet: DiagnosisPacket) -> str:
    columns = sorted({key for row in packet.confusion.values() for key in row})
    header = '| gold vs predicted | ' + ' | '.join(columns) + ' |'
    divider = '| --- ' * (len(columns) + 1) + '|'
    rows = [
        '| '
        + ' | '.join([gold, *(str(packet.confusion[gold].get(c, 0)) for c in columns)])
        + ' |'
        for gold in sorted(packet.confusion)
    ]
    body = '\n'.join([header, divider, *rows]) if columns else '(no failures recorded)'
    return '## Failure counts by gold label and emitted label\n\n' + body


def _render_clusters(
    packet: DiagnosisPacket, evidence_chars: int | None, stage_chars: int | None
) -> str:
    phases = Counter(
        record.failure_stage
        for cluster in packet.clusters
        for record in cluster.samples
    )
    parts = [
        '## Failure clusters',
        f'Sampled failure stages (not full-corpus counts): {dict(phases)}',
    ]
    for index, cluster in enumerate(packet.clusters, start=1):
        parts.append(_render_cluster(index, cluster, evidence_chars, stage_chars))
    if not packet.clusters:
        parts.append('(no failures recorded)')
    return '\n\n'.join(parts)


def _render_cluster(
    index: int,
    cluster: FailureCluster,
    evidence_chars: int | None,
    stage_chars: int | None,
) -> str:
    head = '\n'.join(
        (
            f'### Cluster {index}: {cluster.key}',
            f'- gold: {cluster.gold}',
            f'- emitted: {cluster.predicted}',
            f'- evidence disagreement: {"yes" if cluster.conflict else "no"}',
            f'- items: {cluster.count}',
        )
    )
    samples = [_render_sample(r, evidence_chars, stage_chars) for r in cluster.samples]
    return '\n\n'.join([head, *[s for s in samples if s]])


def _render_sample(
    record: FailureRecord, evidence_chars: int | None, stage_chars: int | None
) -> str:
    lines = [
        f'#### Item {record.example_id}',
        f'- gold: {", ".join(record.gold)}',
        f'- trace symptom: {record.failure_stage}',
    ]
    evidence = truncate(record.evidence_json, evidence_chars, TRUNCATION_MARKER)
    if evidence:
        lines.append(f'- evidence: {evidence}')
    for name, value in sorted(record.stage_outputs.items()):
        rendered = truncate(value, stage_chars, TRUNCATION_MARKER)
        if rendered:
            lines.append(f'- {name}: {rendered}')
    return '\n'.join(lines)


#: Appended to the instructions only when a ``verification/*.py`` path is
#: editable, i.e. at Stage 3. Stage 3's first replicate (2026-09-12) proposed 31
#: candidates and edited a prompt every time, never once the two functions the
#: stage had unlocked, so the packet was not telling it what they are. This
#: block describes their contract and their effect and stops there: which rule
#: to change is the Builder's to decide, and naming a direction here would be
#: doing its work for it.
_VERIFICATION_NOTE: Final = """## The two verification functions

Both are pure Python, read as text and run on the stage's JSON, so editing them
changes how a decision is made rather than what the model is asked. They issue
no model calls.

- `verification/cross_modal.py` exports `modalities_agree(record, hypotheses)
  -> bool`. It decides whether what was seen and what was heard point at the
  same label. Its answer becomes `modalities_agree` in the verification record,
  which verify and synthesize can read through `{verification_json}`.
  `routing.yaml` sets `verify_on: always` or `verify_on: conflict`; in the
  latter mode this check controls whether the student verify call runs.
  A failing check always triggers verification.
- `verification/alternative_hypotheses.py` exports `rank(record, hypotheses)
  -> list`. It orders the hypotheses; the output controller emits its labels
  from the front of that order after filtering unsupported readings and
  applying the requested cardinality. The controller never re-sorts by confidence.
  Only distinct observed evidence ids count; t1 is task context, not evidence.

A replacement must pass a static check: no imports beyond `__future__` and
`typing`, no attribute access, no loop other than `for`, no comprehension other
than the ones already there, calls only to Python builtins from a closed list
(`len`, `sorted`, `sum`, `max`, `min`, `str`, `int`, `float`, `bool`, `list`,
`dict`, `set`, `abs`, `all`, `any`, `enumerate`, `range`, `round`) or to
functions you define in the same file, at most 20,000 bytes, and no branch keyed
on a particular label string. Keep the exported name and signature.
"""

#: What counts as one identifier in a packet: an id is present only when it
#: appears as a whole token, so ``mme_er_wild_226`` is not "in" ``mme_er_wild_2260``.
_TOKEN: Final = re.compile(r'[A-Za-z0-9_\-]+')


def assert_no_forbidden_ids(text: str, forbidden_ids: Iterable[str]) -> None:
    """Raise if any benchmark id reached the Builder's input.

    Ids are matched as whole tokens. An id that is not itself one token (a file
    name with its suffix) is matched as a substring instead.
    """
    forbidden = {item for item in forbidden_ids if item}
    if not forbidden:
        return
    tokens = set(_TOKEN.findall(text))
    hits = {item for item in forbidden if _TOKEN.fullmatch(item) and item in tokens}
    hits |= {item for item in forbidden if not _TOKEN.fullmatch(item) and item in text}
    if hits:
        raise LeakageError(
            'benchmark ids present in builder input: ' + ', '.join(sorted(hits)[:10])
        )


def _render_correct(
    packet: DiagnosisPacket, evidence_chars: int | None, stage_chars: int | None
) -> str:
    if not packet.correct_examples:
        return ''
    parts = [
        '## Correct D_build examples to preserve',
        'These are boundary examples, not additional failures.',
    ]
    parts += [
        _render_sample(r, evidence_chars, stage_chars) for r in packet.correct_examples
    ]
    return '\n\n'.join(parts)


def _render_history(packet: DiagnosisPacket) -> str:
    if not packet.history:
        return ''
    parts = [
        '## Search memory (D_build and review only)',
        'The current files are authoritative. No validation results are included.',
    ]
    if packet.selection_protocol == 'macro_feedback':
        parts = [
            '## Search memory (proposal-dev and aggregate selection-dev)',
            'Selection feedback is adaptive development data, never final evaluation.',
        ]
    for entry in packet.history:
        parts.append(
            entry.model_dump_json(
                exclude={'selection'}
                if packet.selection_protocol == 'current'
                else None
            )
        )
    return '\n\n'.join(parts)


_WORKFLOW_NOTE = """Workflow search (narrow tested switches, not arbitrary graph code):
- routing.yaml workflow: multistep (default) or direct_same_tools.
- direct_same_tools uses one nominal student call (normal parse retry applies),
  the original video plus the SAME cached transcript/prosody evidence. It reuses
  prompts/generate_hypotheses.md, followed by a fixed JSON output contract asking
  for visual cues, hypotheses and four grounded step texts. It does not call the
  extract_visual, verify or synthesize prompts. The final controller stays fixed.
- multistep retains the four-stage pipeline; verify_on: always or conflict.
- routing.yaml rank_policy: bundle (default), confidence, evidence_weighted.
  confidence orders by descending confidence; evidence_weighted orders by
  confidence + weighted supporting cues - weighted contradicting cues.
  audio_weight and visual_weight are each in [0, 2], default 1.0. These tested
  helpers do not use task names, example IDs or label priors. Bundle Python code
  remains under the original static allowlist. No new model or network tool.
- Keep stages containing generate_hypotheses for direct_same_tools. All original
  prompt files remain required; change only files needed for your candidate.
"""
