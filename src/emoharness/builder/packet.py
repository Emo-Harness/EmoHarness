"""The diagnosis packet: what the Builder sees in one round (ADR-0006 §3).

Text first. The packet carries the harness files verbatim, the confusion matrix and
cluster counts measured on the construction set, and a bounded number of failure
records; only a sampled subset carries keyframes, and videos are never sent.

The wording never names a held-out capability (ADR-0005, ADR-0012): the Builder is
told the seen request wording and nothing else about the benchmark.
"""

from __future__ import annotations

import re
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
from emoharness.common.constants import MAX_CANDIDATES_PER_ROUND
from emoharness.common.errors import LeakageError
from emoharness.common.text import truncate

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
  {"summary": "...", "candidates": [{"name": "...", "targeted_cluster": "...",
  "rationale": "...", "files": [{"path": "...", "content": "..."}]}]}
- Every "content" is the complete new text of that file. Diffs and partial files are
  rejected. A file you do not list stays as it is.
- Each candidate targets exactly one cluster and repeats that cluster's key verbatim
  in "targeted_cluster".
- Propose at most {max_candidates} candidates; each is evaluated on its own.
- You may replace only the paths listed under "Editable paths" below. A candidate that
  touches any other path is rejected before it is evaluated.
- The five-step answer format and the code that emits the final answer are fixed."""


class DiagnosisPacket(BaseModel):
    """One round's diagnosis, assembled before any Builder call."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stage: int = Field(ge=1)
    round_index: int = Field(ge=0)
    bundle_files: Mapping[str, str] = Field(default_factory=dict)
    build_recognition: float = Field(ge=0.0, le=1.0)
    confusion: Mapping[str, Mapping[str, int]] = Field(default_factory=dict)
    clusters: tuple[FailureCluster, ...] = ()
    keyframe_samples: tuple[tuple[str, tuple[str, ...]], ...] = ()
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
    )
    return '\n\n'.join(blocks) + '\n'


def _render_instructions(packet: DiagnosisPacket) -> str:
    editable = '\n'.join(f'- {path}' for path in packet.editable_paths)
    accuracy = f'{packet.build_recognition:.4f}'
    return '\n'.join(
        (
            _INSTRUCTIONS.replace(_CANDIDATE_SLOT, str(packet.max_candidates)),
            '',
            f'Stage: {packet.stage}. Round: {packet.round_index}.',
            f'Current construction-set accuracy: {accuracy}.',
            '',
            'Editable paths:',
            editable if editable else '- (none at this stage)',
        )
    )


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
    parts = ['## Failure clusters']
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
    lines = [f'#### Item {record.example_id}', f'- gold: {", ".join(record.gold)}']
    evidence = truncate(record.evidence_json, evidence_chars, TRUNCATION_MARKER)
    if evidence:
        lines.append(f'- evidence: {evidence}')
    for name, value in sorted(record.stage_outputs.items()):
        rendered = truncate(value, stage_chars, TRUNCATION_MARKER)
        if rendered:
            lines.append(f'- {name}: {rendered}')
    return '\n'.join(lines)


#: What counts as one identifier in a packet: an id is present only when it
#: appears as a whole token, so ``mme_er_wild_226`` is not "in" ``mme_er_wild_2260``.
_TOKEN: Final = re.compile(r'[A-Za-z0-9_\-]+')


def assert_no_forbidden_ids(text: str, forbidden_ids: Iterable[str]) -> None:
    """Raise if any benchmark id reached the Builder's input (ADR-0012 §firewall).

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
