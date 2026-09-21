"""The per-round verifier: the Builder's model reviewing the Builder's patches.

One call per round (ADR-0006 §8, added 2026-09-11). The verifier sees the
current harness files, the failure clusters by key and size, and every
candidate as a rationale plus a unified diff. It answers with one verdict per
candidate; a candidate it does not approve is never applied or evaluated, and
its reasons are written into the round's provenance. The verifier is a check
on the proposal's form, not a second optimiser: it never sees D_val, scores,
or keyframes, and it cannot edit anything.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from emoharness.builder.client import (
    DEFAULT_BACKOFF_S,
    DEFAULT_MAX_RETRIES,
    GeminiChat,
    OpenAIChat,
)
from emoharness.builder.config import BuilderConfig
from emoharness.builder.schema import BuilderProposal, CandidatePatch
from emoharness.common.errors import ValidationError
from emoharness.common.jsonish import extract_json_object

VERIFIER_ROLE: Final = 'verifier'
_NO_VERDICT: Final = 'the verifier returned no verdict for this candidate'
_DIFF_CONTEXT_LINES: Final = 3

_INSTRUCTIONS: Final = """You review patches that another model proposed for a harness around a frozen
video model. You are the check before anything is evaluated: a candidate you
approve is scored on held-back data; one you reject is discarded with your
reasons on the record. You cannot edit anything.

The harness answers requests of the form "As an emotion recognition expert,
identify the most prominent emotion displayed in the video. Candidate labels:
... ." The same wording occurs with "sentiment" in place of "emotion". It runs
four prompts (extract_visual, generate_hypotheses, verify, synthesize) whose JSON
answers are parsed by fixed code. The five-step answer format and the final
answer line are produced by code the proposer cannot change. Evidence ids are
v1.. (what was seen), a1.. (transcript sentences, then prosody phrases), and t1
(the wording of the question); nothing else exists to cite.

Reject a candidate when any of the following holds; otherwise approve it.

1. Overfitting to the failures shown: a rule keyed to a specific clip,
   transcript wording, name, scene, or to the particular items of the failure
   clusters, rather than a general reading rule that would transfer to unseen
   clips.
2. Label forcing: a default, tie-break, or bias toward a particular label; a
   rule to prefer or suppress a label independent of the evidence; wording that
   tells the model which label to answer or how often a label should win.
3. Contract damage: removing or renaming a required placeholder ({noun},
   {label_set}, {evidence_json}, {hypotheses_json}, {audio_text}, {step_titles}),
   changing the JSON keys the code parses (face, body, scene, temporal_change;
   coarse, hypotheses, label, support, contradiction, confidence;
   modalities_agree; steps, justification), asking for prose instead of JSON,
   or changing how many strings `steps` carries.
4. Evidence discipline: telling the model to infer cues that were not observed,
   to cite ids that are not in the record, to treat the wording of the question
   as evidence for the answer, or to describe sound or emotion in the visual
   stage, which must report observations only.
5. Scope: a file outside the editable list, a change the rationale does not
   describe, or a rationale that misstates what the diff does.
6. Configuration damage: a YAML edit that names a stage, module, or key the
   harness does not have, or drops a required stage.

Do not reject for style, length, tone, or because you would have written the
change differently. A change that is a plausible general improvement passes,
even if you doubt it will help.

Reply with one JSON object and nothing else, of exactly this shape:
  {"verdicts": [{"name": "...", "approved": true, "issues": ["..."],
  "notes": "..."}], "summary": "..."}
Give exactly one verdict per candidate, in the order the candidates are listed,
with "name" copied verbatim from that candidate's "name:" line (the name only,
not the heading number). "issues" lists the numbered rule and the offending
text for a rejection and stays empty for an approval."""


class Verdict(BaseModel):
    """What the verifier concluded about one candidate."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(min_length=1)
    approved: bool
    issues: tuple[str, ...] = ()
    notes: str = ''


class VerificationReport(BaseModel):
    """One round's verdicts, in the verifier's own order."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    verdicts: tuple[Verdict, ...] = ()
    summary: str = ''

    def verdict_for(
        self, name: str, position: int | None = None, total: int | None = None
    ) -> Verdict:
        """The verdict for ``name``; a candidate the verifier skipped is not approved.

        Names are compared after case-folding and collapsing punctuation, and a
        leading heading number ("Candidate 2: ...") is ignored, so a verifier
        that writes ``promote-smiles`` for ``Promote smiles`` still counts. When
        no name matches but the verifier answered exactly one verdict per
        candidate, the verdict at the same position is taken: the prompt asks
        for them in order. Anything looser would let one verdict cover two
        candidates.
        """
        wanted = _name_key(name)
        for verdict in self.verdicts:
            key = _name_key(verdict.name)
            if key == wanted or _ORDINAL.sub('', key) == wanted:
                return verdict
        if position is not None and total is not None and total == len(self.verdicts):
            return self.verdicts[position]
        return Verdict(name=name, approved=False, issues=(_NO_VERDICT,))


_ORDINAL: Final = re.compile(r'^candidate\d+')


def _name_key(name: str) -> str:
    return ''.join(char for char in name.casefold() if char.isalnum())


class ReviewRequest(BaseModel):
    """Everything the verifier is shown for one round; recorded in provenance."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    workflow_search: bool = False
    stage: int = Field(ge=1)
    round_index: int = Field(ge=0)
    bundle_files: Mapping[str, str] = Field(default_factory=dict)
    editable_paths: tuple[str, ...] = ()
    clusters: tuple[str, ...] = Field(
        default=(), description='"<cluster key>: <count> items" per cluster'
    )
    candidates: tuple[CandidatePatch, ...] = ()


@runtime_checkable
class Verifier(Protocol):
    """Anything that turns one review request into one report."""

    @property
    def model_id(self) -> str: ...

    def review(self, request: ReviewRequest) -> VerificationReport: ...


class FakeVerifier:
    """Approves everything, or answers from a canned list, and records each request."""

    def __init__(
        self,
        reports: Sequence[VerificationReport] = (),
        model_id: str = 'fake-verifier',
    ) -> None:
        self._reports = tuple(reports)
        self._model_id = model_id
        self._requests: list[ReviewRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def requests(self) -> tuple[ReviewRequest, ...]:
        return tuple(self._requests)

    def review(self, request: ReviewRequest) -> VerificationReport:
        index = len(self._requests)
        self._requests.append(request)
        if index < len(self._reports):
            return self._reports[index]
        return VerificationReport(
            verdicts=tuple(
                Verdict(name=candidate.name, approved=True)
                for candidate in request.candidates
            ),
            summary='approved without review (fake verifier)',
        )


def review_request(
    stage: int,
    round_index: int,
    bundle_files: Mapping[str, str],
    editable_paths: Sequence[str],
    clusters: Sequence[tuple[str, int]],
    proposal: BuilderProposal,
    candidates: Sequence[CandidatePatch] | None = None,
) -> ReviewRequest:
    """Assemble the request for one round; ``candidates`` defaults to the proposal's."""
    chosen = tuple(candidates) if candidates is not None else proposal.candidates
    return ReviewRequest(
        stage=stage,
        round_index=round_index,
        bundle_files=dict(bundle_files),
        editable_paths=tuple(editable_paths),
        clusters=tuple(f'{key}: {count} items' for key, count in clusters),
        candidates=chosen,
    )


def render_review_text(request: ReviewRequest) -> str:
    """The verifier's text input: rules, current files, clusters, and each diff."""
    instructions = _INSTRUCTIONS
    if request.workflow_search:
        instructions += """

Authorized workflow-search extension (these keys are supported):
The normal four-stage workflow remains available. routing.yaml may set workflow
multistep or direct_same_tools, rank_policy bundle/confidence/evidence_weighted,
and audio_weight/visual_weight in [0,2]. verify_on always/conflict remains valid.
direct_same_tools intentionally uses only generate_hypotheses as one video-plus-
audio request; code appends a fixed combined JSON contract including visual cues,
hypotheses and four grounded steps. Other prompts are not called on this path.
It is not contract damage to select this workflow or propose joint observation
and reasoning in its generate_hypotheses prompt. All prompt files/placeholders,
canonical labels, grounding, and the final five-step controller remain required.
No external model, free-form graph or unrestricted Python interface is added.
"""
    blocks = [
        instructions,
        f'Stage: {request.stage}. Round: {request.round_index}.',
        'Editable paths:\n'
        + ('\n'.join(f'- {p}' for p in request.editable_paths) or '- (none)'),
        '## Failure clusters the proposer was shown\n'
        + ('\n'.join(f'- {row}' for row in request.clusters) or '(none)'),
        _render_files(request.bundle_files),
        _render_candidates(request),
    ]
    return '\n\n'.join(blocks) + '\n'


def _render_files(files: Mapping[str, str]) -> str:
    parts = ['## Current harness files']
    for path, content in sorted(files.items()):
        parts.append(f'### {path}\n```\n{content}\n```')
    return '\n\n'.join(parts)


def _render_candidates(request: ReviewRequest) -> str:
    parts = ['## Candidates']
    for index, candidate in enumerate(request.candidates, start=1):
        parts.append(_render_candidate(index, candidate, request.bundle_files))
    if not request.candidates:
        parts.append('(none)')
    return '\n\n'.join(parts)


def _render_candidate(
    index: int, candidate: CandidatePatch, files: Mapping[str, str]
) -> str:
    lines = [
        f'### Candidate {index}',
        f'- name: {candidate.name}',
        f'- targeted_cluster: {candidate.targeted_cluster}',
        f'- rationale: {candidate.rationale}',
        f'- files: {", ".join(file.path for file in candidate.files)}',
    ]
    for file in candidate.files:
        lines.append(
            f'#### diff of {file.path}\n```\n{_diff(files, file.path, file.content)}```'
        )
    return '\n'.join(lines)


def _diff(files: Mapping[str, str], path: str, new_content: str) -> str:
    old = files.get(path, '')
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f'{path} (current)',
        tofile=f'{path} (proposed)',
        n=_DIFF_CONTEXT_LINES,
    )
    text = ''.join(diff)
    if not text:
        return '(no textual change)\n'
    return text if text.endswith('\n') else text + '\n'


def parse_report(text: str) -> VerificationReport:
    """Parse a verifier response, tolerating code fences and leading prose."""
    payload = extract_json_object(text)
    if payload is None:
        raise ValidationError('verifier response contains no JSON object')
    try:
        return VerificationReport.model_validate(_coerced(payload))
    except PydanticValidationError as exc:
        raise ValidationError(
            f'verifier response does not match the schema: {exc}'
        ) from exc


def _coerced(payload: dict[str, Any]) -> dict[str, Any]:
    """Verdict fields the model tends to bend: issues as a string, notes missing."""
    verdicts = payload.get('verdicts')
    if not isinstance(verdicts, list):
        return payload
    fixed = []
    for verdict in verdicts:
        if isinstance(verdict, dict):
            issues = verdict.get('issues', ())
            if isinstance(issues, str):
                issues = [issues] if issues.strip() else []
            verdict = {
                **verdict,
                'issues': [str(item) for item in issues] if issues else [],
                'notes': str(verdict.get('notes') or ''),
            }
        fixed.append(verdict)
    return {**payload, 'verdicts': fixed}


class OpenAIVerifier:
    """The Builder's OpenAI model, asked to review, billed as ``verifier``."""

    def __init__(
        self,
        config: BuilderConfig,
        ledger_path: Path,
        experiment: str,
        client: httpx.Client | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
    ) -> None:
        self._chat = OpenAIChat(
            config,
            ledger_path,
            experiment,
            VERIFIER_ROLE,
            client=client,
            max_retries=max_retries,
            backoff_s=backoff_s,
            reasoning_effort=config.verifier.reasoning_effort,
        )

    @property
    def model_id(self) -> str:
        return self._chat.model_id

    def review(self, request: ReviewRequest) -> VerificationReport:
        return parse_report(self._chat.complete(render_review_text(request)))


class GeminiVerifier:
    """The Builder's Gemini model, asked to review, billed as ``verifier``."""

    def __init__(
        self, config: BuilderConfig, ledger_path: Path, experiment: str
    ) -> None:
        self._chat = GeminiChat(config, ledger_path, experiment, VERIFIER_ROLE)

    @property
    def model_id(self) -> str:
        return self._chat.model_id

    def review(self, request: ReviewRequest) -> VerificationReport:
        return parse_report(self._chat.complete(render_review_text(request)))


def verifier_for(
    config: BuilderConfig, ledger_path: Path, experiment: str
) -> Verifier | None:
    """The configured verifier, or None when the config switches it off."""
    if not config.verifier.enabled:
        return None
    if config.provider == 'gemini':
        return GeminiVerifier(config, ledger_path, experiment)
    return OpenAIVerifier(config, ledger_path, experiment)


__all__ = [
    'VERIFIER_ROLE',
    'FakeVerifier',
    'GeminiVerifier',
    'OpenAIVerifier',
    'ReviewRequest',
    'VerificationReport',
    'Verdict',
    'Verifier',
    'parse_report',
    'render_review_text',
    'review_request',
    'verifier_for',
]
