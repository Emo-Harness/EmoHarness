"""Output controller: what is emitted, in what shape.

Implementation detail of the harness; ``HarnessPipeline.run`` is the only caller.
Human-written and frozen for the Builder. Every rule here is derived from the
public prompt grammar, never from a task name or a label string: cardinality comes
from the quantifier, the noun is copied from the prompt, and labels are copied
verbatim from the candidate list. Steps that cite nothing real are removed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, NamedTuple

from emoharness.common.constants import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    LABEL_SEPARATOR,
    MULTILABEL_CAP,
    NUM_STEPS,
    STEP_PREFIX,
    STEP_TITLES,
)
from emoharness.data.label_set import LabelSet
from emoharness.data.prompt_template import ParsedPrompt
from emoharness.harness._text import split_sentences
from emoharness.harness.schema import (
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    StageTrace,
    VerificationResult,
)

#: Steps the Target writes; the last one is the controller's fixed decision.
MODEL_WRITTEN_STEPS: Final = NUM_STEPS - 1

_DECISION_SINGULAR: Final = 'Based on the analysis, the predicted {noun} is {labels}.'
_DECISION_PLURAL: Final = 'Based on the analysis, the predicted {noun} are {labels}.'
_NO_CUE_TEXT: Final = 'No reliable cue observed.'
_TITLE_SEPARATOR: Final = ': '

_CITATION_RE: Final = re.compile(r'\[([^\[\]]*)\]')
_SPACE_BEFORE_PUNCT_RE: Final = re.compile(r'\s+([.,;:!?])')


class _ScoredHypothesis(NamedTuple):
    """A hypothesis reduced to what the evidence gate needs."""

    label: str
    confidence: float
    support: int
    contradiction: int

    @property
    def passes(self) -> bool:
        """At least one real supporting item, and more support than contradiction."""
        return self.support >= 1 and self.support > self.contradiction


def build_output(
    example_id: str,
    parsed: ParsedPrompt,
    record: EvidenceRecord,
    hypotheses: HypothesisSet,
    verification: VerificationResult,
    step_texts: Sequence[str],
    *,
    cap: int = MULTILABEL_CAP,
    traces: Sequence[StageTrace] = (),
    fallback_used: bool = False,
) -> HarnessOutput:
    """Assemble the final record. Refusals are impossible: labels are never empty."""
    labels = _gate_labels(parsed, verification.ranked, record, cap)
    steps = _render_steps(step_texts, parsed.noun, labels, record)
    return HarnessOutput(
        example_id=example_id,
        labels=labels,
        steps=steps,
        response_text=_render_response(steps, labels),
        evidence=record,
        hypotheses=hypotheses,
        verification=verification,
        traces=tuple(traces),
        fallback_used=fallback_used,
    )


def _gate_labels(
    parsed: ParsedPrompt,
    ranked: Sequence[Hypothesis],
    record: EvidenceRecord,
    cap: int = MULTILABEL_CAP,
) -> tuple[str, ...]:
    """The labels to emit: evidence-gated, ranked, capped, and never empty."""
    label_set = LabelSet.from_parsed(parsed)
    cardinality = label_set.cardinality(cap)
    maximum = min(cap, cardinality.maximum or cap)
    scored = _score_hypotheses(ranked, record, label_set)
    # The ranker owns decision order. Re-sorting here silently disables code edits.
    passing = tuple(item for item in scored if item.passes)
    labels = label_set.canonical_many(item.label for item in passing)[:maximum]
    if len(labels) < cardinality.minimum:
        return _fill(labels, scored, label_set.labels, cardinality.minimum, maximum)
    return labels


def _score_hypotheses(
    ranked: Sequence[Hypothesis], record: EvidenceRecord, label_set: LabelSet
) -> tuple[_ScoredHypothesis, ...]:
    """Canonicalize labels and count the support and contradiction ids that exist."""
    known = record.decision_ids()
    scored = []
    for hypothesis in ranked:
        label = label_set.canonical(hypothesis.label)
        if label is None:
            continue
        scored.append(
            _ScoredHypothesis(
                label=label,
                confidence=hypothesis.confidence,
                support=_count_known(hypothesis.support, known),
                contradiction=_count_known(hypothesis.contradiction, known),
            )
        )
    return tuple(scored)


def _render_steps(
    step_texts: Sequence[str],
    noun: str,
    labels: Sequence[str],
    record: EvidenceRecord,
) -> tuple[str, ...]:
    """Exactly ``NUM_STEPS`` steps: four grounded ones plus the fixed decision."""
    known = record.decision_ids()
    bodies = tuple(
        _grounded(_text_at(step_texts, index), known)
        for index in range(MODEL_WRITTEN_STEPS)
    )
    titled = tuple(
        f'{title}{_TITLE_SEPARATOR}{body}'
        for title, body in zip(STEP_TITLES[:MODEL_WRITTEN_STEPS], bodies, strict=True)
    )
    return titled + (_decision_sentence(noun, labels),)


def _decision_sentence(noun: str, labels: Sequence[str]) -> str:
    """The final step, whose wording the Builder may not change."""
    template = _DECISION_SINGULAR if len(labels) == 1 else _DECISION_PLURAL
    return template.format(
        noun=_noun_form(noun, len(labels)), labels=LABEL_SEPARATOR.join(labels)
    )


def _render_response(steps: Sequence[str], labels: Sequence[str]) -> str:
    """The text the official judge receives: numbered steps then the answer tag."""
    lines = tuple(
        f'{STEP_PREFIX} {number}{_TITLE_SEPARATOR}{step}'
        for number, step in enumerate(steps, start=1)
    )
    answer = ANSWER_OPEN + LABEL_SEPARATOR.join(labels) + ANSWER_CLOSE
    return '\n'.join(lines + (answer,))


def _count_known(ids: Sequence[str], known: frozenset[str]) -> int:
    return len(set(ids) & known)


def _fill(
    labels: tuple[str, ...],
    scored: Sequence[_ScoredHypothesis],
    label_set: Sequence[str],
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    """Top up an under-full answer from the ranked list, then from the candidates."""
    filled = list(labels)
    for candidate in tuple(item.label for item in scored) + tuple(label_set):
        if len(filled) >= minimum or len(filled) >= maximum:
            break
        if candidate not in filled:
            filled.append(candidate)
    return tuple(filled)


def _text_at(texts: Sequence[str], index: int) -> str:
    return texts[index] if index < len(texts) else ''


def _grounded(text: str, known: frozenset[str]) -> str:
    """Drop unknown ids, then drop every sentence left without a citation."""
    cleaned = _CITATION_RE.sub(lambda match: _keep_ids(match.group(1), known), text)
    kept = tuple(
        _squeeze(sentence)
        for sentence in split_sentences(cleaned)
        if _CITATION_RE.search(sentence)
    )
    return ' '.join(part for part in kept if part) or _NO_CUE_TEXT


def _keep_ids(raw: str, known: frozenset[str]) -> str:
    kept = tuple(part.strip() for part in raw.split(',') if part.strip() in known)
    return f'[{LABEL_SEPARATOR.join(kept)}]' if kept else ''


def _squeeze(text: str) -> str:
    return _SPACE_BEFORE_PUNCT_RE.sub(r'\1', ' '.join(text.split()))


def _noun_form(noun: str, count: int) -> str:
    """The singular of a plural noun when exactly one label is emitted."""
    if count == 1 and noun.endswith('s'):
        return noun[:-1]
    return noun
