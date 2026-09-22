"""Cross-modal verification: the built-in rule and the bundle's replacement for it.

Implementation detail of the harness; ``HarnessPipeline.run`` is the only caller.
ADR-0005 gives the Stage 1 scaffold exactly one deterministic check, a cross-modal
agreement flag with no conflict-resolution rule. The same rule exists twice: once
here in Python, and once as a plain-dict function inside the bundle that the
Builder may rewrite at Stage 3. The bundle version wins when it runs; when it
raises, the built-in rule takes over and the reason is recorded in ``notes``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final, NamedTuple

from pydantic import ValidationError as PydanticValidationError

from emoharness.common.logging import get_logger
from emoharness.data.label_set import LabelSet
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.schema import (
    EvidenceRecord,
    Hypothesis,
    HypothesisSet,
    VerificationResult,
)

_SEEN_MODALITIES: Final[tuple[str, ...]] = ('visual',)
_HEARD_MODALITIES: Final[tuple[str, ...]] = ('audio',)

_logger = get_logger(__name__)


class _Payload(NamedTuple):
    """One item's records in both the object and the plain-dict form."""

    record: EvidenceRecord
    hypotheses: HypothesisSet
    record_json: dict[str, Any]
    hypotheses_json: dict[str, Any]


def apply_bundle_verification(
    bundle: HarnessBundle, record: EvidenceRecord, hypotheses: HypothesisSet
) -> VerificationResult:
    """Run the bundle's agreement check and ranker, falling back when they fail."""
    # Sanitize before running code, so neither the ranker nor its fallback can
    # count the question or repeated citations as additional observations.
    hypotheses = hypotheses.model_copy(
        update={
            'hypotheses': _validate_ranked(
                (item.model_dump(mode='json') for item in hypotheses.hypotheses), record
            )
        }
    )
    payload = _Payload(
        record=record,
        hypotheses=hypotheses,
        record_json=record.model_dump(mode='json'),
        hypotheses_json=hypotheses.model_dump(mode='json'),
    )
    agree, agree_note = _bundle_agreement(bundle, payload)
    ranked, rank_note = _bundle_ranking(bundle, payload)
    notes = '; '.join(note for note in (agree_note, rank_note) if note)
    return VerificationResult(modalities_agree=agree, ranked=ranked, notes=notes)


def _bundle_agreement(bundle: HarnessBundle, payload: _Payload) -> tuple[bool, str]:
    try:
        agreed = bundle.agree_fn()(payload.record_json, payload.hypotheses_json)
        if not isinstance(agreed, bool):
            raise TypeError('modalities_agree must return a bool')
        return agreed, ''
    except Exception as exc:  # the bundle is Builder-written; never trust it
        _logger.warning('bundle agreement check failed: %s', exc)
        fallback = _cross_modal_agreement(payload.record, payload.hypotheses.hypotheses)
        return fallback, f'agreement check fell back to the built-in rule ({exc})'


def _bundle_ranking(
    bundle: HarnessBundle, payload: _Payload
) -> tuple[tuple[Hypothesis, ...], str]:
    policy = bundle.routing.rank_policy
    if policy != 'bundle':
        return _policy_ranking(bundle, payload), ''
    try:
        rows = bundle.rank_fn()(payload.record_json, payload.hypotheses_json)
        ranked = _validate_ranked(rows, payload.record)
    except Exception as exc:  # the bundle is Builder-written; never trust it
        _logger.warning('bundle ranker failed: %s', exc)
        return _default_ranking(payload.hypotheses), f'ranker fell back ({exc})'
    if not ranked:
        return _default_ranking(payload.hypotheses), 'ranker returned no usable rows'
    return ranked, ''


def _cross_modal_agreement(
    record: EvidenceRecord, hypotheses: Sequence[Hypothesis]
) -> bool:
    """True when seen and heard cues favour the same label, or one side is silent."""
    seen = _top_label(hypotheses, record, _SEEN_MODALITIES)
    heard = _top_label(hypotheses, record, _HEARD_MODALITIES)
    if seen is None or heard is None:
        return True
    return seen == heard


def _validate_ranked(
    rows: Iterable[Any], record: EvidenceRecord
) -> tuple[Hypothesis, ...]:
    """Keep the rows that are hypotheses over real candidates with real ids."""
    known = record.decision_ids()
    label_set = LabelSet.from_parsed(record.text)
    valid: list[Hypothesis] = []
    for row in rows:
        hypothesis = _as_hypothesis(row)
        if hypothesis is None:
            continue
        label = label_set.canonical(hypothesis.label)
        if label is None:
            continue
        valid.append(
            hypothesis.model_copy(
                update={
                    'label': label,
                    'support': _known_ids(hypothesis.support, known),
                    'contradiction': _known_ids(hypothesis.contradiction, known),
                }
            )
        )
    return tuple(valid)


def _default_ranking(hypotheses: HypothesisSet) -> tuple[Hypothesis, ...]:
    """The built-in order: confidence first, then how much evidence was cited."""
    return tuple(
        sorted(
            hypotheses.hypotheses,
            key=lambda item: (-item.confidence, -len(item.support)),
        )
    )


def _top_label(
    hypotheses: Sequence[Hypothesis],
    record: EvidenceRecord,
    modalities: tuple[str, ...],
) -> str | None:
    rows = sorted(
        (
            (_support_count(hypothesis, record, modalities), -index, hypothesis.label)
            for index, hypothesis in enumerate(hypotheses)
        ),
        reverse=True,
    )
    if not rows or rows[0][0] == 0:
        return None
    return rows[0][2]


def _support_count(
    hypothesis: Hypothesis, record: EvidenceRecord, modalities: tuple[str, ...]
) -> int:
    return len(
        [
            item_id
            for item_id in hypothesis.support
            if _modality_of(item_id, record) in modalities
        ]
    )


def _modality_of(item_id: str, record: EvidenceRecord) -> str | None:
    """The modality of one citable item, or None when the id is unknown."""
    for item in record.items:
        if item.id == item_id:
            return item.modality
    return None


def _as_hypothesis(row: Any) -> Hypothesis | None:
    """One row of Builder-written output as a hypothesis, or None when it is not one."""
    try:
        return Hypothesis.model_validate(row)
    except PydanticValidationError:
        return None


def _known_ids(ids: Sequence[str], known: frozenset[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item_id for item_id in ids if item_id in known))


def _policy_ranking(bundle: HarnessBundle, payload: _Payload) -> tuple[Hypothesis, ...]:
    """Narrow code-owned helpers operating only on validated cue IDs."""
    modalities = {item.id: item.modality for item in payload.record.items}
    weights = {
        'audio': bundle.routing.audio_weight,
        'visual': bundle.routing.visual_weight,
    }

    def score(item: Hypothesis) -> float:
        if bundle.routing.rank_policy == 'confidence':
            return item.confidence
        support = sum(weights.get(modalities.get(i, ''), 0.0) for i in item.support)
        contradiction = sum(
            weights.get(modalities.get(i, ''), 0.0) for i in item.contradiction
        )
        return item.confidence + support - contradiction

    return tuple(sorted(payload.hypotheses.hypotheses, key=lambda item: -score(item)))
