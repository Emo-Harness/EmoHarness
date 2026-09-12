"""Deterministic ranking of alternative readings.

The ranker preserves model confidence but uses the evidence modality that is
most informative for the question. Audio evidence is especially important for
an expressed reaction, while visual evidence remains useful for visible affect.
No candidate-label prior is used.
"""

from __future__ import annotations

from typing import Any


def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[dict[str, Any]]:
    """Hypotheses ordered from strongest to weakest."""
    noun = question_noun(record)
    rows = []
    index = 0
    for hypothesis in hypothesis_list(hypotheses):
        confidence = confidence_of(hypothesis)
        evidence_score = weighted_support(record, hypothesis, noun)
        contradiction = contradiction_count(record, hypothesis)
        score = confidence + evidence_score - contradiction * 0.03
        rows = rows + [(-score, -confidence, index, hypothesis)]
        index = index + 1
    ordered = sorted(rows)
    return [row[3] for row in ordered]


def question_noun(record: dict[str, Any]) -> str:
    """Return the question noun when it is available."""
    if 'text' in record and 'noun' in record['text']:
        return str(record['text']['noun'])
    return ''


def confidence_of(hypothesis: dict[str, Any]) -> float:
    """The stated confidence, or zero when it is absent."""
    if 'confidence' in hypothesis:
        return float(hypothesis['confidence'])
    return 0.0


def weighted_support(record: dict[str, Any], hypothesis: dict[str, Any], noun: str) -> float:
    """Give direct modality evidence a modest, question-aware weight."""
    score = 0.0
    for item_id in id_list(hypothesis, 'support'):
        item = item_for_id(record, str(item_id))
        if len(item) == 0:
            continue
        modality = str(item['modality'])
        if modality == 'audio':
            if noun == 'sentiment':
                score = score + 0.30
            else:
                score = score + 0.35
        elif modality == 'visual':
            if noun == 'sentiment':
                score = score + 0.06
            else:
                score = score + 0.04
        else:
            score = score + 0.01
    return score


def contradiction_count(record: dict[str, Any], hypothesis: dict[str, Any]) -> int:
    """Count only contradictions that cite real evidence items."""
    known = known_ids(record)
    count = 0
    for item_id in id_list(hypothesis, 'contradiction'):
        if str(item_id) in known:
            count = count + 1
    return count


def item_for_id(record: dict[str, Any], item_id: str) -> dict[str, Any]:
    """Find one citable item, or return an empty mapping."""
    for item in item_list(record):
        if str(item['id']) == item_id:
            return item
    return {}


def known_ids(record: dict[str, Any]) -> set[str]:
    """Every citable id in the record."""
    return {str(item['id']) for item in item_list(record)}


def item_list(record: dict[str, Any]) -> list[Any]:
    """The record's citable items, or an empty list."""
    if 'items' in record:
        return list(record['items'])
    return []


def hypothesis_list(hypotheses: dict[str, Any]) -> list[Any]:
    """The hypothesis objects, or an empty list."""
    if 'hypotheses' in hypotheses:
        return list(hypotheses['hypotheses'])
    return []


def id_list(hypothesis: dict[str, Any], key: str) -> list[Any]:
    """One id list of a hypothesis, or an empty list."""
    if key in hypothesis:
        return list(hypothesis[key])
    return []
