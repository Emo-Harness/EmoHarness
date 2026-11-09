"""Alternative-hypothesis ranking with conservative conflict tie-breaking."""

from __future__ import annotations

from typing import Any


def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[dict[str, Any]]:
    """Order hypotheses by confidence, with cautious audio help during conflict."""
    entries = hypothesis_list(hypotheses)
    conflict = modalities_conflict(record, entries)
    maximum = 0.0
    for hypothesis in entries:
        value = confidence_of(hypothesis)
        if value > maximum:
            maximum = value
    rows = []
    index = 0
    for hypothesis in entries:
        confidence = confidence_of(hypothesis)
        support = support_count(record, hypothesis)
        score = confidence
        if conflict and confidence >= maximum - 0.25:
            if has_modality(record, hypothesis, 'audio'):
                score = score + 0.25
            if has_modality(record, hypothesis, 'audio') and has_modality(record, hypothesis, 'visual'):
                score = score + 0.15
        rows = rows + [(-score, -support, index, hypothesis)]
        index = index + 1
    ordered = sorted(rows)
    return [row[3] for row in ordered]


def confidence_of(hypothesis: dict[str, Any]) -> float:
    if 'confidence' in hypothesis:
        return float(hypothesis['confidence'])
    return 0.0


def modalities_conflict(record: dict[str, Any], hypotheses: list[Any]) -> bool:
    seen = top_label(record, hypotheses, 'visual')
    heard = top_label(record, hypotheses, 'audio')
    if seen == '' or heard == '':
        return False
    return seen != heard


def top_label(record: dict[str, Any], hypotheses: list[Any], modality: str) -> str:
    rows = []
    index = 0
    for hypothesis in hypotheses:
        count = modality_support_count(record, hypothesis, modality)
        rows = rows + [(count, -index, str(hypothesis['label']))]
        index = index + 1
    ordered = sorted(rows, reverse=True)
    if len(ordered) == 0:
        return ''
    if ordered[0][0] == 0:
        return ''
    return str(ordered[0][2])


def modality_support_count(record: dict[str, Any], hypothesis: dict[str, Any], modality: str) -> int:
    count = 0
    used = []
    for item_id in id_list(hypothesis, 'support'):
        text_id = str(item_id)
        if text_id not in used and modality_of(record, text_id) == modality:
            used = used + [text_id]
            count = count + 1
    return count


def has_modality(record: dict[str, Any], hypothesis: dict[str, Any], modality: str) -> bool:
    for item_id in id_list(hypothesis, 'support'):
        if modality_of(record, str(item_id)) == modality:
            return True
    return False


def support_count(record: dict[str, Any], hypothesis: dict[str, Any]) -> int:
    count = 0
    known = known_ids(record)
    used = []
    for item_id in id_list(hypothesis, 'support'):
        text_id = str(item_id)
        if text_id in known and text_id not in used:
            used = used + [text_id]
            count = count + 1
    return count


def known_ids(record: dict[str, Any]) -> list[str]:
    ids = []
    for item in item_list(record):
        text_id = str(item['id'])
        if text_id not in ids:
            ids = ids + [text_id]
    return ids


def modality_of(record: dict[str, Any], item_id: str) -> str:
    for item in item_list(record):
        if str(item['id']) == item_id:
            return str(item['modality'])
    return ''


def item_list(record: dict[str, Any]) -> list[Any]:
    if 'items' in record:
        return list(record['items'])
    return []


def hypothesis_list(hypotheses: dict[str, Any]) -> list[Any]:
    if 'hypotheses' in hypotheses:
        return list(hypotheses['hypotheses'])
    return []


def id_list(hypothesis: dict[str, Any], key: str) -> list[Any]:
    if key in hypothesis:
        return list(hypothesis[key])
    return []
