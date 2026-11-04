"""Conservative alternative ranking using observed facial anchors.

Confidence remains authoritative except for a narrow weak-evidence comparison.
No candidate label has a prior, and hypotheses are never created or relabeled.
"""

from __future__ import annotations

from typing import Any


def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    index = 0
    for hypothesis in hypothesis_list(hypotheses):
        rows = rows + [(-confidence_of(hypothesis), -support_count(record, hypothesis), index, hypothesis)]
        index = index + 1
    ordered = sorted(rows)
    baseline = ordered_hypotheses(ordered)
    if len(ordered) < 2:
        return baseline
    if not emotion_task(record):
        return baseline
    leader = ordered[0][3]
    leader_confidence = confidence_of(leader)
    if leader_confidence >= 0.8:
        return baseline
    if diagnostic_count(record, leader) > 0:
        return baseline
    for row in ordered[1:]:
        alternative = row[3]
        confidence = confidence_of(alternative)
        if confidence >= 0.3 and leader_confidence - confidence <= 0.200000001:
            if diagnostic_count(record, alternative) > 0:
                result = [alternative]
                for other in ordered:
                    if other[2] != row[2]:
                        result = result + [other[3]]
                return result
    return baseline


def ordered_hypotheses(rows: list[Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result = result + [row[3]]
    return result


def emotion_task(record: dict[str, Any]) -> bool:
    if 'text' not in record:
        return False
    text = record['text']
    if 'noun' not in text:
        return False
    return str(text['noun']) == 'emotion'


def confidence_of(hypothesis: dict[str, Any]) -> float:
    if 'confidence' in hypothesis:
        return float(hypothesis['confidence'])
    return 0.0


def observed_ids(record: dict[str, Any]) -> set[str]:
    result = set()
    for item in item_list(record):
        if 'id' in item and 'modality' in item:
            item_id = str(item['id'])
            if item_id != 't1' and item['modality'] in ('visual', 'audio'):
                result = result | {item_id}
    return result


def supporting_ids(record: dict[str, Any], hypothesis: dict[str, Any]) -> set[str]:
    known = observed_ids(record)
    result = set()
    for item_id in id_list(hypothesis, 'support'):
        key = str(item_id)
        if key in known:
            result = result | {key}
    return result


def contradicting_ids(record: dict[str, Any], hypothesis: dict[str, Any]) -> set[str]:
    known = observed_ids(record)
    result = set()
    for item_id in id_list(hypothesis, 'contradiction'):
        key = str(item_id)
        if key in known:
            result = result | {key}
    return result


def support_count(record: dict[str, Any], hypothesis: dict[str, Any]) -> int:
    return len(supporting_ids(record, hypothesis))


def face_cues(record: dict[str, Any]) -> list[Any]:
    if 'visual' not in record:
        return []
    visual = record['visual']
    if 'face' in visual:
        return list(visual['face'])
    return []


def diagnostic_count(record: dict[str, Any], hypothesis: dict[str, Any]) -> int:
    support = supporting_ids(record, hypothesis)
    contradiction = contradicting_ids(record, hypothesis)
    faces = face_cues(record)
    counted = set()
    for item in item_list(record):
        if 'id' in item and 'modality' in item and 'cue' in item:
            item_id = str(item['id'])
            cue = str(item['cue'])
            if item_id in support and item_id not in contradiction:
                if item['modality'] == 'visual' and cue in faces:
                    if diagnostic_face(cue):
                        counted = counted | {item_id}
    return len(counted)


def diagnostic_face(cue: str) -> bool:
    for absence in ('not ', 'Not ', 'no ', 'No ', 'without ', 'Without '):
        if absence in cue:
            return False
    anchors = (
        'smil', 'Smil', 'laugh', 'Laugh', 'frown', 'Frown',
        'scowl', 'Scowl', 'grimac', 'Grimac', 'tear', 'Tear',
        'cry', 'Cry', 'furrowed brow', 'Furrowed brow',
        'brows drawn', 'eyebrows drawn', 'brows furrow',
        'nose wrinkl', 'lip curl', 'lips press', 'lip press',
        'jaw tight', 'jaw clench', 'eyes wide', 'Eyes wide',
        'wide eyes', 'Wide eyes', 'wide-eyed', 'Wide-eyed'
    )
    for anchor in anchors:
        if anchor in cue:
            return True
    return False


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
