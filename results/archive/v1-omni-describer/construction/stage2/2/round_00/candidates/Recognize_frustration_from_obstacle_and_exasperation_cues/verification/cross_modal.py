"""Cross-modal agreement flag (ADR-0005: one deterministic check, no resolution rule).

Stage 1 rule: the label best supported by seen items must equal the label best
supported by heard items. When one side supports nothing, there is nothing to
disagree with and the flag stays true.

This file is loaded through the static allowlist in
``emoharness.harness.allowlist``: no imports beyond ``__future__`` and ``typing``,
no attribute access, no loops other than ``for``, and calls only to whitelisted
builtins or to functions defined here.
"""

from __future__ import annotations

from typing import Any

SEEN_MODALITIES = ('visual',)
HEARD_MODALITIES = ('audio', 'audio_opinion')


def modalities_agree(record: dict[str, Any], hypotheses: dict[str, Any]) -> bool:
    """True when the seen cues and the heard cues favour the same label."""
    seen = top_label(record, hypotheses, SEEN_MODALITIES)
    heard = top_label(record, hypotheses, HEARD_MODALITIES)
    if seen == '' or heard == '':
        return True
    return seen == heard


def top_label(
    record: dict[str, Any],
    hypotheses: dict[str, Any],
    modalities: tuple[str, ...],
) -> str:
    """The label with the most supporting items in ``modalities``; '' when none."""
    rows = []
    index = 0
    for hypothesis in hypothesis_list(hypotheses):
        count = support_count(record, hypothesis, modalities)
        rows = rows + [(count, -index, str(hypothesis['label']))]
        index = index + 1
    ordered = sorted(rows, reverse=True)
    if len(ordered) == 0:
        return ''
    if ordered[0][0] == 0:
        return ''
    return str(ordered[0][2])


def support_count(
    record: dict[str, Any],
    hypothesis: dict[str, Any],
    modalities: tuple[str, ...],
) -> int:
    """How many of a hypothesis' supporting ids exist and sit in ``modalities``."""
    count = 0
    for item_id in id_list(hypothesis, 'support'):
        if modality_of(record, str(item_id)) in modalities:
            count = count + 1
    return count


def modality_of(record: dict[str, Any], item_id: str) -> str:
    """The modality of one evidence item, or '' when the id is unknown."""
    for item in item_list(record):
        if str(item['id']) == item_id:
            return str(item['modality'])
    return ''


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
