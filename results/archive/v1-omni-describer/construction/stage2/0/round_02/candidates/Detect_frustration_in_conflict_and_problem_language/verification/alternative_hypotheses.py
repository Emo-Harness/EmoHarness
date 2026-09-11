"""Alternative-hypothesis ranking (ADR-0005: no calibration, no label priors).

Stage 1 rule: order by stated confidence, then by how many supporting items exist
in the record, then by the order the hypotheses arrived in.

This file is loaded through the static allowlist in
``emoharness.harness.allowlist``: no imports beyond ``__future__`` and ``typing``,
no attribute access, no loops other than ``for``, and calls only to whitelisted
builtins or to functions defined here.
"""

from __future__ import annotations

from typing import Any


def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[dict[str, Any]]:
    """Hypotheses ordered from strongest to weakest."""
    rows = []
    index = 0
    for hypothesis in hypothesis_list(hypotheses):
        confidence = confidence_of(hypothesis)
        support = support_count(record, hypothesis)
        rows = rows + [(-confidence, -support, index, hypothesis)]
        index = index + 1
    ordered = sorted(rows)
    return [row[3] for row in ordered]


def confidence_of(hypothesis: dict[str, Any]) -> float:
    """The stated confidence, or 0.0 when there is none."""
    if 'confidence' in hypothesis:
        return float(hypothesis['confidence'])
    return 0.0


def support_count(record: dict[str, Any], hypothesis: dict[str, Any]) -> int:
    """How many supporting ids of a hypothesis exist in the record."""
    known = known_ids(record)
    count = 0
    for item_id in id_list(hypothesis, 'support'):
        if str(item_id) in known:
            count = count + 1
    return count


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
