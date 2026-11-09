"""Reading JSON out of an untrusted answer: the table below is the specification.

Both the harness stages and the Builder client depend on exactly this behaviour,
so the shapes that count as a parse are pinned here rather than at either caller.
"""

from __future__ import annotations

import pytest

from emoharness.common.jsonish import extract_json_object, strip_code_fences


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('{"a": 1}', {'a': 1}),
        ('```json\n{"a": 1}\n```', {'a': 1}),
        ('```\n{"a": 1}\n```', {'a': 1}),
        ('Here you go: {"a": 1} - hope that helps.', {'a': 1}),
        ('  \n{"a": 1}\n  ', {'a': 1}),
        ('no json here', None),
        ('[1, 2, 3]', None),
        ('{not json}', None),
        ('}{', None),
        ('', None),
    ],
)
def test_extract_json_object(text: str, expected: dict[str, int] | None) -> None:
    assert extract_json_object(text) == expected


def test_extract_json_object_takes_the_outermost_braces() -> None:
    assert extract_json_object('{"a": {"b": 1}} trailing') == {'a': {'b': 1}}


def test_strip_code_fences_leaves_plain_text() -> None:
    assert strip_code_fences('plain') == 'plain'


def test_strip_code_fences_keeps_the_first_block() -> None:
    assert strip_code_fences('a\n```json\nfirst\n```\n```\nsecond\n```') == 'first\n'
