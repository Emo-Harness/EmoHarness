"""The static allowlist that guards Builder-written verification code."""

from __future__ import annotations

import pytest

from emoharness.common.errors import PatchRejectedError
from emoharness.harness.allowlist import (
    MAX_VERIFICATION_SOURCE_BYTES,
    _restricted_import,
    check_verification_source,
    load_verification_function,
)
from emoharness.harness.loader import HarnessBundle

SAFE_SOURCE = """
from __future__ import annotations

from typing import Any


def rank(record: dict[str, Any], hypotheses: dict[str, Any]) -> list[Any]:
    rows = []
    for item in hypotheses['hypotheses']:
        rows = rows + [(-float(item['confidence']), str(item['label']))]
    return [row[1] for row in sorted(rows)]
"""

REJECTED_SOURCES = {
    'plain import': 'import os\n',
    'foreign import': 'from pathlib import Path\n',
    'relative import': 'from . import sibling\n',
    'attribute access': 'def f(x):\n    return x.upper()\n',
    'open': 'def f(p):\n    return open(p)\n',
    'eval': 'def f(s):\n    return eval(s)\n',
    'while loop': 'def f():\n    while True:\n        return 1\n',
    'lambda': 'f = lambda x: x\n',
    'dunder': "def f():\n    return __import__('os')\n",
    'class': 'class C:\n    pass\n',
    'try': 'def f():\n    try:\n        return 1\n    except Exception:\n        return 2\n',
    'unknown call': 'def f():\n    return print(1)\n',
    'with block': 'def f(p):\n    with p as handle:\n        return handle\n',
}


def test_stage_one_bundle_passes_the_allowlist(bundle: HarnessBundle) -> None:
    for name, source in bundle.verification_sources.items():
        assert check_verification_source(source, f'{name}.py') == ()


def test_safe_source_passes() -> None:
    assert check_verification_source(SAFE_SOURCE, 'safe.py') == ()


@pytest.mark.parametrize('label', sorted(REJECTED_SOURCES))
def test_rejected_constructs(label: str) -> None:
    violations = check_verification_source(REJECTED_SOURCES[label], 'patch.py')
    assert violations, label
    assert all(item.startswith('patch.py:') for item in violations)


def test_syntax_error_is_a_violation() -> None:
    violations = check_verification_source('def f(:\n', 'patch.py')
    assert len(violations) == 1
    assert 'syntax error' in violations[0]


def test_oversized_source_is_rejected() -> None:
    source = '# ' + 'x' * MAX_VERIFICATION_SOURCE_BYTES + '\n'
    violations = check_verification_source(source, 'patch.py')
    assert 'exceeds' in violations[0]


def test_load_runs_a_checked_function() -> None:
    rank = load_verification_function(SAFE_SOURCE, 'rank', 'safe.py')
    payload = {
        'hypotheses': [
            {'label': 'a', 'confidence': 0.1},
            {'label': 'b', 'confidence': 0.9},
        ]
    }
    assert rank({}, payload) == ['b', 'a']


def test_load_rejects_violating_source() -> None:
    with pytest.raises(PatchRejectedError) as excinfo:
        load_verification_function('import os\n', 'rank', 'patch.py')
    assert 'plain import' in str(excinfo.value)


def test_load_rejects_missing_function() -> None:
    with pytest.raises(PatchRejectedError, match='no callable'):
        load_verification_function(SAFE_SOURCE, 'modalities_agree', 'safe.py')


def test_loaded_code_has_no_ambient_builtins() -> None:
    source = 'def f():\n    return sorted([2, 1])\n'
    assert load_verification_function(source, 'f', 'safe.py')() == [1, 2]
    escaped = load_verification_function(
        'def f():\n    return undefined_name\n', 'f', 'safe.py'
    )
    with pytest.raises(NameError):
        escaped()


def test_import_shim_allows_only_typing_and_future() -> None:
    assert _restricted_import('typing') is not None
    with pytest.raises(PatchRejectedError):
        _restricted_import('os')
    with pytest.raises(PatchRejectedError):
        _restricted_import('typing', level=1)
