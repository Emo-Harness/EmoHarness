"""Reading one JSON object out of untrusted model output.

Both the harness stages and the Builder client ask a model for "one JSON object
and nothing else" and get prose, code fences, or both. This is the one place
that decides what counts as a parse; what to do about a failure is the caller's.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

_FENCE_RE: Final = re.compile(r'```[a-zA-Z0-9_+-]*\s*(.*?)```', re.DOTALL)


def strip_code_fences(text: str) -> str:
    """The contents of the first fenced block, or the text unchanged."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model answer, or None when there is none.

    Fences are stripped, then everything from the first ``{`` to the last ``}``
    is parsed. Anything that does not parse, or parses as something other than
    an object, is None.
    """
    candidate = strip_code_fences(text).strip()
    start = candidate.find('{')
    end = candidate.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
