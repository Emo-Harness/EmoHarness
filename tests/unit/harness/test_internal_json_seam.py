"""The one internal seam kept under direct test: the stricter retry prompt.

Every other behaviour of the harness is exercised through ``HarnessPipeline.run``.
What counts as a JSON parse is specified in tests/unit/common/test_jsonish.py.
"""

from __future__ import annotations

from emoharness.harness._stages import STRICT_JSON_REMINDER, with_reminder
from emoharness.target.base import TargetRequest

REQUEST = TargetRequest(prompt='say something')


def test_with_reminder_appends_and_does_not_mutate() -> None:
    stricter = with_reminder(REQUEST)

    assert REQUEST.prompt == 'say something'
    assert stricter.prompt.endswith(STRICT_JSON_REMINDER)
    assert stricter.prompt.startswith(REQUEST.prompt)
