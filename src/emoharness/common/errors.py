"""Exception hierarchy. Every error raised by the package derives from EmoHarnessError."""

from __future__ import annotations

from typing import NoReturn


class EmoHarnessError(Exception):
    """Base class for all package errors."""


class ValidationError(EmoHarnessError):
    """Input data or configuration failed validation at a system boundary."""


class PromptParseError(ValidationError):
    """A prompt did not match the public MME-Emotion template grammar."""


class PatchRejectedError(EmoHarnessError):
    """A Builder patch touched a path or construct outside the allowed set."""


class TargetError(EmoHarnessError):
    """A Target model call failed after retries."""


class TargetUnavailableError(TargetError):
    """The Target endpoint itself is down or its engine is dead.

    Unlike a failed call, this is never absorbed as one wrong answer: a row that
    kept going would record a thousand fallbacks against a dead server.
    """


class LeakageError(EmoHarnessError):
    """A leakage check found benchmark material where it must not be."""


def assert_never(value: NoReturn) -> NoReturn:
    """Exhaustiveness guard: passing a still-possible value is a type error.

    ``typing.assert_never`` only exists from Python 3.11; this is the same
    contract for the 3.10 floor the project supports.
    """
    raise EmoHarnessError(f'unhandled value: {value!r}')
