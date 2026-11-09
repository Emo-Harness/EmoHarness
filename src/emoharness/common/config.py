"""One loader for every YAML configuration in the project.

Configs are frozen pydantic models and they are all read the same way: parse the
document, expand ``${VAR}`` and ``${VAR:-default}`` references against the
environment so one committed file can point at different endpoints per machine,
then validate. A document that does not fit its model is a boundary failure,
reported with the file it came from and the field that failed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from emoharness.common.errors import ValidationError
from emoharness.common.io import read_yaml

M = TypeVar('M', bound=BaseModel)

_ENV_PATTERN: Final = re.compile(
    r'\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}'
)
_ROOT_FIELD: Final = '<document>'


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references from the environment.

    A reference with no default that names an unset variable raises
    ``ValidationError``; plain text without ``${...}`` is returned unchanged.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group('name')
        default = match.group('default')
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise ValidationError(
            f'environment variable {name!r} is not set and has no default'
        )

    return _ENV_PATTERN.sub(_replace, text)


def expand_strings(obj: Any) -> Any:
    """Recursively apply :func:`expand_env` to every string leaf of a document."""
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, Mapping):
        return {key: expand_strings(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str | bytes):
        return [expand_strings(item) for item in obj]
    return obj


def load_model(path: Path, model: type[M]) -> M:
    """Read ``path`` as YAML, expand its environment references, and validate it."""
    document = expand_strings(read_yaml(path))
    try:
        return model.model_validate(document)
    except PydanticValidationError as exc:
        raise ValidationError(f'{path}: {_first_problem(exc)}') from exc


def _first_problem(exc: PydanticValidationError) -> str:
    """The first field pydantic rejected, and how many others it also rejected."""
    errors = exc.errors()
    if not errors:  # pragma: no cover - pydantic always reports at least one
        return str(exc)
    first = errors[0]
    field = '.'.join(str(part) for part in first['loc']) or _ROOT_FIELD
    rest = f' (and {len(errors) - 1} more)' if len(errors) > 1 else ''
    return f'{field}: {first["msg"]}{rest}'
