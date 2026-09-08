"""Static safety check for the pure functions the Builder may write at Stage 3.

ADR-0005 lets the Builder edit ``harness/verification/*.py``. Those files are read
as text and executed here, so they are first parsed and checked against a closed
list of AST nodes, a closed list of callable builtins, and a size limit. Anything
outside the list is a violation and the file is rejected; nothing is repaired.
"""

from __future__ import annotations

import ast
import builtins
import importlib
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

from emoharness.common.errors import PatchRejectedError

MAX_VERIFICATION_SOURCE_BYTES: Final = 20_000
ALLOWED_IMPORT_MODULES: Final[frozenset[str]] = frozenset({'__future__', 'typing'})

ALLOWED_BUILTIN_NAMES: Final[tuple[str, ...]] = (
    'abs',
    'all',
    'any',
    'bool',
    'dict',
    'enumerate',
    'float',
    'int',
    'len',
    'list',
    'max',
    'min',
    'range',
    'round',
    'set',
    'sorted',
    'str',
    'sum',
    'tuple',
    'zip',
)
ALLOWED_BUILTINS: Final[Mapping[str, Any]] = {
    name: getattr(builtins, name) for name in ALLOWED_BUILTIN_NAMES
}

FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        'breakpoint',
        'classmethod',
        'compile',
        'delattr',
        'eval',
        'exec',
        'exit',
        'getattr',
        'globals',
        'hasattr',
        'help',
        'input',
        'locals',
        'memoryview',
        'object',
        'open',
        'property',
        'quit',
        'setattr',
        'staticmethod',
        'super',
        'type',
        'vars',
    }
)

ALLOWED_NODES: Final[frozenset[str]] = frozenset(
    {
        'Add',
        'And',
        'AnnAssign',
        'Assign',
        'BinOp',
        'BitAnd',
        'BitOr',
        'BoolOp',
        'Break',
        'Call',
        'Compare',
        'Constant',
        'Continue',
        'Dict',
        'DictComp',
        'Div',
        'Eq',
        'Expr',
        'FloorDiv',
        'For',
        'FormattedValue',
        'FunctionDef',
        'GeneratorExp',
        'Gt',
        'GtE',
        'If',
        'IfExp',
        'ImportFrom',
        'In',
        'Is',
        'IsNot',
        'JoinedStr',
        'List',
        'ListComp',
        'Load',
        'Lt',
        'LtE',
        'Mod',
        'Module',
        'Mult',
        'Name',
        'Not',
        'NotEq',
        'NotIn',
        'Or',
        'Pass',
        'Pow',
        'Return',
        'Set',
        'SetComp',
        'Slice',
        'Store',
        'Sub',
        'Subscript',
        'Tuple',
        'UAdd',
        'USub',
        'UnaryOp',
        'alias',
        'arg',
        'arguments',
        'comprehension',
        'keyword',
    }
)

_NODE_MESSAGES: Final[Mapping[str, str]] = {
    'Attribute': 'attribute access is not allowed',
    'AsyncFunctionDef': 'async functions are not allowed',
    'Await': 'await is not allowed',
    'ClassDef': 'class definitions are not allowed',
    'Global': 'global is not allowed',
    'Import': 'plain import is not allowed; use from typing import ...',
    'Lambda': 'lambda is not allowed',
    'Nonlocal': 'nonlocal is not allowed',
    'Raise': 'raise is not allowed',
    'Try': 'try/except is not allowed',
    'While': 'while loops are not allowed',
    'With': 'with blocks are not allowed',
    'Yield': 'yield is not allowed',
}


def check_verification_source(source: str, path: str) -> tuple[str, ...]:
    """Every reason ``source`` may not be executed. Empty means it may."""
    size = len(source.encode('utf-8'))
    if size > MAX_VERIFICATION_SOURCE_BYTES:
        return (f'{path}: {size} bytes exceeds {MAX_VERIFICATION_SOURCE_BYTES}',)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return (f'{path}: syntax error: {exc}',)
    defined = frozenset(
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )
    found: list[str] = []
    for node in ast.walk(tree):
        found.extend(_check_node(node, defined, path))
    return tuple(found)


def load_verification_function(source: str, name: str, path: str) -> Callable[..., Any]:
    """Check ``source``, execute it with no ambient builtins, and return ``name``."""
    violations = check_verification_source(source, path)
    if violations:
        raise PatchRejectedError(f'{path} rejected: ' + '; '.join(violations))
    namespace: dict[str, Any] = {'__builtins__': _exec_builtins()}
    exec(compile(source, path, 'exec'), namespace)  # noqa: S102
    function = namespace.get(name)
    if not callable(function):
        raise PatchRejectedError(f'{path} defines no callable named {name!r}')
    return cast('Callable[..., Any]', function)


def _exec_builtins() -> dict[str, Any]:
    """The only builtins a verification module ever sees."""
    return {**ALLOWED_BUILTINS, '__import__': _restricted_import}


def _restricted_import(
    name: str,
    _globals: Mapping[str, Any] | None = None,
    _locals: Mapping[str, Any] | None = None,
    _fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Import shim: only ``__future__`` and ``typing`` resolve."""
    if level != 0 or name not in ALLOWED_IMPORT_MODULES:
        raise PatchRejectedError(f'import of {name!r} is not allowed')
    return importlib.import_module(name)


def _check_node(node: ast.AST, defined: frozenset[str], path: str) -> tuple[str, ...]:
    """Violations contributed by one node."""
    kind = type(node).__name__
    if kind not in ALLOWED_NODES:
        return (
            _message(path, node, _NODE_MESSAGES.get(kind, f'{kind} is not allowed')),
        )
    checker = _CHECKERS.get(kind)
    return checker(node, defined, path) if checker is not None else ()


def _check_call(node: ast.AST, defined: frozenset[str], path: str) -> tuple[str, ...]:
    if not isinstance(node, ast.Call):
        return ()
    if not isinstance(node.func, ast.Name):
        return (_message(path, node, 'only calls to plain names are allowed'),)
    called = node.func.id
    if called in defined or called in ALLOWED_BUILTINS:
        return ()
    return (_message(path, node, f'call to {called!r} is not allowed'),)


def _check_import(
    node: ast.AST, _defined: frozenset[str], path: str
) -> tuple[str, ...]:
    if not isinstance(node, ast.ImportFrom):
        return ()
    module = node.module or ''
    if module in ALLOWED_IMPORT_MODULES and node.level == 0:
        return ()
    return (_message(path, node, f'import from {module!r} is not allowed'),)


def _check_name(node: ast.AST, _defined: frozenset[str], path: str) -> tuple[str, ...]:
    if not isinstance(node, ast.Name):
        return ()
    return _check_identifier(node.id, node, path)


def _check_function(
    node: ast.AST, _defined: frozenset[str], path: str
) -> tuple[str, ...]:
    if not isinstance(node, ast.FunctionDef):
        return ()
    decorated = tuple(
        _message(path, node, 'decorators are not allowed') for _ in node.decorator_list
    )
    return decorated + _check_identifier(node.name, node, path)


def _check_arg(node: ast.AST, _defined: frozenset[str], path: str) -> tuple[str, ...]:
    if not isinstance(node, ast.arg):
        return ()
    return _check_identifier(node.arg, node, path)


def _check_alias(node: ast.AST, _defined: frozenset[str], path: str) -> tuple[str, ...]:
    if not isinstance(node, ast.alias):
        return ()
    return _check_identifier(node.asname or node.name, node, path)


def _check_identifier(name: str, node: ast.AST, path: str) -> tuple[str, ...]:
    if name.startswith('__') and name.endswith('__'):
        return (_message(path, node, f'dunder name {name!r} is not allowed'),)
    if name in FORBIDDEN_NAMES:
        return (_message(path, node, f'name {name!r} is not allowed'),)
    return ()


def _message(path: str, node: ast.AST, text: str) -> str:
    line = getattr(node, 'lineno', 0)
    return f'{path}:{line}: {text}'


_CHECKERS: Final[
    Mapping[str, Callable[[ast.AST, frozenset[str], str], tuple[str, ...]]]
] = {
    'Call': _check_call,
    'FunctionDef': _check_function,
    'ImportFrom': _check_import,
    'Name': _check_name,
    'alias': _check_alias,
    'arg': _check_arg,
}
