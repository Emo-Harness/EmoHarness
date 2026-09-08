"""Tests for emoharness.common.config: the one loader every YAML config goes through."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.config import expand_env, expand_strings, load_model
from emoharness.common.errors import ValidationError


class Inner(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    endpoint: str = Field(min_length=1)


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    inner: Inner
    tags: tuple[str, ...] = ()


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding='utf-8')
    return path


def test_expand_env_uses_environment_value_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_VAR', 'http://example.com:1234/v1')

    assert expand_env('${EMOHARNESS_TEST_VAR:-http://fallback/v1}') == (
        'http://example.com:1234/v1'
    )


def test_expand_env_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EMOHARNESS_TEST_VAR', raising=False)

    assert (
        expand_env('${EMOHARNESS_TEST_VAR:-http://fallback/v1}') == 'http://fallback/v1'
    )


def test_expand_env_raises_when_missing_and_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('EMOHARNESS_TEST_VAR', raising=False)

    with pytest.raises(ValidationError, match='is not set'):
        expand_env('${EMOHARNESS_TEST_VAR}')


def test_expand_env_leaves_plain_text_untouched() -> None:
    assert expand_env('http://127.0.0.1:8000/v1') == 'http://127.0.0.1:8000/v1'


def test_expand_env_expands_multiple_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EMOHARNESS_HOST', 'example.com')
    monkeypatch.delenv('EMOHARNESS_PORT', raising=False)

    result = expand_env('http://${EMOHARNESS_HOST}:${EMOHARNESS_PORT:-9000}/v1')

    assert result == 'http://example.com:9000/v1'


def test_expand_strings_recurses_into_lists_and_nested_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_VAR', 'resolved')
    raw = {
        'plain': 1,
        'items': ['${EMOHARNESS_TEST_VAR}', 'literal', 3],
        'nested': {'inner': '${EMOHARNESS_TEST_VAR}'},
    }

    assert expand_strings(raw) == {
        'plain': 1,
        'items': ['resolved', 'literal', 3],
        'nested': {'inner': 'resolved'},
    }


def test_expand_strings_leaves_non_strings_alone() -> None:
    assert expand_strings(3.5) == 3.5
    assert expand_strings(None) is None
    assert expand_strings(b'${NOPE}') == b'${NOPE}'


def test_load_model_validates_a_document(tmp_path: Path) -> None:
    path = _write(
        tmp_path / 'sample.yaml',
        'name: alpha\ncount: 3\ninner: {endpoint: http://x/v1}\ntags: [a, b]\n',
    )

    config = load_model(path, Sample)

    assert config.name == 'alpha'
    assert config.count == 3
    assert config.inner.endpoint == 'http://x/v1'
    assert config.tags == ('a', 'b')


def test_load_model_expands_environment_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_ENDPOINT', 'http://gpu-box:9001/v1')
    path = _write(
        tmp_path / 'sample.yaml',
        'name: ${EMOHARNESS_TEST_ENDPOINT:-fallback}\n'
        'inner: {endpoint: "${EMOHARNESS_TEST_ENDPOINT}"}\n'
        'tags: ["${EMOHARNESS_TEST_ENDPOINT}"]\n',
    )

    config = load_model(path, Sample)

    assert config.inner.endpoint == 'http://gpu-box:9001/v1'
    assert config.tags == ('http://gpu-box:9001/v1',)


def test_load_model_names_the_file_and_the_failing_field(tmp_path: Path) -> None:
    path = _write(tmp_path / 'sample.yaml', 'name: alpha\ninner: {endpoint: ""}\n')

    with pytest.raises(ValidationError) as excinfo:
        load_model(path, Sample)

    message = str(excinfo.value)
    assert str(path) in message
    assert 'inner.endpoint' in message


def test_load_model_counts_the_other_failures(tmp_path: Path) -> None:
    path = _write(tmp_path / 'sample.yaml', 'count: 0\ninner: {}\n')

    with pytest.raises(ValidationError, match='more'):
        load_model(path, Sample)


def test_load_model_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    path = _write(tmp_path / 'sample.yaml', '- just\n- a\n- list\n')

    with pytest.raises(ValidationError, match=str(path)):
        load_model(path, Sample)


def test_load_model_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / 'sample.yaml',
        'name: alpha\ninner: {endpoint: http://x/v1}\nunexpected: 1\n',
    )

    with pytest.raises(ValidationError, match='unexpected'):
        load_model(path, Sample)


def test_load_model_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='cannot read YAML'):
        load_model(tmp_path / 'missing.yaml', Sample)


def test_load_model_propagates_an_unset_environment_variable(tmp_path: Path) -> None:
    path = _write(
        tmp_path / 'sample.yaml',
        'name: alpha\ninner: {endpoint: "${EMOHARNESS_UNSET_VAR}"}\n',
    )

    with pytest.raises(ValidationError, match='EMOHARNESS_UNSET_VAR'):
        load_model(path, Sample)
