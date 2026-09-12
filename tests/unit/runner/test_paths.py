"""Project layout and the never-overwrite-silently rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.runner.paths import (
    RESULTS_DIR_ENV,
    ProjectPaths,
    guard_write,
    source_root,
)


class TestFromRoot:
    def test_derives_every_directory_from_one_root(self, tmp_path: Path) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        assert paths.manifests == tmp_path.resolve() / 'data' / 'manifests'
        assert (
            paths.raw_predictions == tmp_path.resolve() / 'results' / 'raw_predictions'
        )
        assert paths.ledger == tmp_path.resolve() / 'results' / 'ledger.jsonl'

    def test_falls_back_to_the_checkout_for_committed_inputs(
        self, tmp_path: Path
    ) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        assert paths.configs == source_root() / 'configs'
        assert paths.bundle == source_root() / 'harness'

    def test_prefers_a_root_that_carries_its_own_configs(self, tmp_path: Path) -> None:
        (tmp_path / 'configs').mkdir()
        (tmp_path / 'harness').mkdir()
        paths = ProjectPaths.from_root(tmp_path)
        assert paths.configs == tmp_path.resolve() / 'configs'
        assert paths.bundle == tmp_path.resolve() / 'harness'

    def test_the_results_directory_can_be_moved_without_moving_the_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second student writes its own results tree and shares data/."""
        monkeypatch.setenv(RESULTS_DIR_ENV, 'results/students/qwen3vl-2b')
        paths = ProjectPaths.from_root(tmp_path)
        root = tmp_path.resolve()
        assert paths.results == root / 'results' / 'students' / 'qwen3vl-2b'
        assert paths.raw_predictions == paths.results / 'raw_predictions'
        assert paths.ledger == paths.results / 'ledger.jsonl'
        # The shared caches are keyed off data/, which never moves.
        assert paths.derived == root / 'data' / 'derived'
        assert paths.manifests == root / 'data' / 'manifests'

    @pytest.mark.parametrize('value', ['/elsewhere', '../outside', 'a/../../b'])
    def test_a_results_override_may_not_escape_the_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(RESULTS_DIR_ENV, value)
        with pytest.raises(ValidationError, match=RESULTS_DIR_ENV):
            ProjectPaths.from_root(tmp_path)

    def test_an_empty_results_override_falls_back_to_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RESULTS_DIR_ENV, '   ')
        assert (
            ProjectPaths.from_root(tmp_path).results == tmp_path.resolve() / 'results'
        )

    def test_ensure_creates_every_artifact_directory(self, tmp_path: Path) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        paths.ensure()
        assert all(directory.is_dir() for directory in paths.directories)

    def test_is_frozen(self, tmp_path: Path) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        with pytest.raises(Exception, match='frozen|immutable'):
            paths.root = tmp_path  # type: ignore[misc]


class TestArtifactNames:
    def test_predictions_are_named_after_row_and_item_set(self, tmp_path: Path) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        assert paths.predictions('L5', 'full').name == 'full.jsonl'
        assert paths.harness_outputs('L5', 'full').name == 'full.outputs.jsonl'
        assert paths.predictions('L5', 'full').parent.name == 'L5'

    def test_an_external_item_set_becomes_a_safe_file_name(
        self, tmp_path: Path
    ) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        assert (
            paths.predictions('L3', 'external:ext_test').name
            == 'external_ext_test.jsonl'
        )
        assert paths.judge_run('L3', 'external:val').name == 'external_val'

    def test_stage_dir_carries_stage_and_replicate(self, tmp_path: Path) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        assert paths.stage_dir(2, 1).parts[-2:] == ('stage2', '1')


class TestGuardWrite:
    def test_returns_the_path_and_creates_its_parent(self, tmp_path: Path) -> None:
        target = guard_write(tmp_path / 'nested' / 'artifact.json', force=False)
        assert target.parent.is_dir()

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path) -> None:
        existing = tmp_path / 'artifact.json'
        existing.write_text('{}', encoding='utf-8')
        with pytest.raises(ValidationError, match='already exists'):
            guard_write(existing, force=False)

    def test_allows_an_overwrite_with_force(self, tmp_path: Path) -> None:
        existing = tmp_path / 'artifact.json'
        existing.write_text('{}', encoding='utf-8')
        assert guard_write(existing, force=True) == existing
