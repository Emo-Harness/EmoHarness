"""The pre-registration document: what it freezes, and that it is reproducible."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.runner.context import RunContext, build_context
from emoharness.runner.leakage import leakage_sweep
from emoharness.runner.preregistration import (
    artifact_hashes,
    build_preregistration,
    config_hashes,
    handcrafted_prompt_text,
    write_preregistration,
)


@pytest.fixture
def registered(ctx: RunContext) -> RunContext:
    """A context whose leakage report exists, which pre-registration requires."""
    if not ctx.paths.leakage_report.is_file():
        leakage_sweep(ctx)
    return ctx


class TestArtifactHashes:
    def test_covers_every_frozen_artifact(self, registered: RunContext) -> None:
        names = ' '.join(artifact_hashes(registered))
        for expected in (
            'subset id list',
            'benchmark manifest',
            'external manifest',
            'external splits',
            'leakage report',
            'Stage 1 harness bundle',
            'L2 handcrafted prompt',
            'configs/',
        ):
            assert expected in names

    def test_every_value_is_a_sha256(self, registered: RunContext) -> None:
        assert all(len(value) == 64 for value in artifact_hashes(registered).values())

    def test_is_stable_across_calls(self, registered: RunContext) -> None:
        assert artifact_hashes(registered) == artifact_hashes(registered)

    def test_refuses_to_pre_register_before_the_artifacts_exist(
        self, tmp_path: Path
    ) -> None:
        empty = build_context(tmp_path, fake=True, force=True)
        with pytest.raises(ValidationError, match='cannot pre-register'):
            artifact_hashes(empty)

    def test_config_hashes_cover_the_committed_configs(
        self, registered: RunContext
    ) -> None:
        digests = config_hashes(registered.paths.configs)
        assert 'experiments/ladder.yaml' in digests
        assert 'preprocess.yaml' in digests


class TestDocument:
    def test_renders_the_same_bytes_twice(self, registered: RunContext) -> None:
        assert build_preregistration(registered) == build_preregistration(registered)

    def test_carries_the_hypotheses_and_the_ladder(
        self, registered: RunContext
    ) -> None:
        text = build_preregistration(registered)
        for expected in ('H1 seen improvement', 'H-B Builder gain', 'H3 gap closed'):
            assert expected in text
        assert '| L5 | harness |' in text

    def test_lists_the_frozen_hashes(self, registered: RunContext) -> None:
        text = build_preregistration(registered)
        for value in artifact_hashes(registered).values():
            assert value in text

    def test_carries_no_timestamp(self, registered: RunContext) -> None:
        assert 'generated at' not in build_preregistration(registered).lower()

    def test_the_l2_prompt_is_readable_from_the_package(self) -> None:
        assert 'Step' in handcrafted_prompt_text()


class TestWrite:
    def test_writes_the_document(self, tmp_path: Path) -> None:
        target = write_preregistration(tmp_path / 'prereg.md', 'body', force=False)
        assert target.read_text(encoding='utf-8') == 'body'

    def test_refuses_to_clobber_without_force(self, tmp_path: Path) -> None:
        path = tmp_path / 'prereg.md'
        write_preregistration(path, 'first', force=False)
        with pytest.raises(ValidationError, match='already exists'):
            write_preregistration(path, 'second', force=False)

    def test_amends_with_force(self, tmp_path: Path) -> None:
        path = tmp_path / 'prereg.md'
        write_preregistration(path, 'first', force=False)
        write_preregistration(path, 'second', force=True)
        assert path.read_text(encoding='utf-8') == 'second'
