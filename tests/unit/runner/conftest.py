"""Fixtures for the runner: a small, fully prepared project tree built with fakes.

The tree is built once per test module and every context points at it with
``force=True``, so a test may rewrite any artifact it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.data.fakes import make_fake_benchmark, make_fake_external
from emoharness.runner.context import RunContext, build_context
from emoharness.runner.ingest import (
    draw_and_write_subset,
    prepare_benchmark,
    prepare_external,
)
from emoharness.runner.items import load_benchmark, load_external
from emoharness.runner.perception import preprocess_all

PER_TASK = 2
PER_SOURCE = 2
PER_KIND = {'build': 2, 'val': 2, 'ext_test': 2}
RESAMPLES = 50


def make_context(root: Path) -> RunContext:
    """A fake-mode context over ``root`` that may overwrite its own artifacts."""
    return build_context(root, fake=True, force=True, resamples=RESAMPLES)


def build_project(root: Path) -> RunContext:
    """Manifests, subset, and a filled preprocessing cache, all synthetic."""
    ctx = make_context(root)
    source = ctx.paths.fake_raw / 'mme_emotion'
    make_fake_benchmark(source, per_task=PER_TASK)
    prepare_benchmark(ctx, source)
    prepare_external(ctx, make_fake_external(ctx.paths.fake_raw, PER_SOURCE), PER_KIND)
    draw_and_write_subset(ctx)
    examples = load_benchmark(ctx.paths) + load_external(ctx.paths)
    preprocess_all(examples, ctx.clip_cache(), ctx.preprocess, ctx.clip_preprocessor())
    return ctx


@pytest.fixture(scope='module')
def prepared_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project root with both manifests, the subset, and the clip cache."""
    root = tmp_path_factory.mktemp('runner_project')
    build_project(root)
    return root


@pytest.fixture
def ctx(prepared_root: Path) -> RunContext:
    """A fresh context over the prepared tree."""
    return make_context(prepared_root)
