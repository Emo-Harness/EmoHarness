"""Item-set resolution: full, clean, subset, and the three construction splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.data.manifest import write_manifest
from emoharness.data.tasks import Split
from emoharness.runner.context import RunContext
from emoharness.runner.items import (
    by_id,
    examples_for,
    external_split,
    gold_of,
    is_external,
    load_benchmark,
    load_external,
    select_items,
    task_of,
)
from emoharness.runner.paths import ProjectPaths


class TestSelectItems:
    def test_full_keeps_every_benchmark_item(self, ctx: RunContext) -> None:
        examples = load_benchmark(ctx.paths)
        assert select_items(examples, 'full', ctx.paths) == examples

    def test_subset_keeps_only_the_pre_registered_ids(self, ctx: RunContext) -> None:
        examples = load_benchmark(ctx.paths)
        chosen = select_items(examples, 'subset', ctx.paths)
        assert 0 < len(chosen) <= len(examples)

    def test_an_external_item_set_selects_one_split(self, ctx: RunContext) -> None:
        chosen = select_items(load_external(ctx.paths), 'external:val', ctx.paths)
        assert chosen and all(e.split is Split.VAL for e in chosen)

    def test_the_three_construction_splits_are_disjoint(self, ctx: RunContext) -> None:
        external = load_external(ctx.paths)
        sets = [
            {
                e.example_id
                for e in select_items(external, f'external:{name}', ctx.paths)
            }
            for name in ('build', 'val', 'ext_test')
        ]
        assert all(
            sets[i] & sets[j] == set() for i in range(3) for j in range(i + 1, 3)
        )

    def test_clean_drops_benchmark_items_whose_clip_is_in_construction(
        self, ctx: RunContext
    ) -> None:
        benchmark = load_benchmark(ctx.paths)
        external = load_external(ctx.paths)
        clip = 'f' * 64
        examples = (
            benchmark[0].model_copy(update={'media_hash': clip}),
            *benchmark[1:],
        )
        patched = (external[0].model_copy(update={'media_hash': clip}), *external[1:])
        write_manifest(ctx.paths.external_manifest, patched)
        try:
            chosen = select_items(examples, 'clean', ctx.paths)
        finally:
            write_manifest(ctx.paths.external_manifest, external)
        assert chosen == examples[1:]

    def test_clean_is_full_when_no_clip_is_shared(self, ctx: RunContext) -> None:
        examples = load_benchmark(ctx.paths)
        assert select_items(examples, 'clean', ctx.paths) == examples

    def test_an_unknown_item_set_is_rejected(self, ctx: RunContext) -> None:
        with pytest.raises(ValidationError, match='unknown item set'):
            select_items((), 'everything', ctx.paths)

    def test_an_unknown_external_split_is_rejected(self, ctx: RunContext) -> None:
        with pytest.raises(ValidationError, match='unknown external split'):
            select_items((), 'external:holdout', ctx.paths)


class TestExamplesFor:
    def test_reads_the_benchmark_manifest_for_a_benchmark_item_set(
        self, ctx: RunContext
    ) -> None:
        assert all(e.is_benchmark for e in examples_for('full', ctx.paths))

    def test_reads_the_external_manifest_for_a_construction_split(
        self, ctx: RunContext
    ) -> None:
        assert all(
            not e.is_benchmark for e in examples_for('external:build', ctx.paths)
        )

    def test_a_missing_manifest_names_the_step_that_writes_it(
        self, tmp_path: Path
    ) -> None:
        paths = ProjectPaths.from_root(tmp_path)
        with pytest.raises(ValidationError, match='run the prepare step first'):
            examples_for('full', paths)


class TestJoins:
    def test_is_external_recognizes_the_prefix(self) -> None:
        assert is_external('external:val')
        assert not is_external('subset')

    def test_external_split_maps_to_the_enum(self) -> None:
        assert external_split('external:ext_test') is Split.EXT_TEST

    def test_by_id_task_of_and_gold_of_agree_on_the_same_items(
        self, ctx: RunContext
    ) -> None:
        examples = examples_for('full', ctx.paths)
        assert set(by_id(examples)) == set(task_of(examples)) == set(gold_of(examples))

    def test_task_of_skips_construction_items(self, ctx: RunContext) -> None:
        assert task_of(examples_for('external:val', ctx.paths)) == {}

    def test_gold_of_keeps_labels_verbatim(self, ctx: RunContext) -> None:
        examples = examples_for('full', ctx.paths)
        gold = gold_of(examples)
        assert all(gold[e.example_id] == e.labels for e in examples)
