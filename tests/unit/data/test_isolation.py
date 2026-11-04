import pytest

from emoharness.common.errors import ValidationError
from emoharness.data.isolation import assert_isolated, quarantine_overlaps
from emoharness.data.manifest import Example
from emoharness.data.tasks import Split


def test_transitive_groups_are_quarantined_without_moving_examples(
    external_example: Example,
) -> None:
    a = external_example.model_copy(
        update={
            'example_id': 'a',
            'source_id': 's1',
            'media_path': 'a',
            'media_hash': 'h1',
            'split': Split.BUILD,
        }
    )
    b = external_example.model_copy(
        update={
            'example_id': 'b',
            'source_id': 's2',
            'media_path': 'b',
            'media_hash': 'h1',
            'split': Split.VAL,
        }
    )
    c = external_example.model_copy(
        update={
            'example_id': 'c',
            'source_id': 's2',
            'media_path': 'c',
            'media_hash': 'h2',
            'split': Split.EXT_TEST,
        }
    )
    with pytest.raises(ValidationError, match='share 1'):
        assert_isolated((a, b, c))
    kept, removed = quarantine_overlaps((a, b, c))
    assert kept == (c,) and removed == (a, b)
    assert_isolated(kept)


def _isolate_construction_script():
    """``examples/data/isolate_construction.py``, imported by path (examples/ is not a package)."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / 'examples/data/isolate_construction.py'
    spec = importlib.util.spec_from_file_location('emoharness_example_isolate', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_isolate_construction_writes_the_isolated_manifest(
    tmp_path, external_example: Example
) -> None:
    import json

    from emoharness.data.manifest import read_manifest, write_manifest

    shared = {'source_id': 's1', 'media_hash': 'h1'}
    rows = (
        external_example.model_copy(
            update={
                'example_id': 'b',
                'media_path': 'b',
                'split': Split.BUILD,
                **shared,
            }
        ),
        external_example.model_copy(
            update={'example_id': 'v', 'media_path': 'v', 'split': Split.VAL, **shared}
        ),
        external_example.model_copy(
            update={
                'example_id': 'other',
                'source_id': 's2',
                'media_path': 'o',
                'media_hash': 'h2',
                'split': Split.BUILD,
            }
        ),
    )
    manifests = tmp_path / 'data/manifests'
    manifests.mkdir(parents=True)
    write_manifest(manifests / 'external.jsonl', rows)
    script = _isolate_construction_script()

    assert script.main(['--root', str(tmp_path)]) == 0

    kept = read_manifest(manifests / 'external_v3.jsonl')
    assert [e.example_id for e in kept] == ['v', 'other']
    record = json.loads(
        (tmp_path / 'data/splits/external_v3_isolation.json').read_text()
    )
    assert record['quarantined_ids'] == ['b']
    assert record['counts'] == {'val': 1, 'build': 1}
    # a second run refuses to overwrite without --force
    assert script.main(['--root', str(tmp_path)]) != 0
    assert script.main(['--root', str(tmp_path), '--force']) == 0
