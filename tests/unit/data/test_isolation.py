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
