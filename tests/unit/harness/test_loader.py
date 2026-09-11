"""Loading, hashing, and copying the harness bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from emoharness.common.errors import PatchRejectedError, ValidationError
from emoharness.harness.loader import (
    HarnessBundle,
    export_schemas,
    load_bundle,
    write_bundle_copy,
)
from emoharness.harness.schema import EvidenceRecord, HarnessOutput

from .conftest import STAGE_MARKERS

#: The bundle contract, spelled out here so a silent change to it fails a test.
STAGES: Final[tuple[str, ...]] = (
    'extract_visual',
    'generate_hypotheses',
    'verify',
    'synthesize',
)
EVIDENCE_SCHEMA_FILE: Final = 'evidence_schema.json'
OUTPUT_SCHEMA_FILE: Final = 'output_schema.json'
REQUIRED_FILES: Final[tuple[str, ...]] = (
    'manifest.json',
    'sampling.yaml',
    'routing.yaml',
    'audio_pipeline.yaml',
    EVIDENCE_SCHEMA_FILE,
    OUTPUT_SCHEMA_FILE,
    *(f'prompts/{stage}.md' for stage in STAGES),
    'verification/cross_modal.py',
    'verification/alternative_hypotheses.py',
)
REQUIRED_PLACEHOLDERS: Final[Mapping[str, tuple[str, ...]]] = {
    'extract_visual': ('{noun}',),
    'generate_hypotheses': ('{noun}', '{label_set}', '{evidence_json}'),
    'verify': (
        '{noun}',
        '{label_set}',
        '{audio_text}',
        '{evidence_json}',
        '{hypotheses_json}',
    ),
    'synthesize': (
        '{noun}',
        '{label_set}',
        '{evidence_json}',
        '{hypotheses_json}',
        '{step_titles}',
    ),
}


@pytest.fixture
def bundle_dir(repo_root: Path, tmp_path: Path) -> Path:
    return write_bundle_copy(repo_root / 'harness', tmp_path / 'copy', {})


def test_stage_one_bundle_loads(bundle: HarnessBundle) -> None:
    assert bundle.manifest.version == 'stage1-v0'
    assert bundle.manifest.stage == 1
    assert bundle.manifest.parent is None
    assert bundle.manifest.builder == 'human'
    assert bundle.routing.stages == STAGES
    assert bundle.audio_pipeline.channels == ('transcript', 'prosody')
    assert bundle.sampling.face_crops.enabled is False


def test_every_required_file_is_committed(repo_root: Path) -> None:
    for name in REQUIRED_FILES:
        assert (repo_root / 'harness' / name).is_file(), name


@pytest.mark.parametrize('stage', STAGES)
def test_prompts_carry_their_placeholders(bundle: HarnessBundle, stage: str) -> None:
    text = bundle.prompt(stage)
    for placeholder in REQUIRED_PLACEHOLDERS[stage]:
        assert placeholder in text, placeholder


def test_prompt_bodies_drop_the_documentation_comment(bundle: HarnessBundle) -> None:
    for text in bundle.prompts.values():
        assert '<!--' not in text


@pytest.mark.parametrize('stage', STAGES)
def test_stage_markers_are_unique(bundle: HarnessBundle, stage: str) -> None:
    matching = [
        name for name, text in bundle.prompts.items() if STAGE_MARKERS[stage] in text
    ]
    assert matching == [stage]


def test_prompts_name_no_task_and_no_label_prior(bundle: HarnessBundle) -> None:
    forbidden = ('emotion', 'sentiment', 'intent', 'multi-label', 'fine-grained')
    for text in bundle.prompts.values():
        lowered = text.lower()
        assert not any(word in lowered for word in forbidden)


def test_unknown_stage_prompt_raises(bundle: HarnessBundle) -> None:
    with pytest.raises(ValidationError, match='no prompt'):
        bundle.prompt('nope')


def test_verification_functions_run(bundle: HarnessBundle) -> None:
    record = {
        'items': [
            {'id': 'v1', 'modality': 'visual'},
            {'id': 'a1', 'modality': 'audio'},
        ]
    }
    hypotheses = {
        'coarse': 'x',
        'hypotheses': [
            {'label': 'sadness', 'support': ['v1', 'a1'], 'confidence': 0.6},
            {'label': 'anger', 'support': [], 'confidence': 0.9},
        ],
    }
    assert bundle.agree_fn()(record, hypotheses) is True
    ranked = bundle.rank_fn()(record, hypotheses)
    assert [row['label'] for row in ranked] == ['anger', 'sadness']


def test_editable_paths_follow_the_stage_ladder(bundle: HarnessBundle) -> None:
    assert bundle.editable_paths(1) == ()
    stage_two = bundle.editable_paths(2)
    assert 'routing.yaml' in stage_two
    assert 'prompts/verify.md' in stage_two
    assert not any(path.endswith('.py') for path in stage_two)
    stage_three = bundle.editable_paths(3)
    assert set(stage_two) < set(stage_three)
    assert 'verification/cross_modal.py' in stage_three


def test_editable_paths_rejects_unknown_stage(bundle: HarnessBundle) -> None:
    with pytest.raises(ValidationError, match='unknown harness stage'):
        bundle.editable_paths(9)


def test_content_hash_changes_with_any_file(bundle_dir: Path) -> None:
    before = load_bundle(bundle_dir).content_hash()
    assert before == load_bundle(bundle_dir).content_hash()
    (bundle_dir / 'prompts' / 'verify.md').write_text(
        load_bundle(bundle_dir).prompt('verify') + '\nOne more line.\n',
        encoding='utf-8',
    )
    assert load_bundle(bundle_dir).content_hash() != before


def test_missing_file_is_reported(bundle_dir: Path) -> None:
    (bundle_dir / 'routing.yaml').unlink()
    with pytest.raises(ValidationError, match='routing.yaml'):
        load_bundle(bundle_dir)


def test_prompt_without_required_placeholder_is_rejected(bundle_dir: Path) -> None:
    (bundle_dir / 'prompts' / 'synthesize.md').write_text(
        'Write it.\n', encoding='utf-8'
    )
    with pytest.raises(ValidationError, match='is missing'):
        load_bundle(bundle_dir)


def test_prompt_with_unknown_placeholder_is_rejected(bundle_dir: Path) -> None:
    path = bundle_dir / 'prompts' / 'extract_visual.md'
    path.write_text(
        path.read_text(encoding='utf-8') + '\n{gold_label}\n', encoding='utf-8'
    )
    with pytest.raises(ValidationError, match='unknown'):
        load_bundle(bundle_dir)


def test_routing_with_unknown_stage_is_rejected(bundle_dir: Path) -> None:
    (bundle_dir / 'routing.yaml').write_text(
        'stages: [peek_at_labels]\n'
        'verification: {agreement_check: cross_modal, ranker: alternative_hypotheses}\n',
        encoding='utf-8',
    )
    with pytest.raises(ValidationError, match='unknown stages'):
        load_bundle(bundle_dir)


def test_routing_with_unknown_module_is_rejected(bundle_dir: Path) -> None:
    (bundle_dir / 'routing.yaml').write_text(
        'stages: [verify]\n'
        'verification: {agreement_check: cross_modal, ranker: secret_ranker}\n',
        encoding='utf-8',
    )
    with pytest.raises(ValidationError, match='unknown modules'):
        load_bundle(bundle_dir)


def test_write_bundle_copy_replaces_files(repo_root: Path, tmp_path: Path) -> None:
    dest = write_bundle_copy(
        repo_root / 'harness',
        tmp_path / 'candidate',
        {'prompts/verify.md': 'Look at {noun} in {label_set}.\n'},
    )
    assert (
        (dest / 'prompts' / 'verify.md').read_text(encoding='utf-8').startswith('Look')
    )
    assert (dest / 'manifest.json').is_file()


@pytest.mark.parametrize(
    'relative', ['../escape.md', 'prompts/../../escape.md', '/etc/passwd', '']
)
def test_write_bundle_copy_refuses_paths_outside_the_bundle(
    repo_root: Path, tmp_path: Path, relative: str
) -> None:
    with pytest.raises(PatchRejectedError, match='outside the harness bundle'):
        write_bundle_copy(
            repo_root / 'harness', tmp_path / 'candidate', {relative: 'x'}
        )
    assert not (tmp_path / 'candidate').exists()


def test_export_schemas_matches_the_models(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert [path.name for path in written] == [
        EVIDENCE_SCHEMA_FILE,
        OUTPUT_SCHEMA_FILE,
    ]
    evidence = json.loads((tmp_path / EVIDENCE_SCHEMA_FILE).read_text(encoding='utf-8'))
    output = json.loads((tmp_path / OUTPUT_SCHEMA_FILE).read_text(encoding='utf-8'))
    assert evidence == EvidenceRecord.model_json_schema()
    assert output == HarnessOutput.model_json_schema()


def test_committed_schemas_are_current(repo_root: Path, tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name in (EVIDENCE_SCHEMA_FILE, OUTPUT_SCHEMA_FILE):
        committed = (repo_root / 'harness' / name).read_text(encoding='utf-8')
        assert json.loads(committed) == json.loads(
            (tmp_path / name).read_text(encoding='utf-8')
        )
