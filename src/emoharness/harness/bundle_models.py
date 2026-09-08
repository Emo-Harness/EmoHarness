"""Layout of the harness bundle and the frozen models mirroring its YAML files.

The bundle is the artifact the Builder edits (ADR-0005). Every file in it is
declared here so that a missing or misspelled path fails at load time rather than
in the middle of a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import HARNESS_MANIFEST_FILE

PROMPT_DIR: Final = 'prompts'
VERIFICATION_DIR: Final = 'verification'
SAMPLING_FILE: Final = 'sampling.yaml'
ROUTING_FILE: Final = 'routing.yaml'
AUDIO_PIPELINE_FILE: Final = 'audio_pipeline.yaml'
EVIDENCE_SCHEMA_FILE: Final = 'evidence_schema.json'
OUTPUT_SCHEMA_FILE: Final = 'output_schema.json'

STAGE_NAMES: Final[tuple[str, ...]] = (
    'extract_visual',
    'generate_hypotheses',
    'verify',
    'synthesize',
)
VERIFICATION_MODULES: Final[tuple[str, ...]] = ('cross_modal', 'alternative_hypotheses')
AGREEMENT_FUNCTION: Final = 'modalities_agree'
RANKER_FUNCTION: Final = 'rank'

MAX_FACE_CROPS_LIMIT: Final = 8

#: Placeholder names every rendered prompt may use, filled by the pipeline.
SUPPORTED_PLACEHOLDERS: Final[tuple[str, ...]] = (
    'prompt',
    'quantifier',
    'noun',
    'label_set',
    'audio_text',
    'evidence_json',
    'hypotheses_json',
    'step_titles',
)

#: Placeholder names a prompt must contain for its stage to render correctly.
REQUIRED_PLACEHOLDERS: Final[Mapping[str, tuple[str, ...]]] = {
    'extract_visual': ('noun',),
    'generate_hypotheses': ('noun', 'label_set', 'evidence_json'),
    'verify': ('noun', 'label_set', 'audio_text', 'evidence_json', 'hypotheses_json'),
    'synthesize': (
        'noun',
        'label_set',
        'evidence_json',
        'hypotheses_json',
        'step_titles',
    ),
}


class BundleManifest(BaseModel):
    """Provenance of one harness version (ADR-0006 §7)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    version: str = Field(min_length=1)
    stage: int = Field(ge=1, le=3)
    parent: str | None = None
    builder: str = Field(min_length=1, description="'human' or a Builder model id")
    created: str = Field(min_length=1, description='ISO date')
    notes: str = ''


class FaceCropsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    enabled: bool
    max_crops: int = Field(ge=0, le=MAX_FACE_CROPS_LIMIT)


class ExtraImagesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    enabled: bool


class SamplingConfig(BaseModel):
    """``sampling.yaml``: which extra images the visual stage may attach."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    face_crops: FaceCropsConfig
    extra_images: ExtraImagesConfig


class VerificationRouting(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    agreement_check: str = Field(min_length=1)
    ranker: str = Field(min_length=1)


class AudioRouting(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    use_opinion: bool


class RoutingConfig(BaseModel):
    """``routing.yaml``: which stages run and which deterministic checks they use."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stages: tuple[str, ...] = Field(min_length=1)
    verification: VerificationRouting
    audio: AudioRouting


class AudioPipelineConfig(BaseModel):
    """``audio_pipeline.yaml``: how much of the cached audio evidence is shown."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    channels: tuple[str, ...] = Field(min_length=1)
    transcript_max_chars: int = Field(ge=0)


def required_files() -> tuple[str, ...]:
    """Every relative path a loadable bundle must contain."""
    fixed = (
        HARNESS_MANIFEST_FILE,
        SAMPLING_FILE,
        ROUTING_FILE,
        AUDIO_PIPELINE_FILE,
        EVIDENCE_SCHEMA_FILE,
        OUTPUT_SCHEMA_FILE,
    )
    prompts = tuple(f'{PROMPT_DIR}/{name}.md' for name in STAGE_NAMES)
    modules = tuple(f'{VERIFICATION_DIR}/{name}.py' for name in VERIFICATION_MODULES)
    return fixed + prompts + modules
