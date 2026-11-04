"""Layout of the harness bundle and the frozen models mirroring its YAML files.

The bundle is the artifact the Builder edits. Every file in it is
declared here so that a missing or misspelled path fails at load time rather than
in the middle of a run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
#: Upper bound a bundle may ask the visual stage for; 64 frames of a 640 px clip
#: stay inside a 32768-token serving context for every student.
MAX_VIDEO_FRAMES_LIMIT: Final = 64

#: Placeholder names every rendered prompt may use, filled by the pipeline.
SUPPORTED_PLACEHOLDERS: Final[tuple[str, ...]] = (
    'prompt',
    'quantifier',
    'noun',
    'label_set',
    'audio_text',
    'evidence_json',
    'hypotheses_json',
    'verification_json',
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
    """Provenance of one harness version."""

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


class VideoFramesConfig(BaseModel):
    """How many frames the visual stage asks for, scaled with clip length.

    The call samples ``clamp(ceil(duration * min_fps), min_frames, max_frames)``
    frames, so a 60 s clip is seen at ``min_fps`` instead of being squeezed into
    the target config's fixed cap. Absent, the target config decides.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    min_fps: float = Field(gt=0.0, le=4.0)
    min_frames: int = Field(ge=1)
    max_frames: int = Field(ge=1, le=MAX_VIDEO_FRAMES_LIMIT)

    @model_validator(mode='after')
    def _bounds(self) -> VideoFramesConfig:
        if self.max_frames < self.min_frames:
            raise ValueError('video.max_frames must be >= video.min_frames')
        return self

    def frames_for(self, duration_s: float) -> int:
        wanted = math.ceil(max(duration_s, 0.0) * self.min_fps)
        return max(self.min_frames, min(self.max_frames, wanted))


class SamplingConfig(BaseModel):
    """``sampling.yaml``: which extra images the visual stage may attach."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    face_crops: FaceCropsConfig
    extra_images: ExtraImagesConfig
    video: VideoFramesConfig | None = None


class VerificationRouting(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    agreement_check: str = Field(min_length=1)
    ranker: str = Field(min_length=1)


class MechanismScreenConfig(BaseModel):
    """Opt-in v5 mechanisms; absent/default settings preserve existing requests."""

    model_config = ConfigDict(frozen=True, extra='forbid')
    enabled: bool = False
    evidence_layout: Literal['legacy_json', 'source_table'] = 'legacy_json'
    verify_style: Literal['all_hypotheses', 'top2_contrast'] = 'all_hypotheses'
    verify_trigger: Literal['legacy', 'conflict_or_low_margin'] = 'legacy'
    margin_threshold: float = Field(default=0.15, ge=0.0, le=0.5)
    review_media: Literal['none', 'video_on_trigger'] = 'none'
    decision_media: Literal['summary_only', 'cached_frames'] = 'summary_only'
    decision_frame_limit: int = Field(default=4, ge=1, le=8)


class RoutingConfig(BaseModel):
    """``routing.yaml``: which stages run and which deterministic checks they use."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    stages: tuple[str, ...] = Field(min_length=1)
    verification: VerificationRouting
    verify_on: Literal['always', 'conflict'] = 'always'
    workflow: Literal['multistep', 'direct_same_tools'] = 'multistep'
    rank_policy: Literal['bundle', 'confidence', 'evidence_weighted'] = 'bundle'
    audio_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    visual_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    mechanism_screen: MechanismScreenConfig = Field(
        default_factory=MechanismScreenConfig
    )


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
