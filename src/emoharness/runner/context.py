"""The run context: paths, configs, and the factories that switch on ``--fake``.

This is the composition root, and the only module in the package that imports a
fake: every seam ships its own double beside its real adapter, and the ``fake``
flag here is the one place that chooses between them. Every factory returns a
fresh object, so callers hoist them out of loops. The Target factories always
wrap their client in a :class:`CachedTarget` over ``data/derived/calls/``: that
cache is what lets a Builder round reuse the stages its patch did not touch, and
what guarantees every row sees identical inputs (ADR-0006 §5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from emoharness.builder.client import Builder
from emoharness.builder.config import BuilderConfig
from emoharness.builder.fakes import fake_builder
from emoharness.builder.gates import GateConfig
from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.evaluation.fakes import ScoreSpec, fake_score_table, fake_step_scorer
from emoharness.evaluation.judge import GeminiStepScorer, OpenAIStepScorer, StepScorer
from emoharness.evaluation.ladder import Ladder
from emoharness.evaluation.schema import StepScores
from emoharness.harness.fake_responder import label_chooser, scripted_responder
from emoharness.perceive.cache import (
    PreprocessCache,
    PreprocessConfig,
    PreprocessedClip,
)
from emoharness.perceive.fakes import fake_perception_tools, fake_preprocess
from emoharness.perceive.pipeline import PerceptionTools, preprocess_clip
from emoharness.runner.paths import ProjectPaths
from emoharness.runner.perception import (
    ClipPreprocessor,
    load_preprocess_config,
    preprocess_cache,
    real_perception_tools,
)
from emoharness.target.base import TargetModel
from emoharness.target.cache import CachedTarget, CallCache
from emoharness.target.config import ProviderName, TargetConfig
from emoharness.target.fake import (
    FAKE_PRICING,
    FAKE_TEACHER_ID,
    scripted_target,
)
from emoharness.target.pricing import Pricing

DEFAULT_EXPERIMENT: Final = 'emoharness'
DEFAULT_TARGET_CONFIG: Final = 'target/qwen3vl_8b_instruct.yaml'
FAKE_ROUNDS: Final = 2
#: How the teacher row sees a clip. Only a Gemini teacher can take native video.
TeacherMode = Literal['native_video', 'matched_protocol']
GEMINI_PROVIDER: Final[ProviderName] = 'gemini'
OPENAI_COMPAT_PROVIDER: Final[ProviderName] = 'openai_compat'
#: Ledger role the teacher row's calls are billed under, whatever the provider.
TEACHER_ROLE: Final = 'teacher_direct'
#: Under data/derived: mp4 copies of clips whose container the server cannot count.
REMUX_DIR: Final = 'remux'
#: Environment variable each proxy-judge provider reads its API key from.
JUDGE_KEY_ENV: Final[Mapping[str, str]] = {
    'gemini': 'GEMINI_API_KEY',
    'openai': 'OPENAI_API_KEY',
}


class CalibrationSpec(BaseModel):
    """The proxy judge's calibration plan (ADR-0007 §3)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    reference_model: str = Field(min_length=1)
    sample_size: int = Field(ge=1)
    min_agreement: float = Field(ge=0.0, le=1.0)


class ProxyJudgeConfig(BaseModel):
    """``configs/builder/proxy_judge.yaml``: the Reasoning gate's scorer."""

    model_config = ConfigDict(
        frozen=True, extra='forbid', populate_by_name=True, protected_namespaces=()
    )

    model_id: str = Field(min_length=1)
    provider: Literal['gemini', 'openai'] = 'gemini'
    api_key_env: str | None = None
    endpoint: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(
        default=None,
        description='openai provider, GPT-5 family: the reasoning_effort to send',
    )
    frame_interval_seconds: float = Field(default=1.0, gt=0.0)
    max_frames: int = Field(default=10, ge=1)
    calibration: CalibrationSpec
    pricing: Mapping[str, float] = Field(
        default_factory=dict, alias='pricing_usd_per_million'
    )

    def key_env(self) -> str:
        """The environment variable holding this judge's API key."""
        return self.api_key_env or JUDGE_KEY_ENV[self.provider]


@dataclass(frozen=True)
class RunContext:
    """Everything an entry point needs, and nothing an entry point should decide."""

    paths: ProjectPaths
    fake: bool
    force: bool
    experiment: str
    preprocess: PreprocessConfig
    ladder: Ladder
    builder: BuilderConfig
    target_config: Path | None
    bundle_override: Path | None
    replicate: int
    resamples: int

    def with_experiment(self, experiment: str) -> RunContext:
        """A copy whose ledger entries are tagged with a different experiment name."""
        return replace(self, experiment=experiment)

    @property
    def ledger(self) -> Path:
        return self.paths.ledger

    def call_cache(self) -> CallCache:
        """The content-addressed Target-call cache shared by every row."""
        return CallCache(self.paths.call_cache)

    def clip_cache(self) -> PreprocessCache:
        """The shared preprocessing cache."""
        return preprocess_cache(self.paths)

    def target(self) -> TargetModel:
        """The frozen Target under test, cached and ledger-recording."""
        return CachedTarget(self._raw_target(), self.call_cache())

    def _raw_target(self) -> TargetModel:
        if self.fake:
            return scripted_target(
                scripted_responder(label_chooser()), self.ledger, self.experiment
            )
        from emoharness.target.openai_compat import OpenAICompatTarget

        return OpenAICompatTarget(
            self.target_model_config(),
            self.ledger,
            self.experiment,
            media_dir=self.paths.derived / REMUX_DIR,
        )

    def pricing(self) -> Pricing:
        """The price list one Target response is charged at."""
        if self.fake:
            return FAKE_PRICING
        return self.target_model_config().pricing

    def target_model_config(self) -> TargetConfig:
        """The Target YAML this run was pointed at."""
        path = self.target_config or self.paths.configs / DEFAULT_TARGET_CONFIG
        if not path.is_file():
            raise ValidationError(f'target config {path} does not exist')
        return load_model(path, TargetConfig)

    def teacher_pricing(self) -> Pricing:
        """The price list one teacher response is charged at."""
        if self.fake:
            return FAKE_PRICING
        return self.teacher_model_config().pricing

    def teacher_model_config(self) -> TargetConfig:
        """The teacher YAML: GPT 5.6 Luna by default, Gemini as the alternative."""
        return load_model(self.paths.teacher_config, TargetConfig)

    def teacher_provider(self) -> ProviderName:
        """Which client answers the teacher row; ``--fake`` never asks a provider."""
        if self.fake:
            return OPENAI_COMPAT_PROVIDER
        return self.teacher_model_config().provider

    def teacher_target(self, mode: TeacherMode) -> TargetModel:
        """The teacher Direct upper bound; a second fake model stands in for ``--fake``."""
        if self.fake:
            inner: TargetModel = scripted_target(
                scripted_responder(label_chooser(mode)),
                self.ledger,
                self.experiment,
                FAKE_TEACHER_ID,
            )
            return CachedTarget(inner, self.call_cache())
        config = self.teacher_model_config()
        if config.provider == GEMINI_PROVIDER:
            from emoharness.target.gemini import GeminiDirectTarget, gemini_config_from

            inner = GeminiDirectTarget(
                gemini_config_from(config), self.ledger, self.experiment, mode
            )
        else:
            from emoharness.target.openai_compat import OpenAICompatTarget

            inner = OpenAICompatTarget(
                config,
                self.ledger,
                self.experiment,
                role=TEACHER_ROLE,
                media_dir=self.paths.derived / REMUX_DIR,
            )
        return CachedTarget(inner, self.call_cache())

    def perception_tools(self) -> PerceptionTools:
        """The audio stack; heavy adapters are only imported for real runs."""
        if self.fake:
            return fake_perception_tools(self.paths.emotion_lexicon)
        return real_perception_tools(self.paths, self.preprocess)

    def clip_preprocessor(self) -> ClipPreprocessor:
        """One clip in, one cache entry out: the fake writer, or the real stack.

        The fake path never opens a video, so it needs no audio models; the real
        path builds its adapters once and reuses them for every clip.
        """
        if self.fake:
            return fake_preprocess
        tools = self.perception_tools()

        def preprocess(
            example: Example, cache: PreprocessCache, config: PreprocessConfig
        ) -> PreprocessedClip:
            return preprocess_clip(example, config, tools, cache)

        return preprocess

    def builder_client(self, stage: int) -> Builder:
        """The Builder for one construction stage."""
        if self.fake:
            return fake_builder(stage, self.builder.max_rounds_per_stage)
        from emoharness.builder.client import GeminiBuilder, OpenAIBuilder

        if self.builder.provider == 'gemini':
            return GeminiBuilder(self.builder, self.ledger, self.experiment)
        return OpenAIBuilder(self.builder, self.ledger, self.experiment)

    def gate_config(self) -> GateConfig:
        """Promotion thresholds, taken from the Builder config's gate block."""
        gates = self.builder.gates
        return GateConfig(
            regression_rate_limit=gates.regression_rate_limit,
            reasoning_margin_points=gates.reasoning_floor_points,
            max_mean_calls=gates.max_mean_calls,
        )

    def proxy_judge(self) -> ProxyJudgeConfig:
        """The proxy judge configuration used by the Reasoning gate."""
        return load_model(self.paths.proxy_judge_config, ProxyJudgeConfig)

    def step_score_table(
        self, items: Mapping[str, ScoreSpec], salt: str = ''
    ) -> Mapping[str, StepScores]:
        """The answers a ``--fake`` step scorer reads back; a real one ignores them.

        ``items`` maps an example id to how many labels it got right, how many it
        was asked for, and how many steps its response carried. ``salt`` makes two
        scorers over the same items disagree the way two judges would.
        """
        return fake_score_table(items, salt)

    def step_scorer(self, table: Mapping[str, StepScores]) -> StepScorer:
        """A judge for extracted steps; ``table`` answers every id in ``--fake`` mode."""
        if self.fake:
            return fake_step_scorer(table)
        config = self.proxy_judge()
        if config.provider == 'openai':
            return OpenAIStepScorer(
                model_id=config.model_id,
                api_key_env=config.key_env(),
                endpoint=config.endpoint,
                ledger_path=self.ledger,
                experiment=self.experiment,
                pricing=config.pricing,
                reasoning_effort=config.reasoning_effort,
            )
        return GeminiStepScorer(
            model_id=config.model_id,
            api_key_env=config.key_env(),
            ledger_path=self.ledger,
            experiment=self.experiment,
            pricing=config.pricing,
        )


def build_context(
    root: Path,
    *,
    fake: bool = False,
    force: bool = False,
    experiment: str = DEFAULT_EXPERIMENT,
    target_config: Path | None = None,
    bundle: Path | None = None,
    replicate: int = 0,
    rounds: int | None = None,
    resamples: int = 10_000,
    source: Path | None = None,
) -> RunContext:
    """Assemble a run context and create the artifact directories it will write."""
    paths = ProjectPaths.from_root(root, source)
    paths.ensure()
    builder = load_model(paths.builder_config, BuilderConfig)
    limit = rounds if rounds is not None else (FAKE_ROUNDS if fake else None)
    if limit is not None:
        builder = builder.model_copy(update={'max_rounds_per_stage': limit})
    return RunContext(
        paths=paths,
        fake=fake,
        force=force,
        experiment=experiment,
        preprocess=load_preprocess_config(paths),
        ladder=load_model(paths.ladder_config, Ladder),
        builder=builder,
        target_config=target_config,
        bundle_override=bundle,
        replicate=replicate,
        resamples=resamples,
    )


def context_settings(ctx: RunContext) -> Mapping[str, Any]:
    """A small, JSON-safe description of a run, for artifact provenance blocks."""
    return {
        'root': str(ctx.paths.root),
        'fake': ctx.fake,
        'experiment': ctx.experiment,
        'replicate': ctx.replicate,
        'preprocess_config_hash': ctx.preprocess.config_hash(),
    }
