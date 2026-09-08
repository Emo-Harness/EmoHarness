"""Read, hash, and copy the harness bundle the Builder edits (ADR-0005, ADR-0006).

A bundle is a directory of plain files. Loading it validates that every required
file exists, that the YAML matches the frozen config models, and that each prompt
carries the placeholders its stage fills. The verification functions inside it are
never imported; they are read as text and executed through the static allowlist.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from string import Formatter
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from emoharness.common.constants import HARNESS_MANIFEST_FILE, STAGE_EDITABLE_GLOBS
from emoharness.common.errors import PatchRejectedError, ValidationError
from emoharness.common.hashing import directory_hash
from emoharness.common.io import read_json, read_text, read_yaml, write_json_atomic
from emoharness.harness.allowlist import load_verification_function
from emoharness.harness.bundle_models import (
    AGREEMENT_FUNCTION,
    AUDIO_PIPELINE_FILE,
    EVIDENCE_SCHEMA_FILE,
    OUTPUT_SCHEMA_FILE,
    PROMPT_DIR,
    RANKER_FUNCTION,
    REQUIRED_PLACEHOLDERS,
    ROUTING_FILE,
    SAMPLING_FILE,
    STAGE_NAMES,
    SUPPORTED_PLACEHOLDERS,
    VERIFICATION_DIR,
    VERIFICATION_MODULES,
    AudioPipelineConfig,
    BundleManifest,
    RoutingConfig,
    SamplingConfig,
    required_files,
)
from emoharness.harness.schema import EvidenceRecord, HarnessOutput

__all__ = [
    'AudioPipelineConfig',
    'BundleManifest',
    'HarnessBundle',
    'RoutingConfig',
    'SamplingConfig',
    'export_schemas',
    'load_bundle',
    'write_bundle_copy',
]

AgreeFn = Callable[[dict[str, Any], dict[str, Any]], bool]
RankFn = Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]

_IGNORED_COPY_PATTERNS = ('__pycache__', '*.pyc')
_COMMENT_RE = re.compile(r'<!--.*?-->\s*', re.DOTALL)


class HarnessBundle(BaseModel):
    """One loaded harness version. Everything the pipeline reads at inference time."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    root: Path
    manifest: BundleManifest
    prompts: Mapping[str, str]
    sampling: SamplingConfig
    routing: RoutingConfig
    audio_pipeline: AudioPipelineConfig
    verification_sources: Mapping[str, str]

    def prompt(self, stage: str) -> str:
        """The prompt template for one stage."""
        try:
            return self.prompts[stage]
        except KeyError as exc:
            raise ValidationError(f'bundle has no prompt for stage {stage!r}') from exc

    def agree_fn(self) -> AgreeFn:
        """The bundle's cross-modal agreement function, checked and executed."""
        module = self.routing.verification.agreement_check
        return cast('AgreeFn', self._verification_function(module, AGREEMENT_FUNCTION))

    def rank_fn(self) -> RankFn:
        """The bundle's hypothesis ranker, checked and executed."""
        module = self.routing.verification.ranker
        return cast('RankFn', self._verification_function(module, RANKER_FUNCTION))

    def content_hash(self) -> str:
        """A stable hash over every file in the bundle."""
        return directory_hash(self.root)

    def editable_paths(self, stage: int) -> tuple[str, ...]:
        """Relative paths the Builder may replace at ``stage`` (ADR-0005)."""
        if stage not in STAGE_EDITABLE_GLOBS:
            raise ValidationError(f'unknown harness stage {stage!r}')
        found = {
            _relative(path, self.root)
            for pattern in STAGE_EDITABLE_GLOBS[stage]
            for path in self.root.glob(pattern)
            if path.is_file()
        }
        return tuple(sorted(found))

    def _verification_function(self, module: str, function: str) -> Callable[..., Any]:
        source = self.verification_sources.get(module)
        if source is None:
            raise ValidationError(f'bundle has no verification module {module!r}')
        path = str(self.root / VERIFICATION_DIR / f'{module}.py')
        return load_verification_function(source, function, path)


def load_bundle(root: Path) -> HarnessBundle:
    """Read and validate the bundle at ``root``."""
    missing = tuple(name for name in required_files() if not (root / name).is_file())
    if missing:
        raise ValidationError(f'harness bundle {root} is missing: {", ".join(missing)}')
    prompts = {
        name: _prompt_body(read_text(root / PROMPT_DIR / f'{name}.md'))
        for name in STAGE_NAMES
    }
    _check_placeholders(prompts)
    routing = RoutingConfig.model_validate(read_yaml(root / ROUTING_FILE))
    _check_routing(routing)
    sources = {
        name: read_text(root / VERIFICATION_DIR / f'{name}.py')
        for name in VERIFICATION_MODULES
    }
    return HarnessBundle(
        root=root,
        manifest=BundleManifest.model_validate(read_json(root / HARNESS_MANIFEST_FILE)),
        prompts=prompts,
        sampling=SamplingConfig.model_validate(read_yaml(root / SAMPLING_FILE)),
        routing=routing,
        audio_pipeline=AudioPipelineConfig.model_validate(
            read_yaml(root / AUDIO_PIPELINE_FILE)
        ),
        verification_sources=sources,
    )


def write_bundle_copy(
    bundle_root: Path, dest_root: Path, replacements: Mapping[str, str]
) -> Path:
    """Copy the bundle to ``dest_root`` and overwrite the given relative paths."""
    targets = {relative: _safe_target(dest_root, relative) for relative in replacements}
    shutil.copytree(
        bundle_root,
        dest_root,
        ignore=shutil.ignore_patterns(*_IGNORED_COPY_PATTERNS),
        dirs_exist_ok=True,
    )
    for relative, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(replacements[relative], encoding='utf-8')
    return dest_root


def export_schemas(bundle_root: Path) -> tuple[Path, ...]:
    """Write the evidence and output JSON schemas into the bundle."""
    pairs: tuple[tuple[str, type[BaseModel]], ...] = (
        (EVIDENCE_SCHEMA_FILE, EvidenceRecord),
        (OUTPUT_SCHEMA_FILE, HarnessOutput),
    )
    written = []
    for name, model in pairs:
        path = bundle_root / name
        write_json_atomic(path, model.model_json_schema())
        written.append(path)
    return tuple(written)


def _prompt_body(text: str) -> str:
    """A prompt without its HTML comments; the header comment documents, never renders."""
    return _COMMENT_RE.sub('', text).strip() + '\n'


def _placeholders_in(text: str) -> frozenset[str]:
    """Every ``{name}`` a template uses; doubled braces are literal text."""
    return frozenset(field for _, field, _, _ in Formatter().parse(text) if field)


def _check_placeholders(prompts: Mapping[str, str]) -> None:
    problems: list[str] = []
    for name, text in prompts.items():
        found = _placeholders_in(text)
        missing = tuple(p for p in REQUIRED_PLACEHOLDERS[name] if p not in found)
        unknown = tuple(p for p in sorted(found) if p not in SUPPORTED_PLACEHOLDERS)
        if missing:
            problems.append(f'{name}.md is missing {", ".join(missing)}')
        if unknown:
            problems.append(f'{name}.md uses unknown {", ".join(unknown)}')
    if problems:
        raise ValidationError('harness prompts: ' + '; '.join(problems))


def _check_routing(routing: RoutingConfig) -> None:
    unknown = tuple(stage for stage in routing.stages if stage not in STAGE_NAMES)
    if unknown:
        raise ValidationError(f'routing.yaml names unknown stages: {unknown}')
    modules = (routing.verification.agreement_check, routing.verification.ranker)
    missing = tuple(name for name in modules if name not in VERIFICATION_MODULES)
    if missing:
        raise ValidationError(f'routing.yaml names unknown modules: {missing}')


def _safe_target(dest_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts or not candidate.parts:
        raise PatchRejectedError(f'{relative!r} is outside the harness bundle')
    resolved = (dest_root / candidate).resolve()
    if dest_root.resolve() not in resolved.parents:
        raise PatchRejectedError(f'{relative!r} is outside the harness bundle')
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
