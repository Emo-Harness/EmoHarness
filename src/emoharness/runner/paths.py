"""Where every artifact lives, and the rule that none is overwritten silently.

``ProjectPaths.from_root`` derives the whole layout from one root directory so a
``--fake`` run can write a throwaway tree without touching the repository. The
committed inputs (``configs/`` and the Stage 1 ``harness/`` bundle) are read from
the root when it carries its own copies and from the source checkout otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from emoharness.common.constants import HARNESS_BUNDLE_DIR
from emoharness.common.errors import ValidationError

BENCHMARK_MANIFEST_FILE: Final = 'mme_emotion.jsonl'
EXTERNAL_MANIFEST_FILE: Final = 'external.jsonl'
SUBSET_FILE: Final = 'subset_1000.json'
EXTERNAL_SPLITS_FILE: Final = 'external_splits.json'
BENCHMARK_CHECKSUM_FILE: Final = 'mme_emotion.sha256'
BENCHMARK_REPORT_FILE: Final = 'mme_emotion_report.json'
EXTERNAL_REPORT_FILE: Final = 'external_report.json'
LEAKAGE_REPORT_FILE: Final = 'leakage_report.json'
LEDGER_FILE: Final = 'ledger.jsonl'
FREEZE_FILE: Final = 'freeze.json'
PREREGISTRATION_FILE: Final = 'preregistration.md'
BUILDER_CONFIG_FILE: Final = 'teacher_gpt_5_6_luna.yaml'
TEACHER_CONFIG_FILE: Final = 'teacher_gpt_5_6_luna.yaml'
CALL_CACHE_DIR: Final = 'calls'
CALIBRATION_DIR: Final = 'calibration'
FAKE_DIR: Final = 'fake'


def source_root() -> Path:
    """The checkout that ships ``configs/`` and ``harness/``, or the working directory."""
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / 'configs').is_dir():
        return candidate
    return Path.cwd().resolve()


def _prefer(local: Path, fallback: Path) -> Path:
    return local if local.is_dir() else fallback


class ProjectPaths(BaseModel):
    """Every directory and well-known file the runner touches."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    root: Path
    configs: Path
    bundle: Path
    data: Path
    raw: Path
    derived: Path
    manifests: Path
    splits: Path
    checksums: Path
    results: Path
    raw_predictions: Path
    judge: Path
    construction: Path
    metrics: Path
    ledger: Path

    @classmethod
    def from_root(cls, root: Path, source: Path | None = None) -> ProjectPaths:
        """Derive the layout from ``root``; committed inputs come from ``source``."""
        base = Path(root).expanduser().resolve()
        checkout = (source or source_root()).resolve()
        data = base / 'data'
        results = base / 'results'
        return cls(
            root=base,
            configs=_prefer(base / 'configs', checkout / 'configs'),
            bundle=_prefer(base / HARNESS_BUNDLE_DIR, checkout / HARNESS_BUNDLE_DIR),
            data=data,
            raw=data / 'raw',
            derived=data / 'derived',
            manifests=data / 'manifests',
            splits=data / 'splits',
            checksums=data / 'checksums',
            results=results,
            raw_predictions=results / 'raw_predictions',
            judge=results / 'judge',
            construction=results / 'construction',
            metrics=results / 'metrics',
            ledger=results / LEDGER_FILE,
        )

    @property
    def directories(self) -> tuple[Path, ...]:
        """Every directory ``ensure`` creates."""
        return (
            self.data,
            self.raw,
            self.derived,
            self.manifests,
            self.splits,
            self.checksums,
            self.results,
            self.raw_predictions,
            self.judge,
            self.construction,
            self.metrics,
            self.call_cache,
            self.docs,
        )

    def ensure(self) -> None:
        """Create every artifact directory; existing directories are left alone."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def benchmark_manifest(self) -> Path:
        return self.manifests / BENCHMARK_MANIFEST_FILE

    @property
    def benchmark_report(self) -> Path:
        return self.manifests / BENCHMARK_REPORT_FILE

    @property
    def external_manifest(self) -> Path:
        return self.manifests / EXTERNAL_MANIFEST_FILE

    @property
    def external_report(self) -> Path:
        return self.manifests / EXTERNAL_REPORT_FILE

    @property
    def external_splits(self) -> Path:
        return self.splits / EXTERNAL_SPLITS_FILE

    @property
    def subset(self) -> Path:
        return self.splits / SUBSET_FILE

    @property
    def benchmark_checksums(self) -> Path:
        return self.checksums / BENCHMARK_CHECKSUM_FILE

    @property
    def leakage_report(self) -> Path:
        return self.checksums / LEAKAGE_REPORT_FILE

    @property
    def call_cache(self) -> Path:
        return self.derived / CALL_CACHE_DIR

    @property
    def preprocess_cache(self) -> Path:
        return self.derived

    @property
    def calibration(self) -> Path:
        return self.results / CALIBRATION_DIR

    @property
    def freeze_record(self) -> Path:
        return self.results / FREEZE_FILE

    @property
    def fake_raw(self) -> Path:
        return self.data / FAKE_DIR

    @property
    def fake_results(self) -> Path:
        return self.results / FAKE_DIR

    @property
    def docs(self) -> Path:
        return self.root / 'docs'

    @property
    def preregistration(self) -> Path:
        return self.docs / PREREGISTRATION_FILE

    @property
    def emotion_lexicon(self) -> Path:
        return self.configs / 'emotion_lexicon.txt'

    @property
    def preprocess_config(self) -> Path:
        return self.configs / 'preprocess.yaml'

    @property
    def ladder_config(self) -> Path:
        return self.configs / 'experiments' / 'ladder.yaml'

    @property
    def sources_config(self) -> Path:
        return self.configs / 'data' / 'sources.yaml'

    @property
    def builder_config(self) -> Path:
        return self.configs / 'builder' / BUILDER_CONFIG_FILE

    @property
    def teacher_config(self) -> Path:
        return self.configs / 'target' / TEACHER_CONFIG_FILE

    @property
    def proxy_judge_config(self) -> Path:
        return self.configs / 'builder' / 'proxy_judge.yaml'

    def predictions(self, row: str, item_set: str) -> Path:
        """Raw predictions for one row on one item set."""
        return self.raw_predictions / row / f'{_slug(item_set)}.jsonl'

    def harness_outputs(self, row: str, item_set: str) -> Path:
        """Per-item ``HarnessOutput`` records, written beside the predictions."""
        return self.raw_predictions / row / f'{_slug(item_set)}.outputs.jsonl'

    def judge_run(self, row: str, item_set: str) -> Path:
        """Directory holding one row's official-judge inputs and outputs."""
        return self.judge / row / _slug(item_set)

    def stage_dir(self, stage: int, replicate: int) -> Path:
        """Provenance directory of one construction stage and replicate."""
        return self.construction / f'stage{stage}' / str(replicate)


def _slug(item_set: str) -> str:
    """A filesystem-safe name for an item set such as ``external:val``."""
    return item_set.replace(':', '_').replace('/', '_')


def guard_write(path: Path, force: bool) -> Path:
    """Return ``path``, refusing to overwrite an existing artifact without ``force``."""
    if path.exists() and not force:
        raise ValidationError(f'{path} already exists; pass --force to overwrite it')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
