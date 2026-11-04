"""Where every run of the paper's pipeline writes, and where the analysis reads it.

The canonical layout, which ``experiments.paper.pipeline`` writes::

    data/manifests/mme_emotion.jsonl        the 5,577-item benchmark minus the construction items
    data/manifests/external_v3.jsonl        the isolated construction split (399 / 394 / 129)
    results/main/<student>/                 Video (L0), Audio (L1), Shared (L3) and the
                                            GPT-5.6 Luna MHE harness bundles/<student> (L5)
        raw_predictions/<row>/clean.jsonl
    results/mhe/<builder>/<student>/        one MHE construction (the Builder's run)
        construction/stage3/0/              loop_result.json, run_config.json, round_*/, final/
        L5/raw_predictions/L5/clean.jsonl   that run's evolved harness on the evaluation items
    results/transfer/manifests/             IntentQA / Social-IQ 2.0 manifests (with question types)
    results/emotion_context/<student>/      IntentQA emotion context, GPT-5.6 Luna bundles:
        context/<dataset>/                  one context per video
        qa_{none,asr,emoctx}/<dataset>/     Frames / Audio / Audio + Emotion
        builder_<builder>/{context,qa_emoctx}/<dataset>/   the same for another Builder
    results/paper/                          every table, figure's data and in-text number

The GPT-5.6 Luna harnesses the paper evaluates are the frozen ``bundles/<student>``,
so their MHE row is the main evaluation's ``L5``. A re-run of a Luna construction
writes under ``results/mhe/gpt5_6_luna/<student>/`` like any other Builder's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Builder:
    """One Builder configuration: the model that proposes harness edits."""

    key: str
    label: str
    #: file under configs/builder/
    config: str
    #: its MHE harnesses are the frozen bundles/<student>
    frozen: bool = False


BUILDERS: Final[dict[str, Builder]] = {
    'gpt5_6_luna': Builder('gpt5_6_luna', 'GPT-5.6 Luna', 'v3_full.yaml', frozen=True),
    'gpt6_luna': Builder('gpt6_luna', 'GPT-6 Luna', 'v3_full_gpt6_luna.yaml'),
    'gpt6_1_sol': Builder('gpt6_1_sol', 'GPT-6.1 Sol', 'v3_full_gpt6_1_sol.yaml'),
}
#: The two Builders the paper analyses (8-test Holm families). GPT-6 Luna is the
#: third configuration of Appendix A (12-test families with all three).
PAPER_BUILDERS: Final = ('gpt5_6_luna', 'gpt6_1_sol')
ALL_BUILDERS: Final = tuple(BUILDERS)
FROZEN_BUILDER: Final = 'gpt5_6_luna'
#: Short names a command line may use for a Builder.
BUILDER_ALIASES: Final = {
    'luna': 'gpt5_6_luna',
    'sol': 'gpt6_1_sol',
    'gpt6-luna': 'gpt6_luna',
}


def builder_key(name: str) -> str:
    """A Builder's key from its key or short name (``luna``, ``sol``, ``gpt6-luna``)."""
    key = BUILDER_ALIASES.get(name, name)
    if key not in BUILDERS:
        names = ', '.join((*BUILDER_ALIASES, *BUILDERS))
        raise ValueError(f'unknown Builder {name!r}; choose from {names}')
    return key


#: The paper's conditions and the ladder rows that produce them.
CONDITION_ROWS: Final = {'video': 'L0', 'audio': 'L1', 'shared': 'L3', 'mhe': 'L5'}
#: The evaluation item set: the benchmark minus the clips seen in construction.
ITEM_SET: Final = 'clean'
#: The isolated construction manifest every MHE run reads.
CONSTRUCTION_MANIFEST: Final = 'external_v3.jsonl'
INTENTQA: Final = 'intentqa_test'


@dataclass(frozen=True)
class Layout:
    """Every path the pipeline writes and the analysis reads, under one root."""

    root: Path = ROOT
    #: the checkout that ships bundles/ (differs from ``root`` in a --fake run)
    checkout: Path = ROOT

    # --- data -------------------------------------------------------------
    @property
    def manifests(self) -> Path:
        return self.root / 'data/manifests'

    @property
    def benchmark_manifest(self) -> Path:
        return self.manifests / 'mme_emotion.jsonl'

    @property
    def construction_manifest(self) -> Path:
        return self.manifests / CONSTRUCTION_MANIFEST

    # --- MME-Emotion: Video, Audio, Shared, frozen MHE ---------------------
    def main_dir(self, student: str) -> Path:
        """``EMOHARNESS_RESULTS_DIR`` of the four conditions for one student."""
        return self.root / 'results/main' / student

    def row_predictions(self, student: str, row: str) -> Path:
        """Raw predictions of one ladder row on the evaluation items."""
        return self.main_dir(student) / 'raw_predictions' / row / f'{ITEM_SET}.jsonl'

    # --- MHE runs ----------------------------------------------------------
    def build_dir(self, builder: str, student: str) -> Path:
        """``EMOHARNESS_RESULTS_DIR`` of one Builder's construction for one student."""
        return self.root / 'results/mhe' / builder / student

    def construction_dir(self, builder: str, student: str) -> Path:
        return self.build_dir(builder, student) / 'construction/stage3/0'

    def mhe_bundle(self, builder: str, student: str) -> Path:
        """The MHE harness the paper evaluates for this Builder and student.

        GPT-5.6 Luna's is ``bundles/<student>``. Another Builder's is the newest
        one: its construction's ``final`` once a run has built it (the harness
        ``evolve`` evaluates), else the shipped ``bundles/<builder>/<student>``
        (GPT-6.1 Sol), so every stage reads the same harness.
        """
        if BUILDERS[builder].frozen:
            return self.checkout / 'bundles' / student
        final = self.construction_dir(builder, student) / 'final'
        shipped = self.checkout / 'bundles' / builder / student
        if final.is_dir() or not shipped.is_dir():
            return final
        return shipped

    def mhe_eval_dir(self, builder: str, student: str) -> Path:
        """``EMOHARNESS_RESULTS_DIR`` of a rebuilt harness's evaluation."""
        return self.build_dir(builder, student) / 'L5'

    def mhe_predictions(self, builder: str, student: str) -> Path:
        """The MHE row of Table 1 for this Builder and student."""
        if BUILDERS[builder].frozen:
            return self.row_predictions(student, 'L5')
        return (
            self.mhe_eval_dir(builder, student)
            / 'raw_predictions/L5'
            / f'{ITEM_SET}.jsonl'
        )

    # --- IntentQA ------------------------------------------------------------
    @property
    def transfer_manifests(self) -> Path:
        return self.root / 'results/transfer/manifests'

    @property
    def emotion_context(self) -> Path:
        return self.root / 'results/emotion_context'

    def emotion_context_step(
        self,
        student: str,
        stage: str,
        condition: str | None,
        dataset: str,
        builder: str | None = None,
    ) -> Path:
        """One step's directory; ``builder`` None or the frozen one is the bundle run."""
        base = self.emotion_context / student
        if builder is not None and not BUILDERS[builder].frozen:
            base = base / f'builder_{builder}'
        leaf = 'context' if stage == 'context' else f'qa_{condition}'
        return base / leaf / dataset

    # --- outputs -------------------------------------------------------------
    @property
    def paper(self) -> Path:
        return self.root / 'results/paper'


LAYOUT: Final = Layout()
