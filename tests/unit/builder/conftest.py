"""Fixtures for the Builder subpackage. Everything is synthetic and local."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from emoharness.builder.clusters import FailureRecord
from emoharness.builder.schema import BuilderProposal, CandidatePatch, CandidateScores
from emoharness.data.prompt_template import ParsedPrompt, Quantifier
from emoharness.harness.schema import (
    AudioEvidence,
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    VerificationResult,
    VisualEvidence,
)

LABELS = ('anger', 'happiness', 'sadness', 'neutral')
BUNDLE_TEXT: Mapping[str, str] = {
    'prompts/extract_visual.md': 'List the visible cues you can see.\n',
    'prompts/generate_hypotheses.md': 'Rank the candidate labels by support.\n',
    'routing.yaml': 'order: [visual, audio]\n',
    'sampling.yaml': 'frames: 8\n',
    'verification/cross_modal.py': 'def agree(a: int, b: int) -> bool:\n    return a == b\n',
    'manifest.json': '{"version": "stage-1"}\n',
}


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    """A tiny Stage 1 style harness bundle on disk."""
    root = tmp_path / 'harness'
    for path, content in BUNDLE_TEXT.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    (root / 'verification' / '__pycache__').mkdir()
    (root / 'verification' / '__pycache__' / 'x.pyc').write_bytes(b'\x00')
    return root


@pytest.fixture
def bundle_files() -> Mapping[str, str]:
    return dict(BUNDLE_TEXT)


@pytest.fixture
def make_record() -> Callable[..., FailureRecord]:
    def _make(
        example_id: str,
        gold: tuple[str, ...] = ('anger',),
        predicted: tuple[str, ...] = ('sadness',),
        conflict: bool = False,
        keyframes: tuple[str, ...] = (),
    ) -> FailureRecord:
        return FailureRecord(
            example_id=example_id,
            gold=gold,
            predicted=predicted,
            conflict=conflict,
            evidence_json=json.dumps({'item': example_id, 'cue': 'lowered gaze'}),
            stage_outputs={'hypotheses': '{"coarse":"low"}', 'verification': '{}'},
            keyframe_paths=keyframes,
        )

    return _make


@pytest.fixture
def make_output() -> Callable[..., HarnessOutput]:
    def _make(
        example_id: str,
        labels: tuple[str, ...] = ('anger',),
        agree: bool = True,
    ) -> HarnessOutput:
        hypothesis = Hypothesis(label=labels[0], confidence=0.7)
        return HarnessOutput(
            example_id=example_id,
            labels=labels,
            steps=('setting', 'face', 'voice', 'synthesis', 'decision'),
            response_text=f'<answer>{labels[0]}</answer>',
            evidence=EvidenceRecord(
                visual=VisualEvidence(face=('brow lowered',)),
                audio=AudioEvidence(speech_present=True, transcript='hello there'),
                text=ParsedPrompt(
                    quantifier=Quantifier.MOST_PROMINENT,
                    noun='emotion',
                    label_set=LABELS,
                ),
            ),
            hypotheses=HypothesisSet(coarse='negative', hypotheses=(hypothesis,)),
            verification=VerificationResult(
                modalities_agree=agree, ranked=(hypothesis,)
            ),
        )

    return _make


@pytest.fixture
def make_scores() -> Callable[..., CandidateScores]:
    def _make(
        name: str,
        recognition: float = 0.5,
        reasoning: float | None = 0.5,
        regression_rate: float = 0.0,
        mean_calls: float = 3.0,
        p95_latency_s: float = 5.0,
    ) -> CandidateScores:
        return CandidateScores(
            name=name,
            recognition=recognition,
            reasoning=reasoning,
            regression_rate=regression_rate,
            mean_calls=mean_calls,
            p95_latency_s=p95_latency_s,
        )

    return _make


@pytest.fixture
def make_patch() -> Callable[..., CandidatePatch]:
    def _make(
        name: str = 'candidate',
        path: str = 'prompts/extract_visual.md',
        content: str = 'List the visible cues, then the audible cues.\n',
        cluster: str = 'anger | sadness | agree',
    ) -> CandidatePatch:
        return CandidatePatch(
            name=name,
            targeted_cluster=cluster,
            rationale='the harness overlooks the audible cues',
            files=({'path': path, 'content': content},),
        )

    return _make


@pytest.fixture
def make_proposal(
    make_patch: Callable[..., CandidatePatch],
) -> Callable[..., BuilderProposal]:
    def _make(*names: str) -> BuilderProposal:
        patches = tuple(
            make_patch(name=name, content=f'Prompt revision {name}.\n')
            for name in names
        )
        return BuilderProposal(candidates=patches, summary='round summary')

    return _make


@pytest.fixture
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test key every OpenAI-backed client reads; never a real one."""
    monkeypatch.setenv('EMOHARNESS_TEST_OPENAI_KEY', 'sk-test')
