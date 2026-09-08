"""Project-wide constants.

Values that experiments may change live in ``configs/``; this module holds facts
about the benchmark protocol and fixed design decisions (see docs/adr/).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

PACKAGE_NAME: Final = 'emoharness'

# Public MME-Emotion prompt template (paper Appendix C.2). The label separator is
# an assumption; the paper does not print it.
OFFICIAL_PROMPT_TEMPLATE: Final = (
    'As an emotion recognition expert, identify {quantifier} {noun} displayed in '
    'the video. Candidate labels: {labels}. Please think step by step. Enclose your '
    'final answer within <answer></answer> tags.'
)
LABEL_SEPARATOR: Final = ', '
ANSWER_OPEN: Final = '<answer>'
ANSWER_CLOSE: Final = '</answer>'

# Official judge protocol (QwenAudio/MME-Emotion scripts).
JUDGE_STEP_MODEL: Final = 'gpt-4.1-2025-04-14'
JUDGE_SCORE_MODEL: Final = 'gpt-4o-2024-11-20'
JUDGE_FRAME_INTERVAL_SECONDS: Final = 1.0
JUDGE_MAX_FRAMES: Final = 10
COT_ALPHA: Final = 0.5
# Scores live on 0..1; gates, margins, and reports quote them in points.
POINTS_PER_UNIT: Final = 100.0
GOLD_LABEL_SEPARATORS: Final[tuple[str, ...]] = (',', ';')
OFFICIAL_SYNONYMS: Final[Mapping[str, str]] = {
    'angry': 'anger',
    'disgusted': 'disgust',
    'happy': 'happiness',
    'sad': 'sadness',
    'surprised': 'surprise',
}

# Output controller (ADR-0010).
STEP_TITLES: Final[tuple[str, ...]] = (
    'Setting and context',
    'Facial and bodily cues',
    'Vocal and verbal cues',
    'Cross-modal synthesis and strongest alternative',
    'Decision',
)
NUM_STEPS: Final = len(STEP_TITLES)
MULTILABEL_CAP: Final = 3
#: How a numbered step is introduced, wherever one is written, counted, or rated.
STEP_PREFIX: Final = 'Step'

# Per-item Target-call budget (ADR-0011).
CALL_BUDGET_EXPECTED: Final = 4
CALL_BUDGET_HARD_CAP: Final = 6
RETRIES_PER_STAGE: Final = 1

# Pre-registered subset (ADR-0013).
SUBSET_SIZE_PER_TASK: Final = 125
SUBSET_SEED: Final = 20260907

# Harness bundle layout and Builder permissions (ADR-0005).
HARNESS_BUNDLE_DIR: Final = 'harness'
HARNESS_MANIFEST_FILE: Final = 'manifest.json'
STAGE_EDITABLE_GLOBS: Final[Mapping[int, tuple[str, ...]]] = {
    1: (),
    2: ('prompts/*.md', '*.yaml'),
    3: ('prompts/*.md', '*.yaml', 'verification/*.py'),
}
MAX_CANDIDATES_PER_ROUND: Final = 4
MAX_ROUNDS_PER_STAGE: Final = 8
EARLY_STOP_ROUNDS: Final = 2
REGRESSION_RATE_LIMIT: Final = 0.05
REASONING_GATE_FLOOR_POINTS: Final = 2.0
GATE_SAMPLE_SIZE: Final = 200

# Leakage thresholds (ADR-0003).
PHASH_HAMMING_THRESHOLD: Final = 8
TRANSCRIPT_JACCARD_THRESHOLD: Final = 0.8
TRANSCRIPT_MIN_WORDS: Final = 5

# Video protocol (ADR-0011).
VIDEO_FPS: Final = 2.0
VIDEO_MIN_FRAMES: Final = 4
VIDEO_MAX_FRAMES: Final = 32
