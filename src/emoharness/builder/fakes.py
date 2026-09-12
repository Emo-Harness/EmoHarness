"""The Builder double: proposals a loop can run without a Builder model.

:class:`emoharness.builder.client.FakeBuilder` answers a packet from a canned
list; this module writes that list. Every proposal is one valid prompt edit, so
a ``--fake`` construction round exercises the real patch, gate, and provenance
paths rather than short-circuiting them.
"""

from __future__ import annotations

from typing import Final

from emoharness.builder.client import FakeBuilder
from emoharness.builder.schema import BuilderProposal, CandidatePatch, PatchFile
from emoharness.common.constants import MAX_ROUNDS_PER_STAGE

FAKE_BUILDER_ID: Final = 'fake-builder'
#: The one bundle file the fake edits; it is writable at stages 2 and 3 alike.
EDITABLE_PROMPT: Final = 'prompts/extract_visual.md'


def _patch_text(stage: int, index: int) -> str:
    return (
        'You are given a short video clip. Describe only what can be seen in it.\n\n'
        'Return one JSON object with the keys `face`, `body`, `scene`, and '
        '`temporal_change`.\n\n'
        'Rules:\n\n'
        '- Report observations, not conclusions.\n'
        f'- Do not state the {{noun}} displayed in the video.\n'
        f'- Revision {index} (stage {stage}): also note whether the framing changes.\n'
        '- Output JSON only: no prose, no code fence, no comments.\n'
    )


def fake_builder(stage: int, rounds: int = MAX_ROUNDS_PER_STAGE) -> FakeBuilder:
    """A Builder that proposes one valid prompt edit per round, and never repeats.

    The edited path is allowed at stages 2 and 3 alike, and the replacement keeps
    the ``{noun}`` placeholder its stage requires, so every candidate loads.
    """
    proposals = tuple(
        BuilderProposal(
            summary=f'stage {stage} round {index}',
            # The last canned round declines to continue, so a --fake stage
            # exercises the autonomous-stop path as well as the round cap.
            continue_search=index + 1 < rounds,
            stop_reason='' if index + 1 < rounds else 'no directions left',
            candidates=(
                CandidatePatch(
                    name=f'stage{stage}_round{index}_visual',
                    targeted_cluster='(none) | (none) | agree',
                    rationale='the visual stage misses changes in framing',
                    files=(
                        PatchFile(
                            path=EDITABLE_PROMPT,
                            content=_patch_text(stage, index),
                        ),
                    ),
                ),
            ),
        )
        for index in range(rounds)
    )
    return FakeBuilder(proposals, model_id=FAKE_BUILDER_ID)
