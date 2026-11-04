"""Small opt-in request transformations for the v5 mechanism screen.

No model client, tools, labels, examples or scoring data are owned here. Audio
 timing is deliberately unavailable: existing caches do not retain word times.
"""

from __future__ import annotations

import json

from emoharness.harness.schema import EvidenceRecord, HypothesisSet, PerceivedInputs


def source_table(record: EvidenceRecord) -> str:
    rows = []
    for item in record.items:
        source = 'observed_visual_summary'
        if item.modality == 'text':
            source = 'question_context_not_evidence'
        elif item.modality == 'audio':
            source = (
                'cached_asr'
                if item.cue.startswith('speech: ')
                else 'cached_prosody'
                if item.cue.startswith('prosody: ')
                else 'cached_audio_availability'
            )
        rows.append(
            {
                'id': item.id,
                'source': source,
                'cue': item.cue,
                'start_seconds': None,
                'end_seconds': None,
            }
        )
    return json.dumps(
        {
            'source_tagged_cues': rows,
            'audio_timing': 'unavailable_in_existing_cache',
            'speech_present': record.audio.speech_present,
            'language': record.audio.language,
            'transcript': record.audio.transcript,
            'prosody': list(record.audio.prosody),
            'question': record.text.model_dump(mode='json'),
        },
        ensure_ascii=False,
        indent=2,
    )


def confidence_margin(hypotheses: HypothesisSet) -> float:
    values = [h.confidence for h in hypotheses.hypotheses]
    return max(0.0, values[0] - values[1]) if len(values) > 1 else 0.0


def top_two(hypotheses: HypothesisSet) -> HypothesisSet:
    return hypotheses.model_copy(update={'hypotheses': hypotheses.hypotheses[:2]})


def decision_frames(inputs: PerceivedInputs, maximum: int) -> tuple[str, ...]:
    frames = inputs.frame_paths
    if len(frames) <= maximum:
        return frames
    if maximum == 1:
        return (frames[len(frames) // 2],)
    return tuple(
        frames[round(i * (len(frames) - 1) / (maximum - 1))] for i in range(maximum)
    )


TOP_TWO_CONTRACT = """Compare the two leading hypotheses explicitly: which
observed cue distinguishes them, and which evidence contradicts each? Return
the SAME JSON schema with revised hypotheses/support/contradiction/confidence.
Do not merely count citations, force a label, or treat question wording as
evidence. Only cite existing evidence IDs; do not invent observation IDs."""
RAW_FRAMES_CONTRACT = """The attached images are fixed-tool frames from the
original clip, in chronological order. Use them to check the visual summary
before choosing hypotheses. A summary is fallible. If a cited summary cue is
not supported by the frames, do not count it as evidence. Do not invent new
evidence IDs or pretend these frames contain audio."""
REVIEW_VIDEO_CONTRACT = """Review the ORIGINAL video using the same student
model. Recheck the conflicting/uncertain visual evidence against the cached
audio transcript and prosody. The visual summary may be wrong. Return the same
hypothesis JSON contract; cite only existing evidence IDs that remain valid,
and do not infer audio from video pixels."""
