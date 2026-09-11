<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Start from what was heard, then check it against what was seen.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Decide what the heard cues on their own point to, ignoring the earlier ranking.
2. Decide what the seen cues on their own point to.
3. Compare the two readings, then re-order and re-score the ranking.

Calibration rule for this reading:

- Apply the following rule only when the candidate list contains the exact labels 'Frustration' and 'Neutral'.
- Concrete signs of frustration include complaints, repeated disagreement, inability or blocked goals, conflict, unfairness, urgency, words such as hard, messed up, cannot, do not know what to do, or gestures that visibly escalate while discussing a problem.
- These semantic and behavioral cues can indicate Frustration even when the voice is quiet, slow, flat, or lacks laughter. Generic statements that the scene is calm, and the audio-model neutral vote, must not outweigh a specific obstacle or complaint.
- Rank 'Frustration' above 'Neutral' when a concrete obstacle or irritated conflict is prominent and there is no stronger cue for sadness, fear, or anger. Rank 'Neutral' only when the record is affectively unmarked and lacks a concrete problem signal.
- Do not infer Frustration from ordinary conversation, mere hand movement, sitting, or moderate prosody alone. Treat audio_opinion as one weak voter, not as ground truth.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
