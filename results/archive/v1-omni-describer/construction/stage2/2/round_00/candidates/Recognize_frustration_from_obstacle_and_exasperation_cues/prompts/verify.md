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

Calibration for the Frustration | Neutral | agree error pattern:

- When the exact candidate list contains Frustration and Neutral, look for an obstacle or blocked goal together with exasperation: repeated questions, objections, inability statements, disagreement, complaints, corrective language, urgent phrasing, abrupt emphasis, faster speech, raised or variable pitch, or emphatic hand gestures.
- These cues support Frustration even when the speaker is seated, the volume is moderate, the face is difficult to read, or the audio-model opinion says neutral. Calm delivery describes arousal, not necessarily emotional valence or the presence of frustration.
- Generic observations such as sitting still, looking toward another person, clothing, room layout, and the absence of laughter are not evidence for Neutral. Do not count those observations as a substitute for affective evidence.
- Prefer Frustration over Sadness when the dominant signal is an active obstacle, protest, irritation, or effort to change a situation. Prefer Sadness only when loss, resignation, grief, or subdued negative affect is more prominent.
- Use Neutral only when the record has no meaningful obstacle or exasperated reaction. Preserve the exact label spelling and capitalization from the candidate list.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
