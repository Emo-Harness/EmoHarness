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

Calibration for the happy | neutral | agree error pattern:

- When the exact candidate list contains an entry for happy and an entry for neutral, treat a visible smile, pleased expression, smiling eyes, or repeated positive facial affect as direct evidence for happy.
- A calm voice, ordinary conversation, lack of laughter, moderate pace, or lack of dramatic movement does not contradict happy and does not by itself support neutral.
- Generic observations such as standing, sitting, looking forward, clothing, furniture, and the setting are not emotional support for neutral. Do not let a long list of such observations outweigh one or more direct affective cues.
- The audio-model opinion is one fallible voter. A neutral opinion must not overrule clear repeated positive facial evidence. Use the exact candidate spelling from the list, including capitalization.
- Select neutral only when the record lacks a stronger direct positive or negative affective cue, or when a visible positive cue is clearly fleeting and contradicted by the dominant expression.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
