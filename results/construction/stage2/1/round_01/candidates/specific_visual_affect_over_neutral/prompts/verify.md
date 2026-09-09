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

- Apply the following rule only when the candidate list contains the exact lowercase labels 'happy' and 'neutral'.
- A clearly observed smile, grin, laughter, warm facial expression, or positive social exchange is specific affective evidence for 'happy'. Do not discard that evidence merely because speech is steady, the volume is moderate, laughter is absent, or the audio opinion says neutral.
- Generic facts such as speaking, looking forward, sitting, an ordinary setting, moderate pitch, and no reported laughter are not positive evidence for 'neutral'. They are only absence of stronger evidence.
- Choose 'happy' when a concrete positive cue is prominent and there is no equally strong cue for a negative or fearful state. Choose 'neutral' when the record truly lacks a specific affective cue or contains a clear neutral expression.
- Treat audio_opinion as one weak voter. It must not override a specific visual cue or explicit positive wording. Do not infer happiness from a pleasant setting, clothing, or mere mouth movement.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
