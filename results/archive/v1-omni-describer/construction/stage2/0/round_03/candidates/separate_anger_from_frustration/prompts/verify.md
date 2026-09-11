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

Calibration for the happy | neutral | agree error pattern:

- Apply this rule when the question asks for emotion and the candidate list contains the exact labels `happy` and `neutral`.
- An explicit visible smile, smiling mouth, smiling eyes, or a sustained pleasant facial expression is positive evidence for `happy`, even when the smile is slight.
- Do not treat no laughter, flat speech, moderate volume, or an audio-model suggestion of `neutral` as evidence against a visible smile. The audio opinion is only one fallible voter.
- Prefer `neutral` only when the visible expression is explicitly neutral or when there is no meaningful positive facial or behavioral cue. If `happy` is missing from the earlier ranking, add it from the candidate list when the record supports it.
- Copy whichever exact candidate-list spelling is selected; never output the calibration wording itself.

Calibration for the Anger | Frustration | agree error pattern:

- Apply this rule when the question asks for emotion and the candidate list contains the exact labels `Anger` and `Frustration`.
- Favor `Anger` when the record has hostile, abrasive, harsh, forceful, insulting, confrontational, or clearly agitated language or delivery, especially when paired with raised or loud speech, sharp tone, rapid emphatic delivery, or aggressive gestures.
- Favor `Frustration` when the record centers on a blocked goal, inability, repeated problem, confusion, or exasperated effort without clear hostility toward another person.
- High pitch, fast speech, or negative valence alone does not establish anger; use the content and tone quality as well as prosody. Conversely, a flat or moderate voice does not rule out anger when the wording is hostile or dismissive.
- If `Anger` is supported by concrete hostile or abrasive cues but is missing from the earlier ranking, add it from the candidate list with those evidence ids before re-ranking.
- Do not let an audio-model suggestion decide between these labels by itself; it is one fallible voter.

Work in this order:

1. Decide what the heard cues on their own point to, ignoring the earlier ranking.
2. Decide what the seen cues on their own point to.
3. Compare the two readings, then re-order and re-score the ranking.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
