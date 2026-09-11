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

Calibration for the Frustration | Neutral | agree error pattern:

- Apply this rule when the question asks for emotion and the candidate list contains the exact labels `Frustration` and `Neutral`.
- Treat an explicit blocked goal, inability, complaint, disagreement, repeated correction or question, problem-focused wording, exasperated gesture, furrowed brow, or tense/forceful movement as positive evidence for `Frustration`, even when the voice is quiet, slow, or flat.
- Lack of laughter, steady volume, calm surroundings, ordinary sitting, and an audio-model suggestion of `neutral` are not evidence against frustration. The audio opinion is one fallible voter, not a veto.
- Prefer `Neutral` only when the record explicitly describes a neutral expression or lacks any meaningful problem, conflict, or exasperation cue. Do not let a generic calm description outrank a concrete frustration cue.
- If `Frustration` is supported by the record but is missing from the earlier ranking, add it from the candidate list with the relevant evidence ids before re-ranking.
- Keep `Sadness` separate: blocked-goal or irritated problem-solving cues favor `Frustration`, while loss, grief, crying, or explicitly downcast affect favor sadness.

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
