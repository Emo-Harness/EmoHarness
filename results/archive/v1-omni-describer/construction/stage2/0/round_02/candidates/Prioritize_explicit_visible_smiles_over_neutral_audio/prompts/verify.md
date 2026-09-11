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

DECISIVE HAPPY-OVER-NEUTRAL RULE:

- When the exact candidate labels `happy` and `neutral` are present, an explicit visual cue such as `smile`, `smiling`, `smiling mouth`, `smiling eyes`, `pleasant expression`, or a clearly sustained positive facial expression must put `happy` above `neutral`.
- This remains true when the audio transcript is ordinary, the prosody is flat or calm, the audio opinion votes neutral, or the smile is described as slight. Those cues do not contradict a visible smile.
- Treat the visual smile as direct evidence and the audio opinion as a single fallible opinion item, not as a veto. Do not retain `neutral` as the top label merely because it has more generic audio support.
- If the earlier ranking omitted `happy`, add a hypothesis for the exact candidate-list label and support it with the valid visual ids that describe the smile. Remove no supported alternative solely because modalities disagree.

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
