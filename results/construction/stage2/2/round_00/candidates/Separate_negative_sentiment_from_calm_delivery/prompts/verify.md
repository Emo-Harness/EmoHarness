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

Calibration for the negative | neutral | agree error pattern:

- When the exact candidate list contains negative and neutral, evaluate sentiment meaning separately from vocal arousal. A steady, moderate, or emotionally restrained voice can still carry negative sentiment.
- Treat transcript meaning as evidence when it describes harm, threat, conflict, criticism, failure, loss, dislike, distress, a problem, or an unfavorable judgment. Calm narration of negative content remains negative.
- Negative visual cues such as distress, tense confrontation, disapproval, or a clearly adverse event support negative. Generic posture, ordinary setting details, and lack of laughter do not support neutral.
- The audio-model opinion is one fallible voter. Neutral prosody or a neutral opinion must not erase explicit negative lexical content or coherent negative context.
- Choose neutral only when the content and visible behavior are genuinely affectively unmarked, not merely because the delivery is calm. Copy the exact candidate spelling and capitalization from the list.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
