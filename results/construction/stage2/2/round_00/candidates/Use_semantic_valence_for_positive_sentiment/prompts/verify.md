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

Calibration for the positive | neutral | agree error pattern:

- When the exact candidate list contains positive and neutral, evaluate sentiment valence separately from arousal. A calm, steady, moderate, or flat delivery can still express positive sentiment.
- Treat the actual transcript as evidence when it contains encouragement, praise, gratitude, hope, affection, celebration, approval, enjoyment, or another clearly favorable meaning. Do not replace that meaning with neutral merely because there is no laughter or strong prosodic variation.
- Positive visual cues such as smiling, friendly engagement, welcoming interaction, or clearly favorable on-screen content support positive. Generic posture and setting facts do not support neutral.
- The audio-model opinion is a fallible voter, not a veto. A neutral opinion should not outweigh clear positive lexical meaning or multiple positive visual cues.
- Choose neutral only when the record is genuinely affectively unmarked or when positive and negative evidence cancel. Copy the exact candidate spelling and capitalization from the list.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
