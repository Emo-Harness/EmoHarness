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

- Apply the following rule only when the candidate list contains the exact lowercase labels 'negative' and 'neutral'.
- Specific negative content includes loss, failure, harm, threat, danger, conflict, criticism, complaint, distress, hostility, rejection, or an explicitly unfavorable evaluation. Such content supports 'negative' even when it is spoken calmly.
- Moderate pitch, steady pace, no laughter, ordinary posture, and an audio-model neutral vote are not evidence that the sentiment is neutral; they only describe delivery or missing cues.
- Rank 'negative' above 'neutral' when a concrete unfavorable semantic or behavioral cue is prominent and no stronger positive interpretation is present. Rank 'neutral' when the content is factual and affectively unmarked, not merely because arousal is low.
- Do not infer negative sentiment from a dark setting, serious clothing, or isolated words without their context. Treat audio_opinion as one weak voter and never let it overrule explicit transcript meaning or specific visible behavior.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
