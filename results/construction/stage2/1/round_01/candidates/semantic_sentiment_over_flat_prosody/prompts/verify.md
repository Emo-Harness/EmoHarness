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

- Apply the following rule only when the candidate list contains the exact lowercase labels 'positive' and 'neutral'.
- Sentiment is the overall affective direction of the clip, including the meaning and interpersonal tone of speech, not just vocal arousal.
- Specific praise, encouragement, gratitude, optimism, affectionate or welcoming language, laughter, smiling, and a clearly pleasant interaction support 'positive'. A calm delivery does not erase positive semantic content.
- Steady pace, moderate volume, no laughter, ordinary posture, and an audio opinion of neutral are weak or non-diagnostic evidence; they do not by themselves support 'neutral'.
- Select 'positive' when at least one concrete positive semantic or behavioral cue is present and no comparably strong negative cue is present. Select 'neutral' when the content and observed behavior are genuinely factual or affectively unmarked. Do not infer positive sentiment from a merely pleasant setting or from a quoted word that is unrelated to the speaker's attitude.
- Treat audio_opinion as one weak voter and never let it overrule explicit transcript meaning or a specific visual cue.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
