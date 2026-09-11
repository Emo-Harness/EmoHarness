<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Use the heard and seen evidence, but do not treat an audio-model opinion as fact.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Read the direct heard evidence on its own. Give priority to the words, meaningful vocal events, and specific prosody. A neutral or low-arousal `audio_opinion` is only one fallible voter.
2. Read the direct seen evidence on its own. Give priority to facial expressions, posture, gestures, and visible change. Generic facts such as sitting, looking forward, or speaking are not evidence for neutral.
3. Compare the readings, then re-order and re-score the ranking using the strongest specific cues rather than the number of redundant evidence items.

Use these distinctions when they fit the candidate list:

- Frustration is supported by obstacles or inability, repeated complaints or objections, phrases such as cannot, do not know how, hard, messed up, or what am I going to do, and agitated or emphatic gestures. It does not require shouting.
- Sadness is better supported by loss, resignation, withdrawal, low-energy distress, or mournful cues. Do not turn every negative statement into sadness.
- Anger is better supported by hostility, accusation, insults, harshness, or openly confrontational delivery. Frustration and anger may both remain hypotheses when the distinction is genuinely unclear.
- Fear or anxiety is supported by threat, danger, inability to feel safe, urgent uncertainty, alarm, or visibly worried reactions.
- Happiness or excitement is supported by smiles, laughter, enthusiasm, positive affect, or energetic movement, not merely by a neutral conversation.
- Neutral is appropriate only when no listed specific emotion has meaningful support. Calm pitch, moderate volume, no laughter, no crying, and no background noise are not positive evidence of neutral.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- Do not cite an audio-opinion item as if it were an observed vocal or visual cue.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- Do not let a long list of generic audio-description ids defeat one or two direct, specific emotional cues.
- Output JSON only: no prose before or after it, no code fence, no comments.
