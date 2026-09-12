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

Important emotion policy for happiness:

- When the noun is emotion and happy and neutral are candidates, explicit positive facial or interactional cues should outweigh a long list of generic observations.
- A visible smile, broad smile, laughter or amusement, warm friendly engagement, playful interaction, celebratory behavior, or a positive affective statement supports happy. Track whether the cue persists or appears at the end rather than dismissing it as incidental.
- Neutral requires the absence of a specific affective cue; sitting, looking, speaking, clothing, a room setting, or ordinary movement are weak evidence and should not cancel a visible smile or amused engagement.
- Do not infer happy from speech alone when it is purely factual and there is no positive vocal or visual cue. Conversely, do not require an extreme grin or laughter before recognizing mild happiness.
- If an earlier ranking omitted happy but the record contains a specific smile, amusement, or friendly-affect cue, add happy with that evidence id and rank it against neutral.

General conflict policy:

- A disagreement is not permission to choose neutral. Resolve it using the strength and specificity of the cues.
- If one modality supplies explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation, and the other modality supplies only ordinary appearance or generic activity, favor the affective reading over neutral.
- Treat generic facts such as sitting, looking, speaking, clothing, or room setting as weak evidence. Treat repeated facial affect, meaningful gestures, affective wording, and consistent prosody as stronger evidence.
- When the evidence remains genuinely balanced, retain both competing labels with close confidences and place neutral first only if neither side contains a specific affective cue.
- The ranked list is consumed downstream, so its first entry must be the best overall interpretation after this comparison, not merely the label favored by the more numerous generic cues.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences, but still resolve the ranking using the conflict policy; do not hide the conflict by dropping a hypothesis.
- Do not treat the task-word item `t1` as evidence for any emotion or sentiment.
- Output JSON only: no prose before or after it, no code fence, no comments.
