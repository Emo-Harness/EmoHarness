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

Important sentiment policy:

- When the noun is sentiment and positive and neutral are candidates, judge the valence of the message, not only the speaker's facial intensity or the factual topic.
- Favorable evaluation, praise, approval, encouragement, gratitude, relief, benefit, hope, amusement, a positive comparison, or language saying that something makes sense, is funny, useful, good, better, meaningful, or fortunate supports positive.
- Words such as thankfully, funny, great, good, meaningful, or giving credit are specific positive evidence even when the surrounding sentence is descriptive or the face is controlled.
- Neutral is appropriate for purely factual description with no favorable or unfavorable evaluation. Do not downgrade positive sentiment merely because the speaker is calm, seated, or discussing a serious or ordinary topic.
- Do not infer positive from a pleasant setting, ordinary politeness, generic speaking, or an uninformative smile alone. If the transcript contains explicit favorable evaluation, it outweighs generic visual neutrality.
- If an earlier ranking omitted positive but the record contains a specific favorable evaluation, add positive as a hypothesis using the relevant evidence ids.

General conflict policy:

- A disagreement is not permission to choose neutral. Resolve it using the strength and specificity of the cues.
- If one modality supplies explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation, and the other modality supplies only ordinary appearance or generic activity, favor the affective reading over neutral.
- Treat generic facts such as sitting, looking, speaking, clothing, or room setting as weak evidence. Treat explicit evaluative wording, repeated emotional wording, expressive facial cues, and consistent prosody as stronger evidence.
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
