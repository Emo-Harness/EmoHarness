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

Conflict-resolution protocol:

- First mark each modality's strongest cue as direct affect, specific semantic evaluation, or generic description.
- Direct affect includes a furrowed brow, downcast gaze with visible tension, distress, hostility, agitation, a threatening or rejecting expression, or a clear change into such a state.
- Generic audio includes bargaining, pricing, routine instructions, factual narration, ordinary questions, and transactional wording without explicit evaluation.
- For sentiment requests, when direct negative visual affect conflicts with generic transactional or factual audio, rank the negative entry first and retain neutral as the alternative. Do not let the number of generic audio items override cue specificity.
- Conversely, when explicit criticism, rejection, threat, distress, or clearly negative delivery is heard while the visual description is merely ordinary, rank the negative entry first and retain neutral as the alternative.
- If both modalities contain direct cues that disagree, keep both close, lower both confidences, and set `modalities_agree` to false.
- Neutral is appropriate only when neither modality contains a specific competing affective cue.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- Do not resolve a conflict by defaulting to neutral. First identify whether either side contains a direct affective cue.
- For sentiment requests, distinguish merely transactional or factual words from negative evaluation. Generic negotiation, narration, or an incomplete factual transcript is not strong evidence against negative facial or bodily affect.
- Direct negative visual cues such as a furrowed brow, downcast gaze with tense posture, visible distress, hostile expression, or agitated movement should outweigh a merely neutral or transactional audio reading when they are the clearest affective display.
- Conversely, explicit criticism, rejection, threat, distress, or clearly negative delivery in the heard cues should outweigh generic neutral appearance descriptions when the visual side has no direct competing affect.
- If one side is generic and the other is direct, rank the direct side first while preserving the generic-side hypothesis as an alternative. If both sides are direct and disagree, keep both close and lower confidence.
- Do not infer negative sentiment from a neutral proper name, ordinary action, or neutral setting alone.
- Re-score using the strength and specificity of cues, not the raw number of generic evidence items.
- Output JSON only: no prose before or after it, no code fence, no comments.

