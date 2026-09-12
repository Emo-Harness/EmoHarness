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

Conflict resolution policy:

- A disagreement is not permission to choose neutral. Resolve it using the strength and specificity of the cues.
- When the noun is emotion and both Frustration (or its exact case-sensitive equivalent) and Neutral are candidates, Frustration is favored by evidence of a blocked goal, difficulty, confusion about what to do, objection, reproach, repeated questioning, interruption, urgency, tense negotiation, or visibly strained/insistent interaction.
- In that same Frustration-versus-Neutral choice, generic sitting, looking, speaking, ordinary clothing, a plain setting, or a calm resting posture are weak evidence and must not outweigh specific problem-solving or interpersonal tension cues.
- Slow, quiet, flat, or hesitant delivery alone is not sufficient for Frustration, but those cues strengthen Frustration when paired with uncertainty, inability, disagreement, repeated requests, or a visibly strained exchange.
- Neutral should be first only when the record contains no specific affective or interactional cue; do not use a neutral label merely because the face is controlled or the task is conversational.
- If one modality supplies explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation, and the other modality supplies only ordinary appearance or generic activity, favor the affective negative reading over neutral.
- A calm face or flat delivery does not cancel negative semantic content; conversely, a single ambiguous facial cue does not cancel clearly neutral language.
- Treat repeated furrowing, withdrawal, tense gestures, negative wording, and consistent prosody as stronger evidence than generic visual inventory.
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
