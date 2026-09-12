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
- Neutral is appropriate only when no specific affective or evaluative cue is present. Sitting, looking, ordinary clothing, a plain room, calm posture, and ordinary speaking are weak evidence and must not outweigh a specific cue.
- When the noun is emotion and both Anger and Frustration are candidates, rank Anger above Frustration when the record contains direct blame, accusation, reproach, hostile challenge, profanity, threats, confrontational rhetorical questions, an abrasive interpersonal target, or repeated insistence that another person is at fault.
- Frustration is the better label for blocked effort, confusion, delay, difficulty, inability, repeated questioning, urgent problem-solving, resignation about an obstacle, or a demanding request when there is no clear hostile or accusatory target. A controlled face does not cancel these cues.
- Do not call ordinary problem-solving Anger merely because it is urgent, but do not call it Neutral merely because the face is composed. Repeated questions, contradiction, insistence, "can't", "don't know", "why", "how", or difficulty can be specific frustration evidence in context.
- A controlled face does not cancel explicit hostile or reproachful language. Conversely, generic gesturing or fast speech alone does not establish Anger or Frustration.
- If one modality supplies explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation, and the other modality supplies only ordinary appearance or generic activity, favor the affective reading over neutral.
- When the evidence remains genuinely balanced, retain both competing labels with close confidences and place neutral first only if neither side contains a specific affective cue.
- For sentiment, favor positive or negative when the content contains a specific favorable or unfavorable evaluation; use neutral only for genuinely factual or descriptive content.
- When the evidence contains positive arousal, distinguish Excited from calm Happiness; when it contains a sustained smile or warmth without arousal, distinguish Happiness from Neutral.
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
