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
- Generic facts such as sitting, looking, clothing, room setting, or ordinary speaking are weak evidence. They must not outweigh specific affective cues.
- When both `Excited` and `Happiness` are candidates, use `Happiness` for calm pleasure, warmth, approval, or a relaxed smile without strong activation. Use `Excited` when positive emotion is accompanied by high arousal or anticipation, especially multiple cues such as fast or emphatic speech, variable or high pitch, exclamations, repeated intensifiers, broad or rapid gestures, open enthusiastic posture, laughter, celebration, eager anticipation, or visibly animated movement.
- A smile can support both labels. It supports Excited rather than ordinary Happiness when the smile is paired with clear energetic, anticipatory, emphatic, or animated behavior. Do not let a long list of generic visual observations force Happiness above a specific high-arousal cue.
- Distinguish Excited from Surprise: Surprise is primarily a sudden reaction to unexpected information, often shown by transient wide eyes, raised eyebrows, or an open mouth. Positive anticipation, celebration, eager agreement, or sustained energetic engagement favors Excited.
- When the noun is emotion and both Anger and Frustration are candidates, rank Anger above Frustration when the record contains direct blame, accusation, reproach, hostile challenge, profanity, threats, confrontational rhetorical questions, an abrasive interpersonal target, or repeated insistence that another person is at fault.
- Frustration is the better label for blocked effort, confusion, delay, or difficulty without a hostile or accusatory target. Do not call ordinary problem-solving Anger merely because it is urgent.
- A controlled face does not cancel explicit hostile or reproachful language. Conversely, generic gesturing or fast speech alone does not establish Anger.
- If one modality supplies explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation, and the other modality supplies only ordinary appearance or generic activity, favor the affective negative reading over neutral.
- When the evidence remains genuinely balanced, retain both competing labels with close confidences and place neutral first only if neither side contains a specific affective cue.
- For sentiment, favor positive or negative when the content contains a specific favorable or unfavorable evaluation; use neutral only for genuinely factual or descriptive content.
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
