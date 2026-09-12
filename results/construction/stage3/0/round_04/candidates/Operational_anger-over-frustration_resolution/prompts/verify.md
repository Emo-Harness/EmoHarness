<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Start from the heard cues, then check them against what was seen.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Read the transcript and prosody on their own. Identify the speaker's interpersonal target, evaluation, and action tendency before looking at the earlier ranking.
2. Read the visible cues on their own. Separate generic activity from facial tension, confrontational orientation, withdrawal, fear, sadness, or positive affect.
3. Compare the modalities and then reorder and rescore the hypotheses. The top entry must be the best overall interpretation, not the label with the most generic supporting ids.

Mandatory conflict policy when the noun is emotion:

- When both Anger and Frustration are candidates, rank Anger above Frustration if the record contains direct blame, accusation, reproach, hostile challenge, profanity, threats, a confrontational rhetorical question, an abrasive interpersonal target, or repeated insistence that another person is at fault.
- Examples of anger evidence include questioning why another person did something, saying that the other person's behavior is unacceptable, accusing them of ignoring or causing something, or using What the hell, What's wrong with that, or similar hostile challenges. A calm or controlled face and flat pitch do not cancel those cues.
- Rank Frustration above Anger for blocked effort, confusion, delay, difficulty, or ordinary problem-solving without a hostile or accusatory target. Fast speech, gesturing, and urgency alone do not establish Anger.
- Do not let the earlier ranking decide this distinction. Explicit interpersonal language is more specific than a count of generic visual cues.
- If the evidence contains explicit affective language, a clear complaint, criticism, rejection, threat, distress, bargaining tension, or negative evaluation while the other modality is generic, favor the specific affective reading over Neutral.
- If the two modalities genuinely disagree, set `modalities_agree` to false, lower confidence, retain the competing labels, and still resolve the ranking using the specificity policy above.
- Do not treat `t1` as evidence for any label. Do not infer an emotion from the task word.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard and seen readings point to the same entry, otherwise `false`.
- `hypotheses`: the reordered list, with each object containing `label`, `support`, `contradiction`, and `confidence`.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- Do not cite `t1` as support or contradiction for an emotion or sentiment.
- When one modality supplies no specific cue, do not manufacture disagreement from its generic observations.
- Output JSON only: no prose before or after it, no code fence, no comments.
