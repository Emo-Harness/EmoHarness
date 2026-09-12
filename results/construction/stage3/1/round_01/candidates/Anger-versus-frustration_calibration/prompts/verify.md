<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Separate direct affective evidence from incidental content, then compare what was heard with what was seen.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Read the heard cues on their own. Use transcript meaning only when it expresses an attitude, appraisal, complaint, approval, threat, distress, blame, hostility, or other affective stance; do not treat an unrelated factual topic as emotional evidence. Also use prosody such as pitch, loudness, speed, pauses, and vocal variation. Explicit profanity, insults, threats, hostile commands, direct blame, abrasive wording, repeated confrontational challenges, and forceful demands are stronger anger evidence than a merely difficult situation.
2. Read the seen cues on their own. Give priority to direct facial and bodily affect cues, expression changes, gaze, posture, gestures, and temporal change. Furrowed brows, glaring, aggressive gestures, forceful movement, confrontational leaning, and sustained hostile expression can corroborate anger. Do not let clothing, setting, ordinary sitting, or generic mouth movement outweigh direct expression cues.
3. Decide what each modality points to, using the candidate labels exactly as written. Distinguish nearby negative states carefully: frustration is primarily blocked goals, obstacles, inability, repeated problems, confusion, or strained effort; anger is outwardly hostile, blaming, insulting, threatening, abrasive, or confrontational. A person can be both frustrated and angry, so retain both when supported, but rank anger higher when the language or expression is explicitly directed against someone or something. Do not call every complaint anger, and do not soften explicit hostility into frustration merely because the face is restrained.
4. Compare the two readings. When they disagree, retain the meaningful alternatives, mark `modalities_agree` false, and lower confidence rather than automatically selecting the less intense label or the first earlier hypothesis. Resolve close cases using explicitness, target-directed hostility, cue directness, specificity, persistence across time, and cross-modal corroboration.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis should be supported by actual ids and should not receive credit merely because it was ranked first earlier. Do not award anger or frustration credit for generic scene details; cite the specific words, prosody, gestures, or expressions that distinguish them.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- Do not invent a translation, event, facial expression, or emotional cue.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- Keep support and contradiction lists free of duplicate ids.
- Output JSON only: no prose before or after it, no code fence, no comments.
