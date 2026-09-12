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

1. Read the heard cues on their own. Use transcript meaning only when it expresses an attitude, appraisal, complaint, approval, threat, distress, insistence, helplessness, or other affective stance; do not treat an unrelated factual topic as emotional evidence. Prosody such as pitch, loudness, speed, pauses, vocal variation, and clipped or hesitant delivery is evidence about display, but do not infer a strong emotion from a single generic prosody tag.
2. Read the seen cues on their own. Prioritize direct facial and bodily affect cues, expression changes, gaze, posture, gestures, and temporal change. Clothing, setting, ordinary sitting, looking, speaking, and generic mouth movement are incidental. A calm face does not by itself prove Neutral, and a large collection of incidental observations must not outweigh a specific strained or distressed cue.
3. Decide what each modality points to, using the candidate labels exactly as written. For an emotion question, identify the affect visibly or vocally displayed rather than merely the subject matter. In particular, recognize frustration when a person is dealing with an obstacle, inability, repeated problem, blocked goal, insistent correction, strained questioning, or resigned difficulty, even if the person remains seated or has a restrained expression. Do not call a plainly factual, untroubled exchange frustrated merely because it discusses a problem.
4. Compare the two readings. When they disagree, retain the meaningful alternatives and mark modalities_agree false. Do not automatically select Neutral because the face is restrained, because the transcript is short, or because many generic visual items support the earlier reading. Resolve close cases using cue directness, specificity, persistence across time, and cross-modal corroboration; a specific affective stance can outweigh many incidental cues.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis should be supported by actual ids and should not receive credit merely because it was ranked first earlier. Neutral requires an absence of meaningful affective evidence, not merely the presence of ordinary posture or scene descriptions.

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
