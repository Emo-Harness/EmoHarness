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

1. Read the heard cues on their own. For sentiment, decide whether the words express a stance toward a subject or interaction. Approval, praise, gratitude, encouragement, endorsement, and favorable evaluation support positive valence. Criticism, complaint, rejection, regret, dismissal, threat, insult, concern, and unfavorable evaluation support negative valence. A factual mention of a person, event, plot, place, or topic is not automatically emotional. For emotion, a subdued reading can be supported by a coherent combination of low or flat pitch, quiet or slowed delivery, long pauses, broken or uncertain wording, and a persistently withdrawn vocal manner. One generic pitch, loudness, speed, or pause tag alone is not decisive.
2. Read the seen cues on their own. Prioritize direct facial and bodily affect, expression changes, gaze, posture, gestures, and temporal change. Sadness is supported by a specific and persistent combination such as downcast gaze or head, withdrawn or tearful expression, reduced movement, slumped posture, visible distress, or a change toward subdued affect. Neutral means genuinely affect-free observation. Clothing, setting, ordinary sitting, generic mouth movement, and the number of scene descriptions are incidental. Do not make Neutral win merely because it can cite more incidental visual items.
3. Decide what each modality points to using the candidate labels exactly as written. For sentiment, classify the expressed stance and valence, not merely the apparent mood of the speaker. For emotion, distinguish sadness from neutral using the specificity and persistence of subdued facial, bodily, vocal, and temporal cues. A factual topic alone is not sadness, but uncertainty, resignation, withdrawal, and sustained low-energy delivery can be meaningful when they occur together. Do not turn a single calm or slow cue into sadness without corroboration.
4. Compare the readings. When they disagree, set `modalities_agree` to false and retain the meaningful alternatives. Resolve the disagreement by this priority: explicit evaluative wording, direct affective expression, specificity of the cue, persistence or temporal change, then cross-modal corroboration. Do not automatically select Neutral because the face is calm, and do not automatically follow the modality with more evidence items. Incidental scene descriptions and ordinary posture cannot cancel a specific, repeated subdued cue.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis must receive credit only from ids that genuinely support it. Neutral means no clear positive or negative stance or affect, not merely factual wording, ordinary posture, or plentiful incidental items. When several mutually consistent subdued cues support Sadness, give them cumulative weight; when only one generic cue exists, retain Neutral. Preserve a competing hypothesis when the evidence is genuinely ambiguous, but put the best-supported reading first.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- Do not invent a translation, event, facial expression, or emotional cue.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences rather than hiding the conflict.
- Keep support and contradiction lists free of duplicate ids.
- Output JSON only: no prose before or after it, no code fence, no comments.
