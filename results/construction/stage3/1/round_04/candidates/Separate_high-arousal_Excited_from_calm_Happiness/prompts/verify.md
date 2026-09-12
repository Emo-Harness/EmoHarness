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

1. Read the heard cues on their own. For sentiment, decide whether the words express a stance toward a subject or interaction. Approval, praise, gratitude, encouragement, endorsement, and favorable evaluation support positive valence. Criticism, complaint, rejection, regret, dismissal, threat, insult, concern, and unfavorable evaluation support negative valence. A factual mention of a person, event, plot, place, or topic is not automatically emotional. A fragment must be judged only from the words that are present. Prosody can corroborate affect, but one generic pitch, loudness, speed, or pause tag is not decisive.
2. Read the seen cues on their own. Prioritize direct facial and bodily affect, expression changes, gaze, posture, gestures, and temporal change. Clothing, setting, ordinary sitting, generic mouth movement, and the number of scene descriptions are incidental. Do not make Neutral win merely because it can cite more incidental visual items.
3. Decide what each modality points to using the candidate labels exactly as written. For sentiment, classify the expressed stance and valence, not merely the apparent mood of the speaker. For emotion, identify the affect visibly or vocally displayed. An explicit appraisal remains meaningful even when calmly delivered or accompanied by a neutral face. Conversely, negative subject matter alone is not negative sentiment.
4. Compare the readings. When they disagree, set `modalities_agree` to false and retain the meaningful alternatives. Resolve the disagreement by this priority: explicit evaluative wording, direct affective expression, specificity of the cue, persistence or temporal change, then cross-modal corroboration. Do not automatically select Neutral because the face is calm, and do not automatically follow the modality with more evidence items.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis must receive credit only from ids that genuinely support it. Neutral means no clear positive or negative stance, not merely factual wording, ordinary posture, or plentiful incidental items. Preserve a competing hypothesis when the evidence is genuinely ambiguous, but put the best-supported reading first.

When both Happiness and Excited are candidates, distinguish pleasant valence from activation. Happiness is appropriate for a calm smile or agreeable content without notable intensity. Excited means positive high activation: emphatic or repeated enthusiastic wording, broad or escalating smiles, animated gestures, fast or variable delivery, raised energy, or visible temporal intensification. Do not let a larger number of incidental visual items make Happiness win over specific activation cues; conversely, do not call a single quiet smile Excited without an activation cue.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- Do not invent a translation, event, facial expression, or emotional cue.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences rather than hiding the conflict.
- Keep support and contradiction lists free of duplicate ids.
- Keep support only for cues that genuinely support that hypothesis; do not use the quantity of incidental items as evidence.
- Output JSON only: no prose before or after it, no code fence, no comments.
