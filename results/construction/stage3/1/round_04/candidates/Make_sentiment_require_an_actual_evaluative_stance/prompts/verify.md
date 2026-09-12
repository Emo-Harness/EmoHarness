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

1. Read the heard cues on their own. For sentiment, first identify the target, person, action, or proposition toward which the speaker is taking a stance. Approval, praise, gratitude, endorsement, favorable evaluation, or encouragement directed at that target support positive valence. Criticism, complaint, rejection, regret, dismissal, threat, insult, concern, or unfavorable evaluation directed at that target support negative valence. A factual mention of a person, event, plot, place, or topic is not automatically emotional. A fragment must be judged only from the words that are present. Prosody can corroborate affect, but one generic pitch, loudness, speed, or pause tag is not decisive.
2. Read the seen cues on their own. Prioritize direct facial and bodily affect, expression changes, gaze, posture, gestures, and temporal change. Clothing, setting, ordinary sitting, generic mouth movement, and the number of scene descriptions are incidental. Do not make Neutral win merely because it can cite more incidental visual items.
3. Decide what each modality points to using the candidate labels exactly as written. For sentiment, classify the expressed stance and valence, not merely the apparent mood of the speaker. An explicit appraisal remains meaningful even when calmly delivered or accompanied by a neutral face. Conversely, negative subject matter alone is not negative sentiment.
4. Compare the readings. When they disagree, set `modalities_agree` to false and retain the meaningful alternatives. Resolve the disagreement by this priority: explicit evaluative wording, direct affective expression, specificity of the cue, persistence or temporal change, then cross-modal corroboration. Do not automatically select Neutral because the face is calm, and do not automatically follow the modality with more evidence items.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis must receive credit only from ids that genuinely support it. Neutral means no clear positive or negative stance, not merely factual wording, ordinary posture, or plentiful incidental items. Preserve a competing hypothesis when the evidence is genuinely ambiguous, but put the best-supported reading first.

For sentiment, do not equate positive valence with any pleasant-sounding word. Generic advertising slogans, broad exhortations, religious or moral instructions, a speaker reporting that another person is happy, quoted praise, conversational agreement, and factual descriptions remain Neutral unless the speaker is clearly evaluating a target favorably. Likewise, a serious topic is not Negative without criticism or an unfavorable stance. Once a clear target-directed appraisal is present, use the corresponding valence even if the delivery is calm.

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
