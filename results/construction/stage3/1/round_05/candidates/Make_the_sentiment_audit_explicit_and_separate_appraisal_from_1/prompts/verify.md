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

1. Read the heard cues on their own. For sentiment, first perform an appraisal audit on the actual words: ask whether the speaker approves, praises, recommends, enjoys, appreciates, encourages, endorses, or favorably evaluates something, or instead criticizes, rejects, complains about, regrets, dismisses, threatens, insults, or unfavorably evaluates something. These explicit stance words support the corresponding positive or negative candidate even when spoken calmly. A factual mention of a person, event, plot, place, product, or topic is not automatically emotional. A generic instruction, question, advertisement setting, or energetic delivery without an appraisal is not by itself positive. A fragment must be judged only from the words that are present. Prosody can corroborate affect, but one generic pitch, loudness, speed, or pause tag is not decisive.
2. Read the seen cues on their own. Prioritize direct facial and bodily affect, expression changes, gaze, posture, gestures, and temporal change. Clothing, setting, ordinary sitting, generic mouth movement, and the number of scene descriptions are incidental. Do not make Neutral win merely because it can cite more incidental visual items, and do not make positive win merely because a person smiles while discussing a neutral topic.
3. Decide what each modality points to using the candidate labels exactly as written. For sentiment, classify the expressed stance and valence, not merely the apparent mood of the speaker. Explicit praise, approval, favorable evaluation, enjoyment, gratitude, encouragement, or recommendation remains positive even when understated or accompanied by a calm face. Explicit criticism, rejection, complaint, regret, dismissal, insult, threat, or unfavorable evaluation remains negative. Factual narration with no clear appraisal is neutral. For emotion, identify the affect visibly or vocally displayed.
4. Compare the readings. When they disagree, set `modalities_agree` to false and retain the meaningful alternatives. Resolve the disagreement by this priority: explicit evaluative wording, direct affective expression, specificity of the cue, persistence or temporal change, then cross-modal corroboration. Do not automatically select Neutral because the face is calm, and do not automatically follow the modality with more evidence items. Do not convert subject matter into sentiment: a serious topic is not automatically negative, and a positive-sounding topic is not automatically positive without an expressed stance.
5. Re-order and re-score the hypotheses after this second reading. A hypothesis must receive credit only from ids that genuinely support it. Neutral means no clear positive or negative stance, not merely factual wording, ordinary posture, or plentiful incidental items. Preserve a competing hypothesis when the evidence is genuinely ambiguous, but put the best-supported reading first.

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
