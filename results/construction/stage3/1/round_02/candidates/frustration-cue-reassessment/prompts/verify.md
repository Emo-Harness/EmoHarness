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

1. Read the heard cues on their own. First determine whether the transcript expresses a stance rather than merely naming a topic or event. Criticism, complaint, dismissal, rejection, regret, threat, insult, unfavorable evaluation, or concern can support negative sentiment; approval, praise, gratitude, encouragement, or favorable evaluation can support positive sentiment. A factual sentence about a person, movie, place, or event is not automatically emotional. Also use prosody such as pitch, loudness, speed, pauses, and vocal variation without treating one generic prosody tag as decisive.
2. Read the seen cues on their own. Give priority to direct facial and bodily affect cues, expression changes, gaze, posture, gestures, and temporal change. Do not let clothing, setting, ordinary sitting, or generic mouth movement outweigh direct expression cues. Ordinary visual evidence is not evidence for Neutral merely because it is plentiful.
3. For emotion questions, explicitly test frustration separately from Neutral. Frustration may be low-arousal and need not look angry: repeated questioning or correction, disbelief about another person's action, an unresolved disagreement, inability or blocked-goal statements, requests to explain or comply, impatient insistence, tense or exasperated gestures, and persistent difficulty can all be direct frustration cues. Do not discard these cues merely because the voice is quiet, flat, or polite. Conversely, a simple factual question, ordinary instruction, routine explanation, or generic gesture without conflict or blocked difficulty is not enough for frustration.
4. Decide what each modality points to, using the candidate labels exactly as written. For a sentiment question, classify the speaker's expressed stance and valence toward the interaction or subject, not just the emotional appearance of the speaker. A negative appraisal may remain negative even when delivered calmly and with a neutral face; conversely, a negative subject matter alone is not negative sentiment. For an emotion question, identify the affect visibly or vocally displayed.
5. Compare the two readings. When they disagree, retain the meaningful alternatives, mark `modalities_agree` false, and lower confidence rather than automatically selecting Neutral or following the first earlier hypothesis. Resolve close cases using the explicitness of the appraisal, cue directness, specificity, persistence across time, and cross-modal corroboration.
6. Re-order and re-score the hypotheses after this second reading. A hypothesis should be supported by actual ids and should not receive credit merely because it was ranked first earlier. Neutral requires no clear positive or negative stance, not merely factual wording, ordinary posture, or a large number of incidental visual items.

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
