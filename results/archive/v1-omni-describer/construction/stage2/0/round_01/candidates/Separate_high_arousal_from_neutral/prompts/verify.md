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

Calibration for the Excited | Neutral | agree error pattern:

- Apply this rule when the question asks for emotion and the candidate list contains the exact labels `Excited` and `Neutral`.
- Support `Excited` when the record combines elevated arousal with anticipation, urgency, emphatic exclamations, rapid or forceful delivery, repeated enthusiasm, or energetic movement.
- High arousal, fast speech, emphatic questions, words about something amazing or eagerly desired, and animated gestures are stronger evidence than the absence of laughter.
- Do not let a low-confidence neutral audio opinion or a generic statement that pitch and volume are moderate override clear high-arousal semantic or behavioral cues.
- Use `Neutral` when the record is genuinely routine, low-arousal, and lacks anticipation, urgency, emphasis, or energetic behavior. If `Excited` is missing from the earlier ranking, add it from the candidate list when the record supports it.
- Copy whichever exact candidate-list spelling is selected; never output the calibration wording itself.

Work in this order:

1. Decide what the heard cues on their own point to, ignoring the earlier ranking.
2. Decide what the seen cues on their own point to.
3. Compare the two readings, then re-order and re-score the ranking.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- An opinion item is one voter, not an observation.
- Output JSON only: no prose before or after it, no code fence, no comments.
