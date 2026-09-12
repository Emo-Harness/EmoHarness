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

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above and use only ids present in the record.
- A diagnostic smile, smiling eyes, pleasant amusement, or warm engagement is meaningful evidence for the supplied happiness label, even if the person is otherwise still or the scene is ordinary. Do not count generic posture and background items as stronger simply because there are more of them.
- Choose `Neutral` only when no reliable positive facial, vocal, or semantic cue is present. An open mouth or raised brows without positive affect may support surprise, but a smile that persists or emerges by the end supports happiness.
- Reconsider the temporal change and the focal person before accepting the earlier neutral ranking. The task wording itself is never affective evidence.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- Output JSON only: no prose before or after it, no code fence, no comments.
