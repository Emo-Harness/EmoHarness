<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Start from what was heard, then check it against what was seen. The earlier ranking may be wrong, especially when the modalities disagree.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Decide what the heard cues on their own point to, ignoring the earlier ranking. Separate the speaker's affect from the literal topic and from the fact that a question was asked.
2. Decide what the seen cues on their own point to. Identify the person with the strongest displayed affect rather than averaging all people or background action.
3. Compare the two readings, then re-order and re-score the ranking.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- A sustained smile, warm engagement, relaxed positive expression, laughter, or repeated positive behavior supports a happiness-like label when that label is available. Do not erase it merely because the voice is calm or the transcript contains a question.
- A surprise-like label requires a reaction to something unexpected: abrupt change, startled movement, sudden exclamation, widened eyes or open mouth together with a reactive context. A single question word, high pitch, or open mouth during ordinary speech is not by itself enough.
- Distinguish sustained positive affect from a brief surprise cue. If the visible positive expression persists while the heard cue is only a short question or exclamation, lower surprise and preserve the supported positive reading. If the visible reaction is abrupt and the smile is incidental, do the reverse.
- When modalities disagree, report `modalities_agree` as `false`, lower confidence rather than pretending certainty, and retain supported readings from both sides. Do not allow disagreement itself to manufacture a new label.
- Prefer direct observed affect over generic appearance, topic sentiment, or the neutral demeanor of another person. Never use `t1` as evidence.
- Use only ids that appear in the record; drop any id you cannot find there. Do not cite an id twice in one list.
- Keep plausible supported alternatives rather than deleting them, and order the hypotheses from most to least likely after adjudication.
- Output JSON only: no prose before or after it, no code fence, no comments.

DETERMINISTIC CHECK (context, not citable evidence):
{verification_json}
Use this check only to locate disagreements. Cite observed cues, never the check itself or `t1`.
