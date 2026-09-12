<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`. `label` must be one entry from the list above copied exactly; the two id lists may contain only ids from the record; `confidence` is between 0 and 1.

Decision policy:

- First determine whether this is an emotion question or a sentiment question. For emotion, identify the state expressed by the person or interaction, not merely the subject matter.
- If `Frustration` is a candidate, treat blocked goals, difficulty, complaints, irritation, impatience, repeated questioning, demanding language, helpless inability, or problem-solving under an obstacle as direct evidence for `Frustration`.
- Frustration can be restrained: a neutral-looking face, slow speech, quiet voice, or seated posture does not cancel clear evidence of a problem, thwarted goal, complaint, or inability. Distinguish it from `Neutral` by whether something is being resisted, failed, questioned, or experienced as difficult.
- Prefer `Neutral` only when the record is genuinely unmarked and contains no obstacle, complaint, irritation, urgency, or difficulty cue. Do not let many generic appearance observations outweigh a diagnostic utterance or gesture.
- Compare `Frustration` with nearby labels rather than treating every negative state as frustration; use sadness for loss or low mood, anger for hostility or attack, and fear for threat or danger when those are better supported.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
