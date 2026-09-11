<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- Generic identity, clothing, background, gaze direction, and stationary posture are weak evidence about emotion.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- For an emotion request, a visible smile, smiling mouth, softened pleasant expression, amused expression, or clear positive facial change is direct evidence for the matching positive emotion label when that label is in the candidate list.
- A slight smile is still evidence. Do not require a broad grin, laughter, or energetic movement before considering `happy`, `happiness`, or `joy` when one of those exact labels is available.
- Do not let a larger number of generic gaze, clothing, background, or neutral-posture observations outweigh a direct smile. Those observations are not contradictions unless they explicitly show a competing affect.
- Use `neutral` when there is no reliable affective expression or when a genuine competing emotion is stronger; do not use it merely because the person is seated, speaking calmly, or otherwise still.
- For a sentiment request, use the valence of the complete record rather than treating a smile as the only possible cue. Do not use this smile rule to override clear negative words or distress cues.
- Keep every label exactly as written in the candidate list, including capitalization and spelling.
- Output JSON only: no prose before or after it, no code fence, no comments.
