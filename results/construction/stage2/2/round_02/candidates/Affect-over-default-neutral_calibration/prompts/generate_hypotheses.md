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

Emotion-reading calibration:

- When the noun is emotion, identify the strongest affect actually displayed, rather than classifying the subject matter or defaulting to Neutral because the person is seated, calm-looking, or speaking factually.
- Neutral requires that no candidate has a clear affective cue. Generic appearance, ordinary setting, mouth movement, and routine conversation are weak evidence and must not outweigh a specific emotional cue.
- Frustration is supported by blocked goals, inability, repeated unresolved questions, complaints about a problem, demands caused by an obstacle, strained problem-solving, or agitation directed at an obstacle.
- Distinguish frustration from anger: frustration centers on obstruction, difficulty, inability, or exasperation; anger requires clearer hostility, attack, accusation, threat, or deliberate aggression.
- Slow, flat, quiet, or sparse speech can accompany subdued frustration or distress, but prosody alone is insufficient; combine it with the transcript or visible behavior when possible.
- Use the most prominent person's displayed state and the strongest specific cues, not the raw number of neutral descriptive items.

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
- Do not use the calibration terms as labels unless the exact term is present in the candidate list; every emitted label must be copied exactly from that list.
- Output JSON only: no prose before or after it, no code fence, no comments.

