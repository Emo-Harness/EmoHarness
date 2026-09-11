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
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- A label should be supported by actual visual or audio evidence. The `t1` item identifies the question type; it is not evidence for Neutral.
- For an emotion question, distinguish a genuinely neutral interaction from frustration. Frustration is supported by blocked or difficult goals, inability, complaints, resistance, corrective or challenging questions, repeated demands, disagreement, exasperation, strained conversational exchange, or visible tension and restless/forceful gestures.
- Frustration does not require shouting, a dramatic facial expression, or fast speech. Flat, quiet, slow, or hesitant delivery can still express frustration when the words or interaction describe difficulty, obstruction, impatience, or an attempt to make someone act or understand.
- Do not treat every question or every gesture as frustration: ordinary information-seeking and calm factual conversation remain Neutral. Require a coherent friction or difficulty cue, but do not let a neutral face outweigh clear conversational friction.
- When the record contains that coherent friction and `Frustration` is a candidate, rank the exact `Frustration` entry above `Neutral`, citing the relevant words, prosody, body movement, or temporal change. Use the candidate's exact spelling and capitalization.
- For sentiment questions, judge overall positive or negative valence instead; do not apply the emotion-specific Frustration rule merely because the topic involves a problem.
- Output JSON only: no prose before or after it, no code fence, no comments.

