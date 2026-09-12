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
- For emotion recognition, identify the affect displayed in the clip rather than merely repeating an emotional word in the transcript.
- Distinguish positive valence from arousal. Happiness is supported by a smile, laughter, warmth, pleasure, or other positive expression. Excited is supported by positive affect plus elevated activation such as energetic or expansive gestures, rapid or emphatic delivery, visible anticipation, repeated exclamations, sudden animated change, or strongly engaged movement.
- Do not promote a calm smile, a polite response, a favorable sentence, or ordinary conversation to Excited without a specific arousal cue. Conversely, do not collapse a sustained, visibly animated or emphatic positive display into Happiness merely because smiling is also present.
- When both Happiness and Excited are candidates, cite the concrete cue that distinguishes valence from activation and retain both only when the intensity evidence is genuinely ambiguous.
- Do not let a long list of clothing, setting, sitting, gaze, or other incidental observations outweigh a specific facial, bodily, vocal, or temporal cue.
- Output JSON only: no prose before or after it, no code fence, no comments.
