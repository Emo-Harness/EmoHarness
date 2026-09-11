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

Positive-arousal calibration:

- When the candidate list contains both Happiness and Excited, use Happiness for a warm, pleased, satisfied, amused, or calmly smiling display.
- Use Excited when the evidence shows high activation or eager anticipation, especially a combination of animated or expansive gestures, energetic movement, rapid or emphatic delivery, repeated exclamations, visible eagerness, or an anticipated favorable event.
- A smile or favorable sentence alone supports Happiness but does not establish Excited.
- Fast speech or variable pitch alone is not enough for Excited; require corroboration from the transcript, movement, facial intensity, or temporal change.
- When both labels are supported, rank the label whose cues are more specific and prominent, and preserve the other as an alternative rather than treating the labels as synonyms.
- Do not let a large number of generic posture or appearance observations outweigh a smaller number of direct high-arousal cues.

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
- Keep every label exactly as written in the candidate list; do not substitute synonyms such as joy, happy, or enthusiasm.
- Output JSON only: no prose before or after it, no code fence, no comments.

