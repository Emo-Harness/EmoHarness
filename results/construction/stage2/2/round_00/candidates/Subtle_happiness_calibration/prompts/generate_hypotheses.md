<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The question wording and the item whose cue is the noun being asked about are not emotional evidence. Do not treat `t` items as support for a label.

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
- A label must have at least one genuine supporting cue when such a cue exists; do not make a label the leader merely because it is the default or because the face is hard to read.
- When an entry corresponding to happy is available, treat a visible smile, smiling mouth, softened or cheerful expression, pleasant amusement, warmth, or contentment as direct support. A slight smile still counts when it persists or is repeated, especially when the audio is friendly or affirming. Rank happy above neutral when these positive cues are the clearest affective evidence and there is no strong contrary cue. Reserve Neutral for an absence of clear positive or negative affect, not merely for a composed posture, ordinary setting, or continued speech.
- Do not infer happiness from a generic open mouth, looking at the camera, or a factual sentence alone. Do not let numerous non-emotional appearance and scene observations outweigh a small number of direct expression cues.
- For sentiment questions, assess favorable or unfavorable meaning in the message separately from visible facial affect; factual content without evaluation remains neutral.
- Output JSON only: no prose before or after it, no code fence, no comments.

