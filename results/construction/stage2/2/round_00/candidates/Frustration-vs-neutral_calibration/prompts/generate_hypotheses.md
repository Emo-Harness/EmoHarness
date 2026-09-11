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
- When an entry corresponding to Frustration is available, use it for observable dissatisfaction, a blocked or difficult goal, inability, repeated complaint, exasperation, impatience, strained insistence, or questioning that is clearly driven by a problem. These cues can be present in the words, prosody, gestures, or temporal change even when the facial expression is restrained. Do not downgrade such evidence to Neutral merely because the setting is ordinary or the speaker is seated. Use Neutral instead for an ordinary interaction lacking a problem, struggle, dissatisfaction, or other affective signal; do not call every question or every movement frustration.
- For sentiment questions, distinguish a calm positive or negative evaluation from genuinely factual neutral speech; do not use the question noun as evidence.
- For emotion questions, weigh the strongest affective event and the speaker's delivery rather than counting many appearance or setting facts as emotional support.
- Output JSON only: no prose before or after it, no code fence, no comments.

