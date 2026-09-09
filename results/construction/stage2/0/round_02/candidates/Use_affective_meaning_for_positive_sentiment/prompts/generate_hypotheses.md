<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is an opinion, not an observation. Treat it as one voter that can be wrong, and never cite it as something that was seen or heard.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

SENTIMENT CALIBRATION FOR POSITIVE VERSUS NEUTRAL:

- Apply this calibration when the noun is `sentiment` and the candidate list contains the exact labels `positive` and `neutral`.
- Determine the sentiment of the communicated utterance or interaction, not merely the speaker's arousal. Explicit approval, gratitude, reassurance, encouragement, relief, praise, enjoyment, or a clearly favorable outcome supports `positive`.
- Words such as `thankfully`, `great`, `funny`, `happy`, `encourage`, `share`, `meaningful`, or a statement of liking or benefit are semantic evidence when they occur in the transcript or a valid audio observation. A calm, steady, or flat delivery does not cancel that meaning.
- An audio-model opinion of neutral is only one fallible voter. Do not use the absence of laughter or a moderate pace as evidence against positive sentiment.
- Prefer `neutral` only when the wording is affectively bland or purely factual and there is no clear favorable or unfavorable meaning. When the record contains clear favorable meaning, rank the exact candidate-list label `positive` above `neutral`, and support it with valid transcript or audio ids.
- Do not confuse a positive description of the subject matter with a positive affective interpretation unless the wording itself conveys approval, benefit, enjoyment, encouragement, or another favorable stance.

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
- A neutral audio opinion cannot by itself outweigh clear favorable transcript meaning for `positive`.
- A label must be copied character for character, including capitalization.
- Output JSON only: no prose before or after it, no code fence, no comments.

