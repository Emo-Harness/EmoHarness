<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is one fallible opinion, not evidence by itself.
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
- Do not use generic calm prosody, sitting still, or a neutral audio opinion as proof of neutral. Neutral requires the absence of a specific emotional direction.
- When the record shows an active blocked goal or dissatisfaction, such as repeated objections, inability, refusal, complaint, correction, urgency about a problem, or words describing something as hard or messed up, support the exact candidate meaning frustration or irritation. Fast or raised speech, emphatic repetition, and larger or tense gestures strengthen that reading.
- Distinguish frustration from sadness: sadness is better supported by loss, grief, withdrawal, crying, or low-energy sorrow, whereas frustration centers on resistance, difficulty, annoyance, or an unsolved obstacle. Do not downgrade frustration merely because the speaker is not shouting.
- A single generic mention of a problem without objection, urgency, or affect is weak and may remain neutral. The audio-model opinion is only a tie-breaker and must not outrank several direct transcript, prosodic, or visual cues.
- Keep every label exactly as written in `{label_set}`. Use the available capitalization and wording for the matching meaning; never invent a label.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
