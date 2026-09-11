<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is an opinion, not an observation. Treat it as a weak voter and never cite it as something seen or heard.
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
- Because the noun may be sentiment, do not equate a steady voice, no laughter, or a calm posture with neutral. Neutral means that the record is substantially informational or affectively uncommitted.
- For sentiment, evaluate the meaning of the spoken content and the visible interaction as well as prosody. Praise, encouragement, approval, gratitude, welcome, affection, enjoyment, celebration, or clearly favorable outcomes support the exact candidate whose meaning is positive. Smiling or friendly engagement is additional support.
- Do not call a clip positive merely because it is fluent, well-lit, factual, or contains a generic slogan. A neutral audio opinion is weak and cannot override explicit positive content or visible warmth.
- If the record contains both directional and calm cues, calm delivery describes intensity, not valence. Rank the supported positive candidate above neutral when the positive evidence is specific and repeated.
- Keep every label exactly as written in `{label_set}`; map meanings to the available exact spelling and capitalization rather than inventing a synonym.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
