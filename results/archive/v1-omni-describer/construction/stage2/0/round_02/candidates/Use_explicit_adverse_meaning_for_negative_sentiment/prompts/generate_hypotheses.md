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

SENTIMENT CALIBRATION FOR NEGATIVE VERSUS NEUTRAL:

- Apply this calibration when the noun is `sentiment` and the candidate list contains the exact labels `negative` and `neutral`.
- Read the meaning and stance of the utterance, not only its vocal calmness. Explicit criticism, complaint, rejection, threat, failure, harm, regret, inability, an adverse consequence, or wording that says something is excessive, wrong, difficult, or unwanted supports `negative`.
- A steady or moderate voice, absence of crying or laughter, and an audio-model vote of neutral do not contradict clearly adverse wording. The audio opinion is one fallible voter.
- Prefer `neutral` only when the content is genuinely factual or affectively uncommitted and there is no meaningful adverse meaning. When the transcript or valid audio observations contain a clear adverse stance or consequence, rank the exact candidate-list label `negative` above `neutral` and cite the relevant ids.
- Do not label a clip negative merely because it contains a neutral question, ordinary description, or a mildly unfavorable isolated word without an adverse stance. Use the full utterance and the strongest supported meaning.

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
- Treat audio opinions as fallible votes, never as observations or vetoes.
- A label must be copied character for character, including capitalization.
- Output JSON only: no prose before or after it, no code fence, no comments.

