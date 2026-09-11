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

FRUSTRATION CALIBRATION:

- Apply this calibration when the noun is `emotion` and the candidate list contains the exact labels `Frustration` and `Neutral`.
- Frustration is supported by an unresolved obstacle or conflict expressed through repeated questions, challenges, disagreement, correction, inability, failure, a problem, an unwanted outcome, or repeated words such as "can't", "don't know", or "what are you going to do".
- Visible repeated or enlarged hand gestures, leaning forward while speaking, tense posture, furrowed brows, or an agitated change over time can reinforce frustration. Cite those visual ids when present.
- Calm volume, moderate pitch, no laughter, and an audio-model vote of neutral do not prove neutral and must not veto clear problem or conflict language. The audio opinion is one fallible voter.
- Prefer `Neutral` only when the expression and wording are explicitly even and there is no meaningful obstacle, disagreement, complaint, or frustrated behavior. When the record contains the signals above, rank the exact candidate-list label `Frustration` above `Neutral` and cite the strongest valid ids.
- Do not infer frustration from ordinary sitting, speaking, looking, or generic scene facts alone. Require a semantic conflict/problem cue or a conspicuous frustrated behavioral cue.

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

