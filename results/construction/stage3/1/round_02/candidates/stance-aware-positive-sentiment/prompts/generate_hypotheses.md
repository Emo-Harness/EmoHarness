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

For sentiment questions, classify the speaker's expressed stance toward the subject or interaction, not merely the topic and not merely the speaker's facial appearance. A positive stance can be explicit praise, approval, gratitude, encouragement, reassurance, admiration, optimism, satisfaction, affectionate valuation, or a favorable assessment of something. Favorable self-reference or saying that something is good, meaningful, useful, funny, great, important, or wanted can support positive when it functions as an evaluation. A factual account, neutral description, question, name, or mention of a positive event is not positive by itself. Do not require an excited voice when the words clearly express approval.

For emotion questions, identify the affect actually displayed by the visible or vocal cues rather than the emotionality of the subject matter.

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
- For sentiment, do not make `neutral` the default merely because visual cues are ordinary or the transcript is descriptive. First test whether the words contain a favorable or unfavorable appraisal, endorsement, concern, rejection, or other stance.
- For sentiment, do not infer positive from a pleasant setting, a smile unrelated to the subject, or a positive event that the speaker does not evaluate.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Keep support tied to the exact evidence: a generic prosody tag alone is weak, while explicit evaluative words or a direct facial cue are stronger.
- Output JSON only: no prose before or after it, no code fence, no comments.

