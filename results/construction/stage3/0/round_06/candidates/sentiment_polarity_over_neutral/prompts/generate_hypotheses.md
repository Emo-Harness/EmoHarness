<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The task-word item `t1` identifies the question and is never evidence for a sentiment or emotion.

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
- If `{noun}` is `sentiment`, judge the evaluative polarity of the spoken and visible content, not just the speaker's facial expression. `positive` means approval, praise, encouragement, gratitude, benefit, optimism, enjoyment, a favorable judgment, or a favorable comparison. `negative` means criticism, rejection, complaint, insult, threat, harm, failure, regret, or another unfavorable judgment.
- For sentiment, use `neutral` only when the record is genuinely factual, logistical, or descriptive and contains no specific favorable or unfavorable evaluation. Do not choose neutral merely because the face is controlled, the speaker is seated, or the topic is presented calmly.
- A calm delivery can still express positive or negative sentiment. Conversely, a smile or energetic voice alone is not enough to override clearly negative wording.
- Prefer a specific evaluative phrase over many generic observations such as sitting, looking, clothing, speaking, or room setting. Do not treat the question wording as evidence.
- If the clip contains several people, identify the person or interaction that is most central and emotionally informative rather than averaging unrelated background people.
- For emotion tasks, use visible and audible affect rather than the subject matter alone; do not transfer the sentiment rules to an emotion label without an affective cue.
- Confidence should reflect the evidence, not a default preference for neutral. When a specific cue supports a non-neutral label, score that label above neutral unless a stronger cue contradicts it.
- Keep every label exactly as written in the candidate list.
- Output JSON only: no prose before or after it, no code fence, no comments.
