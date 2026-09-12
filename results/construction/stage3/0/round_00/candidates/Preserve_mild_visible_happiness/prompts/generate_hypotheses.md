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

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`. `label` must be one entry from the list above copied exactly; the two id lists may contain only ids from the record; `confidence` is between 0 and 1.

Decision policy:

- First determine whether this is an emotion question or a sentiment question. For emotion, classify the displayed affect, not the emotionality of the topic or the mere fact that someone is speaking.
- When lowercase `happy` is a candidate, a slight smile, softened pleasant expression, amused look, contented demeanor, or other clearly positive facial cue is valid evidence for `happy`; happiness need not be broad, exuberant, or accompanied by positive words.
- Do not require a smile to be large, sustained, or shared by every person. Generic still posture, ordinary surroundings, or neutral informational speech do not contradict a localized visible happy cue.
- Rank `neutral` above `happy` only when the face and behavior are genuinely unmarked or when a stronger negative, fearful, angry, or surprised cue explains the expression better. Do not use the number of generic posture observations as a substitute for affective evidence.
- Keep labels copied character for character from the candidate list and distinguish lowercase labels from capitalized labels.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
