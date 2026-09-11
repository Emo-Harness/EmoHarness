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
- For sentiment questions, evaluate the meaning and tone of the message and situation, not just the speaker's facial expression. Criticism, rejection, dismissal, insult, threat, danger, conflict, regret, unpleasant outcomes, ominous subject matter, and explicit negative evaluation support the negative entry when it is present in the candidate list. A composed or smiling face does not cancel clear negative wording. When negative evidence is clear and there is no comparably strong favorable evidence, rank negative above neutral.
- Reserve Neutral for factual or descriptive material with no clear favorable or unfavorable evaluation. Do not use it merely because the video is a talking-head clip, the expression is restrained, or the transcript is short.
- For emotion questions, do not treat the noun in the question or generic appearance facts as emotional evidence. Use direct facial, body, vocal, and temporal cues.
- Output JSON only: no prose before or after it, no code fence, no comments.

