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
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- A label should be supported by actual visual or audio evidence. The `t1` item identifies the question type; it is not evidence that the clip is neutral, positive, or negative.
- Treat the task noun literally. When `{noun}` is sentiment, judge the overall affective valence of the message and the displayed reaction, not merely whether the speaker has a neutral-looking face. Spoken content can be decisive even when prosody is absent and the setting is an ordinary talking-head scene.
- For sentiment, use `positive` when the record conveys approval, encouragement, praise, gratitude, enjoyment, success, hope, reassurance, or another favorable stance; use `negative` for criticism, hostility, trouble, loss, danger, regret, or another unfavorable stance; use `neutral` only when the content is genuinely factual or affectively uncommitted.
- Match those meanings to the actual candidate entry, including its exact capitalization. Do not invent a synonym or change a candidate label.
- Do not default to neutral solely because the clip contains ordinary conversation, a static setting, or no strong facial expression. Compare the positive and negative alternatives against the actual words and actions, then assign confidence accordingly.
- When the evidence supports a favorable sentiment, put the matching positive candidate above neutral and cite the words, prosody, or visible response that support it. Keep neutral as an alternative only when there is real ambiguity.
- For emotion questions rather than sentiment questions, continue to judge the displayed person's emotion from observable facial, bodily, temporal, and vocal cues; do not turn a merely favorable topic into happiness.
- Output JSON only: no prose before or after it, no code fence, no comments.

