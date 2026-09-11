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
- The `t1` item identifies the question type; it is not evidence that the sentiment is neutral.
- When `{noun}` is sentiment, judge the valence of the complete message and situation, not just the speaker's resting face or the absence of expressive prosody. Concrete criticism, hostility, rejection, conflict, danger, loss, trouble, frightening material, regret, dissatisfaction, or explicitly bad outcomes support negative sentiment.
- A neutral-looking talking head does not cancel clearly unfavorable words or subject matter. When concrete negative content is present and there is no equally strong favorable reading, rank the exact negative candidate above neutral and cite the relevant transcript, prosody, scene, or action ids.
- Use Neutral only for genuinely factual, descriptive, or affectively uncommitted content. Do not call a clip neutral merely because the person is calmly reporting an unfavorable event or discussing disturbing/critical material without smiling.
- Distinguish sentiment from emotion: a calm speaker can still convey negative sentiment, while a smile or animated delivery alone does not prove negative or positive sentiment without considering the words.
- Match the semantic choice to an entry from the candidate list exactly as written, including capitalization. Never invent a label or substitute a synonym.
- For emotion questions, return the displayed person's emotion from observable facial, bodily, temporal, and vocal cues rather than assigning negative sentiment from a negative topic.
- Output JSON only: no prose before or after it, no code fence, no comments.

