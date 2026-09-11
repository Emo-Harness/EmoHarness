<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The `text` item describes the task and candidate list. It is not evidence that the displayed state is neutral or any other label.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Interpretation discipline:

- First decide whether the requested noun is sentiment or emotion, then use the evidence appropriate to that request.
- For sentiment, judge the overall valence conveyed by the spoken meaning, visible interaction, and delivery. Do not equate a calm voice, a neutral pose, or a factual-looking setting with neutral sentiment.
- Concrete praise, encouragement, gratitude, enjoyment, success, optimism, warmth, or affirming wording support the candidate entry meaning positive sentiment. When such cues are present and not offset by stronger negative cues, that entry should outrank neutral even if the speaker's face is composed.
- A merely factual topic is not automatically positive; require positive or negative evaluative meaning or affect. Consider the whole clip rather than a single isolated word.
- Include the strongest plausible alternatives, especially when positive and neutral are close, and score them according to the evidence rather than defaulting to neutral.

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
- Do not use the task wording or candidate-list item as substantive emotional evidence.
- Keep every label exactly as written in the candidate list, including capitalization and spelling.
- Output JSON only: no prose before or after it, no code fence, no comments.
