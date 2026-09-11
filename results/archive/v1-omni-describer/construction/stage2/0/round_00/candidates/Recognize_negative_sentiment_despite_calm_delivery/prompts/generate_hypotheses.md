<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is an opinion, not an observation. Treat it as one weak voter.
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
- For sentiment, do not treat calm volume, steady pace, lack of laughter, or a neutral audio opinion as evidence for neutral. Neutral means the content and interaction are not directionally positive or negative.
- Support the exact candidate meaning negative when the heard or seen record conveys criticism, failure, threat, danger, loss, unpleasant consequences, hostility, complaint, rejection, or a clearly adverse situation. The words and depicted situation can establish negative sentiment even when the speaker is composed or smiling faintly.
- Do not infer negative from mere seriousness, a factual description, a dim setting, or one ambiguous facial cue. Require specific adverse content or a coherent negative interaction. Treat the audio-model opinion as a weak corroborating vote, never as the sole reason to choose neutral or negative.
- If positive and negative cues conflict, cite both sides, lower confidence, and rank the direction best supported by direct transcript or repeated visual evidence.
- Keep every label exactly as written in `{label_set}`; map the meaning to the exact available spelling, capitalization, and plurality.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
