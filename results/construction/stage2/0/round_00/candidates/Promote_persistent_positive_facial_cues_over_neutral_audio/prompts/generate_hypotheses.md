<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is an opinion, not an observation. Treat it as one weak voter and never let it alone establish an emotion.
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
- For this most-prominent-emotion task, do not use calm speech, a steady posture, or an audio opinion of `neutral` as positive evidence for a neutral label. Neutral is appropriate only when the record lacks a stronger specific affective cue.
- Inspect visible facial cues before defaulting to neutral. A smile that persists across observations, raised cheeks, narrowed smiling eyes, or a consistently warm expression supports the exact candidate whose meaning is happy, happiness, joy, or similar. Repeated facial evidence should outrank generic flat prosody; one ambiguous mouth shape by itself should not.
- If a positive facial cue is present but the voice is merely calm, keep the positive candidate in first place when it is the strongest specific cue. Mark generic calm audio as weak or contradictory only when it truly conflicts with the visible expression.
- Use an audio-opinion item as a tie-breaker, not as a substitute for seen or heard observations.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Keep every label exactly as written in `{label_set}`. Do not normalize capitalization, spelling, or plurality.
- Output JSON only: no prose before or after it, no code fence, no comments.
