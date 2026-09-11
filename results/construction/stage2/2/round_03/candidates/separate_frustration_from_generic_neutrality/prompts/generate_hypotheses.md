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
- For an emotion request, distinguish Frustration from Neutral rather than treating every calm-looking or seated person as neutral. The frustration entry is supported by direct signs of an obstacle or difficulty, inability, complaint, repeated challenging questions, exasperation, tense or furrowed facial configuration, agitated or emphatic gestures, or insistent delivery.
- Generic facts such as sitting, looking, wearing clothing, a plain setting, or ordinary turn-taking do not support Neutral strongly enough to cancel a direct frustration cue. Fast or high-pitched delivery alone is not sufficient, but it strengthens frustration when paired with a problem, challenge, or strained behavior.
- When direct frustration evidence is present, rank the frustration entry above a neutral entry supported only by generic observations. Keep Neutral as an alternative if the evidence is genuinely mixed, and cite the direct cues in the frustration support list.
- If the evidence is ambiguous, include the frustration and neutral entries rather than omitting frustration solely because it is less visually obvious.
- Output JSON only: no prose before or after it, no code fence, no comments.

