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

- First determine whether this is an emotion question or a sentiment question. For sentiment, judge the valence of the whole communicative act, including the spoken meaning, interaction, and visible behavior, rather than treating a calm face as neutral.
- When the candidate list contains `positive` and `neutral`, treat praise, encouragement, optimism, gratitude, favorable recommendations, constructive or affirming framing, and clearly pleasant interaction as evidence for `positive` even when delivery is measured or the setting is ordinary.
- Use `neutral` for genuinely informational or unmarked material with no favorable or unfavorable framing. Do not call something positive merely because a person is present, speaking politely, or has a generic smile.
- Do not let the number of generic appearance or posture observations outweigh one diagnostic semantic or interaction cue. Still cite the diagnostic ids that support the decision.
- For all other labels, use the same evidence-based comparison and do not invent a cue.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
