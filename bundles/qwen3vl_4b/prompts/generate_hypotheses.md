<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- `t1` is task context, not evidence, and must never be cited.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Before writing the JSON, silently audit every candidate label against the full record. Do not stop after finding one plausible reading.

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`; never invent an id, cite `t1`, or cite an id twice in one list.
- Include every candidate that remains reasonably plausible after the audit, even when its confidence is low. Do not omit a plausible candidate merely because another reading is stronger.
- If the record is ambiguous and several broad states are compatible, retain the strongest neutral or `Other` reading together with the strongest plausible affective readings. If a specific cue supports a candidate, that candidate must be considered for inclusion.
- Do not add labels solely to fill space. A candidate needs an observed facial, bodily, vocal, or semantically relevant sentiment cue, or it must be the appropriate uncertainty reading when no candidate has direct support.
- Use at most four hypotheses, choosing the four best-supported or most plausible readings after considering the complete candidate list. Order them from most to least likely.
- For emotion, distinguish the visible speaker or person being labeled from people or events mentioned in speech. For sentiment, distinguish the speaker's expressed attitude from the neutral subject matter being discussed.
- Confidence must reflect relative evidence, not just whether a label was mentioned in the question. Do not use a fixed prior for any label.
- Keep every label exactly as written in the candidate list.
- Output JSON only: no prose before or after it, no code fence, no comments.
