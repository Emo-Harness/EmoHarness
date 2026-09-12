<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words or sounds.
- Generic identity, clothing, setting, sitting, looking, and mouth movement are weak evidence. A described smile, amused expression, cheerful interaction, or change toward smiling is specific evidence.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Decision calibration for happy versus neutral:

- A direct observed smile, smiling mouth, teeth showing while smiling, amused expression, laugh-like expression, or clearly cheerful interpersonal engagement is positive facial evidence. Do not discard it just because it is slight or because the person is seated.
- When a label exactly matching `happy` is present, rank that label above Neutral when the focal person's visible evidence contains a sustained or repeated smile, amusement, or cheerful engagement and there is no stronger contradictory cue. Copy the actual candidate spelling exactly; this instruction is about meaning, not about inventing a label.
- When a label exactly matching `Happiness` is present instead, apply the same semantic rule to that candidate. Never output a spelling that is not in the supplied list.
- Do not infer happiness from ordinary mouth movement, looking at someone, a pleasant room, clothing, or a single ambiguous open mouth. If the record says neutral expression, no smile, or a clearly downcast or distressed face, keep Neutral or the more specific negative label ahead.
- If several people are visible, weigh the person whose face and behavior are most prominent in the clip; do not let a distant neutral bystander erase a clearly smiling focal person.
- Temporal evidence matters: a change from neutral toward smiling supports happy, while a smile that disappears can be reported as mixed but should not be ignored.
- Do not use the task-word item `t1` as evidence for any emotion.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order hypotheses from most to least likely after applying the calibration above. Two to four are usual.
- Include the strongest plausible alternative when it is in the candidate list.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Confidence must reflect the specificity of facial and behavioral evidence, not the number of generic observations.
- Output JSON only: no prose before or after it, no code fence, no comments.
