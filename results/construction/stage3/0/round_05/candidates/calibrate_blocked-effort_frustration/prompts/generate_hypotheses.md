<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words, meanings, or events that are not present.
- Generic observations such as sitting, looking, clothing, a room, or ordinary speaking are weak evidence. Specific facial tension, posture, gesture, prosody, and affective wording are stronger evidence.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Decision calibration for emotion questions:

- Judge the affective state of the focal person, not merely whether the conversation looks calm. A controlled face or seated posture does not erase a specific affective cue in the words or prosody.
- When both Frustration and Neutral are available, use Frustration for blocked effort, inability, confusion about how to proceed, delay, a problem that is not resolving, repeated questioning caused by difficulty, or strained problem-solving. Frustration does not require shouting, anger, or an openly hostile target.
- Reassuring, quiet, slow, or flat delivery can still accompany Frustration when the content expresses difficulty or being unable to proceed. Treat uncertainty and blocked action as relevant rather than automatically neutral.
- Use Neutral when the record is merely factual, descriptive, routine, or conversational and contains no specific difficulty, blocked goal, uncertainty, or other affective cue. Do not manufacture Frustration from fast speech, ordinary gestures, or generic conversation alone.
- If a cue supports a specific emotion, do not let a larger number of generic visual items outweigh it. Cite the specific cue in `support` and cite genuinely conflicting cues in `contradiction`.
- Do not use the task-word item `t1` as evidence for any emotion.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely after applying the calibration above. Two to four are usual.
- Include the strongest plausible alternative when it is in the candidate list, even if its confidence is lower.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Confidence must reflect the strength and specificity of the evidence, not the number of generic observations.
- Output JSON only: no prose before or after it, no code fence, no comments.
