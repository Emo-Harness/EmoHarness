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

- For emotion questions, classify the person's displayed affect, integrating facial expression, body behavior, spoken meaning, and prosody. Do not let generic appearance descriptions or the task wording create support for neutral.
- Frustration can be quiet or low-arousal. Complaints, obstacles, inability, repeated demands, impatient or irritated questions, argumentative wording, urgency about a problem, and visible agitated gestures support the candidate entry meaning frustration even when the speaker is seated or composed.
- Distinguish frustration from anger: reserve anger for stronger hostility, threat, or overt rage. Distinguish both from neutral: neutral is appropriate only when the record lacks a meaningful affective signal, not merely because the face is restrained.
- If the evidence contains a clear frustration cue, include the frustration entry and rank it above neutral unless a stronger competing emotion is supported. Use the exact candidate spelling and capitalization.

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
