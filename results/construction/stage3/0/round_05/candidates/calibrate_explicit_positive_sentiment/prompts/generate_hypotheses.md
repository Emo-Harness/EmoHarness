<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words, translations, or meanings.
- Generic appearance and activity are weak evidence for sentiment. The speaker's explicit evaluation, appraisal, approval, enjoyment, relief, gratitude, or criticism is stronger evidence.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Decision calibration for sentiment questions:

- Classify the sentiment expressed by the speaker or communication, not just the speaker's facial expression. A neutral-looking person may still express positive sentiment in a favorable evaluation.
- When positive and neutral are available, favor positive when the transcript contains a clear favorable appraisal, approval, enjoyment, amusement, gratitude, relief, benefit, endorsement, or an explicitly optimistic evaluation. Examples include saying something is great, funny, makes more sense, worthwhile, fortunate, or worthy of credit.
- Favorable sentiment can be understated or embedded in an otherwise factual sentence. Do not require a broad smile, excited prosody, or an overtly cheerful voice.
- Do not call a passage positive merely because it mentions a happy person, a desirable object, a religious or advertising topic, a routine greeting, or a neutral fact. Instructional or promotional wording without an identifiable favorable stance is not by itself enough.
- Use neutral for factual description, plot summary, topic mention, or ordinary conversation when there is no clear positive or negative evaluation. Generic fast speech, loudness, smiling, or a pleasant setting cannot by themselves settle sentiment.
- If explicit evaluative language is present, cite the audio item that contains it. Do not let many generic visual items outweigh one specific evaluative cue.
- Do not use the task-word item `t1` as evidence for sentiment.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order hypotheses from most to least likely after applying the calibration above. Two to four are usual.
- Include the strongest plausible alternative when it is in the candidate list.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Confidence must reflect the specificity of the stance, not the amount of generic visual description.
- Output JSON only: no prose before or after it, no code fence, no comments.
