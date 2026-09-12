<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- A text item that merely says `emotion` or `sentiment` identifies the task; it is not evidence for neutral or for any other label.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Decide the label from the overall affective signal of the clip, not from the topic alone. For a sentiment question, use the speaker's or scene's expressed evaluation and tone: approval, encouragement, appreciation, enjoyment, warmth, smiling, laughter, welcoming language, or an upbeat delivery support a positive label; criticism, rejection, complaint, threat, distress, hostility, or a bleak evaluation support a negative label. A factual, educational, or descriptive topic is not automatically neutral, and calm delivery does not erase positive or negative wording.

Neutral is a substantive conclusion, not a safe fallback. Use it only when the available facial, bodily, scene, lexical, and prosodic cues are genuinely informational or affectively unmarked. Do not use generic appearance, speaking, sitting, or ordinary gaze as strong support for neutral when a meaningful positive or negative cue is present. Weigh repeated and temporally consistent cues more than one ambiguous cue, and consider the interaction between words, prosody, expression, and gestures.

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
- If the label list contains positive, negative, and neutral choices, explicitly compare the three whenever the evidence is mixed or subtle; do not emit neutral as the only hypothesis merely because the cue is not extreme.
- Include a plausible alternative when a positive or negative cue competes with neutral, and calibrate confidence to that ambiguity rather than assigning near-certainty to neutral.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
