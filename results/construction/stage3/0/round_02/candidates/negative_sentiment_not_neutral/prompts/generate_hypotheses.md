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

For a sentiment question, judge the conveyed stance and evaluation rather than requiring overt anger or a dramatic face. Negative sentiment can be quiet or conversational and can be expressed by criticism, complaint, dismissal, refusal, warning, threat, conflict, frustration, disappointment, ominous content, or language describing an undesirable situation. A speaker can smile, remain composed, or discuss a neutral-looking scene while the words and interaction carry a negative evaluation. Positive sentiment likewise requires positive evaluation or affiliative affect, not merely a bright setting.

Neutral is appropriate only when the whole clip is affectively unmarked: ordinary factual description with no meaningful approval, disapproval, conflict, concern, distress, or evaluative stance. Do not promote neutral just because the transcript is incomplete, the face is not strongly expressive, or most visual items describe clothing, posture, and setting. Do not use the task noun itself as evidence. Weigh semantic content, speech act, prosody, facial action, gestures, and temporal persistence together, and distinguish what the speaker conveys from the literal topic being discussed.

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
- When negative and neutral are both plausible, include both and explicitly weigh specific negative cues against generic neutral-looking observations.
- Do not assign near-certainty to neutral merely because no single cue is extreme. A coherent low-intensity negative stance should outrank neutral.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
