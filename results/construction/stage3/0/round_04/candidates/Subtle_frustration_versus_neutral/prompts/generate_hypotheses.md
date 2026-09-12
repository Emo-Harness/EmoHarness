<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t1` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- Generic facts such as clothing, sitting, looking, speaking, or a room setting are weak evidence. Specific changes, wording, posture, facial tension, and prosody are stronger.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`.
  - `label` must be one entry from the list above, copied exactly as written there.
  - `support` and `contradiction` must contain only evidence ids from the record.
  - `confidence` must be a number between 0 and 1.

Decision policy for emotion:

- Determine the affective state of the principal person or interaction, not the emotion named by the question and not the mood of a background person.
- Frustration means blocked effort, inability to make progress, confusion about how to proceed, delay, an unresolved problem, repeated failed attempts, corrective questioning, or insistence that something should be different. It can be quiet, slow, controlled, or accompanied by an ordinary face.
- Specific evidence such as I do not understand, I cannot do this, why would they do this, what are you going to do, repeated requests for an explanation, a problem being revisited, or visible tense/problem-solving gestures supports Frustration when that label is available.
- Do not require anger, profanity, raised volume, or an obvious scowl for Frustration. Conversely, urgency or fast speech alone is not enough when there is no blocked effort or difficulty.
- Neutral is appropriate when the record shows ordinary interaction without a specific affective cue. Do not promote neutral merely because the face is restrained or the prosody is calm when the words or temporal behavior describe difficulty or unresolved effort.
- Anger is stronger than Frustration when there is direct blame, accusation, reproach, hostile challenge, profanity, threats, or an abrasive interpersonal target. Sadness is stronger when loss, grief, resignation, or low mood is the central cue. Fear is stronger when anticipated threat or danger is central.
- For sentiment, classify positive and negative evaluative polarity rather than treating all calm speech as neutral. Use neutral only when there is no specific evaluation.
- Do not use the task word or task item `t1` as evidence for any label.
- The most prominent label describes the overall clip; do not let a minor person or generic activity dominate.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports any candidate.
- Output JSON only: no prose before or after it, no code fence, no comments.
