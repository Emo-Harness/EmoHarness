<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- Cues describe observable behavior; infer the displayed affect from their combination rather than from one generic action.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Determine whether the task asks for an emotion or a sentiment. For an emotion task, prioritize the person's displayed face, body, voice, and change over time. Do not classify the topic of a sentence as the displayed emotion. For a sentiment task, use the speaker's expressed attitude and valence as well as delivery.

When the candidate list contains both a general positive state and a high-arousal positive state, distinguish them carefully. Sustained smiling, relaxed pleasant engagement, and approving content support the general positive state. Excited or energized affect is better supported by a combination of animated gestures, rapid or forceful delivery, high or variable pitch, loudness, widened eyes, open mouth, emphatic repetition, obvious escalation, or explicit anticipation. A single smile or a single open mouth is not enough to call something excited. Conversely, do not downgrade clear energetic, escalating evidence merely because the person is also smiling. Apply the same evidence-based care to all other candidate labels and use the closest exact candidate wording.

Do not default to neutral when a real facial, vocal, or temporal affective cue is present. Do not let a long list of irrelevant scene or appearance facts outweigh a small number of direct affect cues. If the record is genuinely ambiguous, include the strongest alternative with honest confidence and cite the cues that separate them.

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
- Keep every label exactly as written in the candidate list, including capitalization and spelling.
- Output JSON only: no prose before or after it, no code fence, no comments.

