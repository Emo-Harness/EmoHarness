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
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- The `t1` item identifies the question type and is not evidence for Neutral or any emotion.
- When `{noun}` is emotion, treat a visible smile, smiling mouth, softened pleasant expression, or repeated smile across the clip as direct evidence for happiness. A slight smile is still evidence; it does not need to be broad or accompanied by laughter.
- If a person visibly smiles and there is no clear competing fear, anger, sadness, disgust, or distress cue, rank the exact candidate meaning happy/happiness above Neutral. Do not let neutral clothing, posture, a plain background, ordinary speech, or many generic scene observations outweigh the direct facial cue.
- Use Neutral when the face and behavior genuinely lack a clear affective display, or when strong contrary cues explain why a smile is incidental or overridden. Do not infer happiness from a merely positive topic when no visible positive display exists.
- If multiple people are visible, identify the person whose emotion is most prominent and use the overall balance of repeated facial and bodily cues rather than counting every neutral background fact as an opposing emotion.
- Match the meaning to one entry from the candidate list exactly, preserving case and spelling. Never invent `happy` or `happiness` if that exact candidate is absent.
- For sentiment questions, assess overall valence rather than requiring a smile; do not apply the smile rule as a substitute for sentiment analysis.
- Output JSON only: no prose before or after it, no code fence, no comments.

