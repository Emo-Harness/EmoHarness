<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words, sounds, or events.
- Generic smiling, sitting, looking, or ordinary speaking is weak evidence. Activation cues such as rapid or emphatic delivery, repeated animated gestures, broad movement, anticipation, exclamations, or visibly heightened energy are stronger evidence.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Decision calibration for Excited versus Happiness:

- Separate valence from arousal. Happiness means positive or smiling affect and can be calm. Excited means positive high activation, enthusiasm, eager anticipation, energetic surprise, or visibly heightened engagement.
- When both Excited and Happiness are available, rank Excited above Happiness when the record contains multiple specific activation cues: fast or emphatic speech, repeated or expansive gestures, broad or animated movement, heightened pitch or loudness together with positive content, exclamatory wording, eager anticipation, or explicit enthusiasm.
- One generic gesture, one smile, variable pitch alone, or ordinary conversational energy is not enough to establish Excited. In those cases prefer Happiness when a smile or other positive cue is present, and prefer Neutral when no positive cue is present.
- Positive content such as an optimistic statement can support either label, but use the activation evidence to decide between them. Do not treat every positive sentence as Excited.
- A controlled face does not cancel strong audio activation, but fast speech alone without positive or enthusiastic content should not be called Excited.
- If the label list uses a different capitalization or spelling, copy the supplied candidate exactly; never normalize or invent a label.
- Do not use the task-word item `t1` as evidence for any emotion.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order hypotheses from most to least likely after applying the calibration above. Two to four are usual.
- Include the strongest plausible alternative when it is in the candidate list.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Confidence must reflect specific activation and valence evidence, not the number of generic observations.
- Output JSON only: no prose before or after it, no code fence, no comments.
