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

For sentiment questions, classify the speaker's expressed stance toward the subject or interaction, not merely the topic and not merely the speaker's facial appearance. A negative stance can be explicit criticism, complaint, dismissal, rejection, regret, threat, insult, warning, concern, accusation, unfavorable comparison, frustration with a situation, or an assertion that something is bad, difficult, unwanted, absurd, or unacceptable. Calm delivery, flat prosody, a neutral face, or polite wording does not cancel an explicit unfavorable appraisal. A factual account of violence, a negative-sounding topic, a place or person's name, and a question without an evaluative stance are not negative by themselves. If the wording is incomplete, use only the stance actually supported by the available words.

For emotion questions, identify the affect actually displayed by the visible or vocal cues rather than the emotionality of the subject matter.

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
- For sentiment, test explicit stance before choosing `neutral`; ordinary visual evidence and plentiful incidental items must not outweigh a direct unfavorable appraisal in the transcript.
- For sentiment, do not infer negative from subject matter alone, from generic loudness or speed, or from a neutral description of an unpleasant event.
- For emotion, do not use a sentiment label merely because the spoken topic is positive or negative.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Keep support tied to the exact evidence: explicit evaluative words and direct facial or vocal affect are stronger than generic scene details or one undifferentiated prosody tag.
- Output JSON only: no prose before or after it, no code fence, no comments.

