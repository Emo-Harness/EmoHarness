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

Before writing hypotheses, identify the signal type. If the noun is sentiment, classify the stance expressed by the actual words and not the topic, setting, speaker identity, or delivery energy. A positive hypothesis is supported by explicit approval, praise, recommendation, encouragement, gratitude, enjoyment, appreciation, agreement, favorable evaluation, or clearly positive wording. A negative hypothesis is supported by explicit criticism, rejection, complaint, regret, dismissal, insult, threat, or unfavorable evaluation. Subject matter such as religion, danger, illness, death, politics, or a disagreement is not itself a sentiment label. Neutral is appropriate when the available words are factual or incomplete and contain no clear appraisal; do not treat a calm voice or neutral face as a contradiction of an explicit appraisal. For emotion, use direct facial, bodily, vocal, and temporal affect rather than the emotional topic being discussed.

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- For sentiment, cite the exact transcript item for an appraisal rather than citing generic prosody or the subject matter. A calm positive appraisal remains positive; a factual mention remains neutral unless the words evaluate it.
- For emotion, prioritize direct affective observations and changes over clothing, setting, ordinary posture, and the number of available evidence items.
- Do not infer the meaning of an unintelligible or foreign fragment beyond what the record establishes.
- Include a meaningful positive, negative, or emotional alternative when a direct cue supports it, even if Neutral has more incidental support items.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.

