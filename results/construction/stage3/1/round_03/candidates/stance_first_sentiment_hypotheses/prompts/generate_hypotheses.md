<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and one starting with `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words, translations, or emotional intent that is not present.
- Visual items are observations, not a vote. Clothing, room layout, sitting, looking, and ordinary mouth movement are incidental unless they directly show affect.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

First identify what kind of reading the question requires. For a sentiment question, classify the stance expressed toward the subject or interaction. For an emotion question, classify the affect visibly or vocally displayed. Do not substitute the speaker's facial appearance for the sentiment of the words.

For sentiment questions, use this adjudication:

- A clear approval, praise, gratitude, encouragement, endorsement, favorable evaluation, enjoyment, or positive judgment supports positive valence.
- A clear criticism, complaint, rejection, regret, dismissal, threat, insult, concern, unfavorable evaluation, or negative judgment supports negative valence.
- A sentence that merely names a person, topic, event, plot detail, location, or factual circumstance is not emotional by itself.
- A calm delivery does not erase an explicit appraisal, and a smile does not turn factual narration into positive sentiment.
- Use prosody as corroboration. Generic loudness, pitch, speed, or a question mark is not by itself a positive or negative stance.
- If the transcript is incomplete, use only the words that are actually present. Do not manufacture the missing conclusion.

For emotion questions, give priority to direct facial or bodily affect, clear expression changes, vocal affect, and persistent temporal change. Static scene facts and ordinary posture should not outweigh a direct affect cue. A label needs actual evidence ids when any relevant cue exists; do not give a neutral reading a large advantage merely because it can cite many incidental items.

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
- Keep support specific: cite the transcript or prosody for sentiment and direct affect cues for emotion; do not put every ordinary visual item under a neutral hypothesis.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Do not treat the noun in the question or the candidate-list wording as evidence for a label.
- Output JSON only: no prose before or after it, no code fence, no comments.
