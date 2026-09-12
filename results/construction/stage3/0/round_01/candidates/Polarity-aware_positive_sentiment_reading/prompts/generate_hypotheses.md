<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and one starting with `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The `t` item identifies the task and is not evidence of an emotion or sentiment.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Every label must be one entry from the supplied list, copied exactly. Order two to four hypotheses from most to least likely.
- For a sentiment question, classify the affective polarity being communicated, not merely the subject matter. Positive sentiment includes approval, praise, encouragement, gratitude, satisfaction, enjoyment, optimism, or a generally warm and favorable message. A factual topic, ordinary setting, steady gaze, or composed posture does not by itself make favorable sentiment neutral.
- Do not default to `neutral` merely because the speaker is sitting still or the transcript describes an ordinary subject. If reliable spoken meaning, delivery, or visible warmth makes the overall message favorable, put the supplied positive label above neutral and cite those cues.
- Conversely, do not invent positivity from a generic face or from a topic that is merely mentioned. Use neutral when the record is genuinely informational and contains no favorable or unfavorable affect.
- Confidence must reflect the whole clip. Do not let many irrelevant appearance or posture observations outvote a small number of diagnostic affective cues.
- A label with no supporting id is allowed only when the record supports nothing. Output JSON only: no prose before or after it, no code fence, no comments.
