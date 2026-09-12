<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and one starting with `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The `t` item identifies the task and is not evidence of sentiment.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`.

Rules:

- Use only ids that appear in `items`; never invent an id or cite an id twice in one list.
- Every label must be one entry from the supplied list, copied exactly. Order two to four hypotheses from most to least likely.
- For a sentiment question, classify the polarity of the communicated message and delivery, not just the speaker's facial composure. Criticism, complaint, rejection, threat, conflict, distress, regret, disturbing or ominous subject matter, dismissive wording, and unfavorable evaluation support the supplied negative label.
- A calm face, steady posture, factual delivery, or ordinary setting does not make a clearly unfavorable message neutral. Conversely, do not call a purely informational statement negative merely because its topic is serious; require an unfavorable attitude, consequence, or affective framing.
- When a speaker uses criticism or a negative evaluation with little visible expression, prefer the supplied negative label over `Neutral` and cite the semantic or prosodic cue. Do not let irrelevant appearance observations outvote diagnostic sentiment cues.
- The task wording is metadata, not evidence. Confidence must reflect the whole message and not raw item count. Output JSON only: no prose before or after it, no code fence, no comments.
