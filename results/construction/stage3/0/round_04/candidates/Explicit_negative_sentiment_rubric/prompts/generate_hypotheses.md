<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t1` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words or meanings that are not in the record.
- The task asks for the overall displayed emotion or sentiment, not a default judgment of neutral based on generic appearance.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`.
  - `label` must be one entry from the list above, copied exactly as written there.
  - `support` and `contradiction` must contain only evidence ids from the record.
  - `confidence` must be a number between 0 and 1.

Decision policy:

- First determine whether the noun is sentiment. For sentiment, identify the dominant evaluative polarity of the speech or scene rather than counting smiles, gaze direction, or neutral posture.
- Negative sentiment includes criticism, rejection, doubt, complaint, denial, dislike, disappointment, failure, harm, danger, loss, an unwanted situation, dismissive wording, or an unfavorable assessment. A speaker can sound calm, speak slowly, smile, or use a factual tone while still expressing negative sentiment.
- Treat explicit negative words, negation, a complaint, an unfavorable comparison, a statement that something is not good or not suitable, and concern about an adverse outcome as specific negative evidence. Do not discard that evidence merely because the face is controlled.
- Positive sentiment includes praise, approval, encouragement, gratitude, affection, humor, enjoyment, hope, benefit, meaningfulness, or a favorable judgment.
- Use neutral only when the record is genuinely descriptive or procedural and contains no specific positive or negative evaluation. A serious topic alone is not enough for negative, but a clear unfavorable evaluation is enough even without dramatic prosody.
- Do not use the task word or task item `t1` as emotional evidence. Do not infer polarity from a label's position in the list.
- For emotion rather than sentiment, use visible affect and prosody; do not force the sentiment rubric onto that task.
- When the transcript is not understood, do not invent its valence. Rely on unambiguous visual and prosodic cues and keep neutral only when no such cue exists.
- The most prominent label describes the overall clip or principal speaker. Do not let a minor background person or an isolated generic cue dominate.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Two to four of them is usual.
- A label with no supporting id is allowed only when nothing in the record supports any candidate.
- Do not cite `t1` as support for any emotion or sentiment.
- Output JSON only: no prose before or after it, no code fence, no comments.
