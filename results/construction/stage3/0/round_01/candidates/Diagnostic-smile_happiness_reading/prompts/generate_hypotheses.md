<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and one starting with `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The `t` item identifies the task and is not evidence of an emotion.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with `label`, `support`, `contradiction`, and `confidence`.

Rules:

- Use only ids that appear in `items`; never invent an id or cite an id twice in one list.
- Every label must be copied exactly from the supplied list. If the list contains `happy`, `happiness`, or another synonymous spelling, copy the one that actually appears.
- For emotion recognition, weigh diagnostic facial affect more than incidental scene and posture facts. A slight or sustained smile, smiling eyes, pleasant amusement, warm engagement, or visibly cheerful expression is evidence for the supplied happiness label even when the person is seated and otherwise still.
- Do not let many generic observations such as looking forward, sitting, clothing, background, or mouth movement outvote one clear smile. Use `Neutral` only when the face and delivery lack a positive affective cue.
- Distinguish happiness from surprise: an open mouth or raised brows alone may indicate surprise, but a smile or pleasant expression that persists or appears at the end supports happiness. Use the strongest overall reading of the clip, including temporal change.
- Do not infer an emotion from the task wording. Confidence must reflect cue specificity rather than raw support count. Output JSON only: no prose before or after it, no code fence, no comments.
