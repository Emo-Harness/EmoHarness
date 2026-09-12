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
- Every label must be copied exactly from the supplied list. Order two to four hypotheses from most to least likely.
- For an emotion question, identify the affect being expressed rather than counting neutral-looking objects, clothing, room details, or the number of posture observations.
- Frustration is supported by complaints, blocked goals, inability or helpless attempts, dissatisfaction, repeated disagreement, insistent or urgent questions, argumentative wording, and gestures showing agitation or effort. It can be present with a restrained face, flat pitch, slow speech, or a seated posture.
- Do not require shouting or a dramatic facial expression for `Frustration`. When the record describes an obstacle or repeated effort and the speaker is dissatisfied or pressed, prefer the supplied frustration label over `Neutral`.
- Distinguish frustration from `Anger` when hostility or attack is not dominant, and from `Sadness` when loss or withdrawal is not dominant. Use `Neutral` only when the affect is genuinely absent, not merely subtle.
- The task wording is not evidence. Confidence must reflect diagnostic cues, not raw item count. Output JSON only: no prose before or after it, no code fence, no comments.
