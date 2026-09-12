<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The task-word item `t1` is not evidence for any emotion.

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
- Determine which person is the main target and which emotion is most prominent over the clip. Do not average a smiling background person with a neutral speaker.
- Do not make `Neutral` the default. Generic facts such as sitting, looking, clothing, speaking, or a plain room are weak evidence and cannot by themselves support Neutral or contradict a specific affective cue.
- When the candidate list contains `happy`, `happiness`, or `joy`, treat a sustained smile, smiling with relaxed eyes, laughter, visible amusement, warm affiliative engagement, playful pleasure, or a consistently positive interaction as direct support. A slight but repeated smile is still evidence; it need not be an exaggerated grin.
- Do not require positive words or loud speech for happiness. Conversely, do not promote a single fleeting mouth movement or an ambiguous open mouth when the rest of the clip is controlled.
- Distinguish happiness from surprise: a transient open mouth or raised eyebrows may support Surprise, but a smile that persists or becomes the endpoint supports happiness when no stronger negative cue is present.
- Use the temporal change field: a positive endpoint or a smile that develops and persists can outweigh earlier neutral frames.
- If the task is sentiment rather than emotion, classify the evaluative polarity of the content: praise, approval, encouragement, benefit, gratitude, or favorable judgment support positive; criticism, rejection, complaint, insult, threat, harm, or unfavorable judgment support negative. Use neutral only for genuinely non-evaluative material.
- Prefer specific affective observations over the number of generic observations. Confidence must reflect the strongest cues and must not be inflated for Neutral merely because the evidence list is long.
- Keep every label exactly as written in the candidate list.
- Output JSON only: no prose before or after it, no code fence, no comments.
