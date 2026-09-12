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
- Treat ordinary posture, sitting, looking, clothing, setting, and generic mouth movement as incidental observations. They are not evidence for `neutral` merely because there are many of them.
- Treat direct facial affect as higher-value evidence than incidental description. A visible smile, smiling mouth, softened expression, laughter, or repeated positive expression supports a happiness-like candidate even when the person is otherwise seated, looking forward, or speaking normally. A slight smile is still evidence; do not require a broad grin or laughter.
- Use temporal persistence and change: a positive expression that is visible across the clip or recurs in several frames is stronger than one ambiguous mouth position. Do not guess a smile from an open mouth or from a pleasant setting.
- Neutral means that no meaningful affective cue is present or that the cues genuinely balance. Do not use Neutral as a default simply because the transcript is factual or because visual evidence contains many non-affective details.
- For an emotion question, classify the displayed affect, not the emotional valence of the topic. For a sentiment question, also consider an expressed positive or negative stance in the words and voice.
- Keep support specific: cite the direct cue ids, not every generic item in the record. Add contradiction ids when a cue genuinely argues against a candidate.
- Output JSON only: no prose before or after it, no code fence, no comments.

