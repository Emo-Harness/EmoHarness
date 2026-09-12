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
- For emotion recognition, identify the displayed affect, not merely the topic or the fact that someone is speaking.
- Treat frustration as a real candidate when a direct cue shows thwarting or interpersonal strain: repeated or challenging questions, arguing or correction, inability or confusion about how to proceed, impatience, exasperated gestures, tense leaning, or a pressured, strained, or persistently agitated delivery. Cite the specific cue that supports it.
- Do not infer frustration from a question alone, ordinary hand movement, a factual topic, a calm conversation, or the mere presence of speech. If no direct affective cue exists, Neutral is appropriate.
- Do not let a long list of clothing, setting, sitting, gaze, or other incidental observations outweigh one specific affective cue. A hypothesis should cite only observations that genuinely support it; do not use incidental items as support for Neutral.
- When frustration and Neutral are both plausible, prefer the reading with the more specific direct affective cue and sustained change, while retaining the alternative when the evidence is genuinely ambiguous.
- Output JSON only: no prose before or after it, no code fence, no comments.
