<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- Treat generic appearance, setting, and ordinary movement as weak evidence unless they clearly carry affect.

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
- For a sentiment request, judge the overall conveyed valence of the words, situation, and delivery, not just whether the face is smiling and not just whether the speaker sounds calm.
- Positive sentiment can be calm. Praise, encouragement, gratitude, enjoyment, success, favorable outcomes, constructive framing, and pleasant or affirming interaction are positive evidence even without an explicit emotion word.
- Negative sentiment can also be calm. Criticism, rejection, failure, harm, complaint, threat, distress, or unfavorable evaluation are negative evidence.
- Do not choose `neutral` merely because the clip is factual, the transcript is a fragment, the speaker is composed, or the visual description contains no dramatic gesture. Choose neutral only when the available record is genuinely without reliable positive or negative valence.
- When positive and neutral are both candidates, do not let many generic visual descriptions outweigh one or more concrete positive semantic cues. Cite the concrete cues that support the positive reading and cite genuinely contrary cues, if any.
- For an emotion request, do not substitute sentiment valence for a specific displayed emotion; use the emotion-specific facial, bodily, temporal, and prosodic cues instead.
- Keep every label exactly as written in the candidate list, including capitalization and spelling.
- Output JSON only: no prose before or after it, no code fence, no comments.
