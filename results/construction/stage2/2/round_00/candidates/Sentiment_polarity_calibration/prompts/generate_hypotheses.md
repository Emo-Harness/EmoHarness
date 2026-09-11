<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- The question wording and the item whose cue is the noun being asked about are not emotional evidence. Do not treat `t` items as support for a label.

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
- A label must have at least one genuine supporting cue when such a cue exists; do not make a label the leader merely because it is the default or because the face is hard to read.
- For sentiment questions, judge the overall evaluative polarity of the spoken message and visible situation, not just facial expressiveness. Encouragement, praise, gratitude, reassurance, hope, enjoyment, approval, and other clearly favorable wording are positive evidence even when the speaker looks composed or the clip is an advertisement or review. Factual or descriptive speech with no favorable or unfavorable evaluation is neutral. When clear favorable evidence is present and there is no comparably strong negative evidence, the positive entry in the candidate list outranks the neutral entry.
- For emotion questions, use visible affect and vocal delivery rather than the noun in the question. Keep the distinction between a calm positive state and a neutral state, but do not erase a genuine smile, warmth, amusement, or other positive expression simply because it is subtle.
- Output JSON only: no prose before or after it, no code fence, no comments.

