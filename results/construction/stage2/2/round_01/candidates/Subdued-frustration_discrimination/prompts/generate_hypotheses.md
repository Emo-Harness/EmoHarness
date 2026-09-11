<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and its modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard (the transcript, then the voice), and `t` from the wording of the question.
- An empty transcript means no intelligible speech was recorded. Do not invent words that are not there.
- Generic clothing, setting, posture, and ordinary turn-taking are not emotional evidence by themselves.

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
- Frustration may be subdued rather than loud. Treat explicit difficulty, inability, obstacles, repeated or impatient questions, complaints, disagreement, blocked plans, demanding instructions, and strained problem-solving as evidence for frustration when the label is available.
- Do not equate flat, quiet, slow, or polite delivery with neutral when the words describe an unresolved problem, repeated failure, inability, or persistent demand. Prosody modifies the strength of the reading; it does not erase clear frustrated content.
- Conversely, do not call ordinary conversation frustration merely because someone is speaking, gesturing, or asking one neutral question. Require at least one concrete problem, impediment, complaint, conflict, or tense interaction cue.
- Choose `neutral` only when the record lacks a reliable affective or problem-related cue, not merely because the face is composed or the body is still.
- If frustration and neutral compete, cite the concrete words, prosody, or tense movement supporting frustration and cite only truly contradictory evidence. Do not count generic appearance items as strong contradictions.
- Keep every label exactly as written in the candidate list, including capitalization and spelling.
- Output JSON only: no prose before or after it, no code fence, no comments.
