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
- For a sentiment request, distinguish negative valence from neutral factual narration. Direct criticism, rejection, complaint, threat, blame, distress, adverse evaluation, explicit dislike, or clearly negative delivery supports the negative entry when that entry is in the candidate list.
- A generic proper name, ordinary action, neutral setting, incomplete factual sentence, or transactional wording is not by itself negative and is not by itself evidence that the whole item is neutral.
- When a direct negative cue is present and the neutral case rests only on generic visual description or factual narration, rank the negative entry higher. Cite the negative wording, prosody, or visible affective cue, and retain neutral as an alternative when the evidence is mixed.
- Do not infer negative sentiment from loudness alone, and do not invent criticism that is absent from the transcript. If the record has a plausible negative cue, include both negative and neutral hypotheses rather than allowing a single unsupported neutral hypothesis to suppress the negative entry.
- Output JSON only: no prose before or after it, no code fence, no comments.

