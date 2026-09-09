<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

How to read the record:

- `items` lists every citable observation with its id and modality. An id starting with `v` came from what was seen, one starting with `a` from what was heard, `o` from a separate audio model, and `t` from the wording of the question.
- An `audio_opinion` item is one fallible voter, not an observation. Do not let a neutral audio opinion override a direct visible expression, meaningful spoken content, or distinctive prosody.
- Generic descriptions such as seated, speaking, looking forward, moderate pitch, no laughter, or no background noise are not evidence of neutral emotion. They describe the recording conditions and should not outweigh a smile, frown, tense movement, widened eyes, distress, laughter, an agitated delivery, or emotionally meaningful words.
- Repeated generic audio-description items are not independent votes. Prefer the few strongest, specific cues over a large count of redundant cues.
- An empty transcript means no intelligible speech was recorded. Do not invent words or translate an unknown utterance.

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Interpret the task carefully:

- If the noun is emotion, identify the most prominent displayed affect. Visible facial action, gaze, posture, gesture, movement, and change over time are primary; vocal affect and the meaning of speech are supporting evidence when they concern the person shown. Calm posture alone does not prove neutral.
- If the noun is sentiment, judge the overall affect conveyed by the clip, including the meaning of speech and the visible interaction. A calm delivery can still convey positive or negative sentiment when the words or interaction clearly do so.
- A visible smile is positive evidence for a happiness-like candidate even when it is slight. A wider smile, laughter, enthusiastic movement, or high arousal can favor an excited-like candidate over a merely happy candidate. Do not discard a positive label solely because the speaker is seated or the audio opinion says neutral.
- Use a negative label for clear distress, hostility, aversion, fear, disappointment, or frustration when the record supports it. Distinguish frustration or blocked effort from sadness: complaints about inability, repeated objections, urgent questioning, or agitated gestures favor frustration; loss, resignation, low-energy distress, and mournful or withdrawn cues favor sadness.
- Reserve neutral for a genuinely low-salience presentation after considering specific facial, bodily, vocal, textual, and temporal cues. It is not a safe default merely because evidence is imperfect.
- Use the candidate label whose meaning best fits the strongest evidence. Never substitute a synonym, change capitalization, or use a label not present in the list. Use `Other` only when no listed specific label fits.

Return a single JSON object and nothing else, with these keys:

- `coarse`: one short phrase for the overall state you read from the record, in your own words.
- `hypotheses`: a list of objects, each with
  - `label`: one entry from the list above, copied exactly as written there,
  - `support`: a list of item ids from the record that speak for that entry,
  - `contradiction`: a list of item ids from the record that speak against it,
  - `confidence`: a number between 0 and 1.

Rules:

- Use only ids that appear in `items`. Never invent an id and never cite an id twice in the same list.
- Order the hypotheses from most to least likely. Put the best-supported label first and make the confidence reflect the strength of the specific cues, not the number of redundant generic items.
- Include two to four plausible labels when the evidence is ambiguous. Include a neutral hypothesis only when the record gives it real support.
- A label with no supporting id is allowed only when nothing in the record supports anything.
- Output JSON only: no prose before or after it, no code fence, no comments.
