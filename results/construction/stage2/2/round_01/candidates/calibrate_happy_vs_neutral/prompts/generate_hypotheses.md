<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders: {prompt}, {quantifier}, {audio_text}. -->

You are the final evidence analyst for one short video clip. Read the evidence record carefully and select the most prominent {noun} from the candidate list.

EVIDENCE RECORD

{evidence_json}

The answer must be copied from this list exactly, including capitalization and punctuation: {label_set}

Return one JSON object with exactly these keys:
- `coarse`: a short description of the overall affective state.
- `hypotheses`: two to four objects, ordered from most to least likely. Each object has `label`, `support`, `contradiction`, and `confidence`.

Rules:
- Every label must be copied exactly from the candidate list.
- Use only evidence ids that occur in `items`; do not invent ids or duplicate an id within one support or contradiction list.
- Treat `audio_opinion` as one fallible voter, not as an observation. It must not override clear visual or spoken evidence by itself.
- Interpret the transcript and observable behavior, not just pitch, pace, volume, or the absence of laughter.
- Do not default to a neutral label merely because the delivery is calm.

Calibration for the `happy` versus `neutral` distinction (case-insensitive label matching, while emitting the exact listed spelling): when both labels are available, a visible smile, repeated pleasant facial expression, warm social engagement, or clearly approving/pleasant behavior is affirmative evidence for `happy`, even when prosody is flat and laughter is absent. Do not require high arousal or laughter. Reserve `neutral` for clips lacking affirmative affective cues, or for a merely ordinary speaking posture with no smile or warmth. A single ambiguous open mouth or generic movement is not enough for `happy`.

Set confidence according to the strength and consistency of the evidence. Do not cite an audio opinion as though it were observed speech. Output JSON only, with no prose or code fence.