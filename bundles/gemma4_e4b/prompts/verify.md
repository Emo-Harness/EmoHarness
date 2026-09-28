<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Re-examine an earlier reading of one video clip. Start from what was heard, then check it against what was seen.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Work in this order:

1. Decide what the heard cues on their own point to, ignoring the earlier ranking. Separate the speaker's affect and delivery from the mere subject matter. A technical problem, difficult situation, serious topic, question, complaint about circumstances, or concerned advice does not by itself prove frustration or fear. Fear requires observed apprehension, threat, vulnerability, inability to cope, urgent uncertainty, dependence on another person, or a delivery that makes such apprehension audible; do not invent it from a neutral fact.
2. Decide what the seen cues on their own point to. Give direct facial expression, gaze, posture, gestures, and temporal change priority over clothing, setting, or the fact that someone is speaking. Ordinary looking, composed posture, and absence of a visible affective change support a neutral reading when no stronger visible cue is present. Wide eyes, protective or recoiling posture, visible distress, or an observable change toward alarm can support fear, but ordinary looking or looking away is not enough.
3. Compare the two readings. When audio suggests tension but the visible people remain composed and there is no explicit affective vocal wording, keep neutral as a serious alternative and do not convert topic difficulty into displayed frustration. When the transcript explicitly expresses irritation or the voice and visible behavior show agitation, frustration can outrank neutral. When the words or delivery express immediate uncertainty, threat, inability, dependence, or apprehension, fear can outrank neutral even if the visual evidence is limited; otherwise do not use fear as a generic synonym for seriousness. Do not use the agreement flag as the answer.
4. Re-order and re-score the ranking, preserving every plausible candidate supported by observed evidence. Do not drop fear merely because the evidence is sparse when a specific observed cue supports it, and do not add fear when no observed cue supports it.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard cues and the seen cues point to the same entry, `false` when they point to different entries.
- `hypotheses`: the same kind of list as the earlier ranking, with the keys `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading.

Rules:

- Keep every label exactly as it is written in the list above.
- Use only ids that appear in the record; drop any id you cannot find there.
- When the two readings disagree, say so through `modalities_agree` and lower the confidences; do not hide the conflict by dropping a hypothesis.
- Do not treat a factual or technical transcript as evidence of frustration or fear unless its wording or delivery expresses the corresponding affect.
- Output JSON only: no prose before or after it, no code fence, no comments.

DETERMINISTIC CHECK (context, not citable evidence):
{verification_json}
Use this check to locate disagreements. Cite observed cues, never the check itself
or t1 (the question). Reconsider conflicting readings only against observed cues.

