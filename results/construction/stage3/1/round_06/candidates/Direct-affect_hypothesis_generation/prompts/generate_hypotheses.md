<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Below is the evidence record collected from one video clip.

EVIDENCE RECORD

{evidence_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this list, character for character: {label_set}

Read the record in two separate passes before ranking anything.

First pass: identify direct affective evidence. For visual evidence, prioritize facial expression, gaze, expression change, posture under tension, meaningful gestures, and temporal change. For audio evidence, prioritize the actual words and clear vocal affect. A short utterance can still be emotionally displayed; do not require a long explanation. When the task is sentiment, classify the speaker's expressed appraisal or stance, not the topic being discussed.

Second pass: remove incidental evidence. Clothing, hair, sensors, room layout, furniture, ordinary sitting or standing, generic mouth movement, ordinary looking, and the number of scene descriptions are not emotional support. Do not let a long list of such observations make Neutral win. A calm face does not erase an explicit approving, rejecting, complaining, threatening, regretful, or otherwise evaluative statement. Conversely, a serious topic, an advertisement, a question, or energetic delivery without an appraisal is not automatically negative or positive.

For emotion questions, distinguish the displayed affect from the subject matter. Mild irritation, impatience, obstruction, repeated questioning, argumentative wording, or tense gestures can support a frustration-like candidate even when the face is restrained. Sustained low energy, helplessness, fear, sadness, amusement, surprise, or happiness should only be used when the record contains cues for them. Generic prosody alone is corroboration, not decisive evidence.

For sentiment questions, look for explicit approval, praise, recommendation, enjoyment, gratitude, encouragement, endorsement, criticism, rejection, complaint, regret, dismissal, insult, threat, or unfavorable evaluation in the words that are actually present. Factual narration and an incomplete fragment must not be upgraded merely because their subject sounds good or bad.

Return a single JSON object and nothing else, with these keys:

- coarse: one short phrase for the overall state, without inventing facts.
- hypotheses: a list of two to four objects, each with label, support, contradiction, and confidence.

Use only labels from the list above. Use only evidence ids that appear in items, without duplicates within a support or contradiction list. Give support only when the cited observation genuinely favors that reading; do not cite incidental observations just to increase a count. Give contradiction when direct evidence points away from the reading. Include the strongest direct reading and the strongest meaningful alternative, even when the alternative is only mildly supported. Include Neutral only when no candidate has clear direct affective support, or as a competing alternative when the evidence is genuinely mixed. Order the hypotheses from most to least likely and calibrate confidence to the quality of the evidence rather than to the amount of incidental detail.

Output JSON only: no prose before or after it, no code fence, no comments.
