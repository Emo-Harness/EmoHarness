Re-examine an earlier reading of one video clip. Start from what was heard, then check it against what was seen.

WHAT WAS HEARD

{audio_text}

FULL EVIDENCE RECORD

{evidence_json}

EARLIER RANKING

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The answer must be copied from this candidate list, character for character: {label_set}

Work in this order:

1. Decide what the heard cues on their own support, ignoring the earlier ranking.
2. Decide what the seen cues on their own support.
3. Compare those readings, inspect the strongest alternative, then re-order and re-score. Check the complete candidate list for a plausible reading omitted earlier; add it only if observations support it.

Additional audit when the requested noun is emotion:

- Separate valence from arousal. Subdued delivery is not necessarily neutral emotion, and sadness does not require crying or explicit mention of sadness.
- Read flat pitch, quiet voice, slow speech, and long pauses as a combined delivery pattern when they co-occur. They are not independent votes for a label and are not a deterministic diagnosis.
- Compare ordinary calmness with withdrawal, resignation, grief, and distress. Relevant discriminators include a downturned or tense mouth, lowered gaze accompanied by facial tension, hunched or withdrawn posture, hesitant or broken speech, expressed loss, helplessness, or a farewell. Use only cues actually present.
- If several subdued cues co-occur with an observed negative or withdrawn presentation, explicitly compare the candidate for sadness with neutral before accepting neutral. Do not discard sadness because the words are a short routine response.
- A single low pitch, slow rate, downward glance, seated posture, or factual sentence does not establish sadness. Without a discriminating cue, remain uncertain rather than forcing a negative emotion.
- A flat or quiet voice does not contradict happiness, appreciation, fear, or frustration by itself. Distinguish competing low-arousal states using the full presentation and meaning.
- Preserve strongly supported emotional self-description or clear expressive behavior. Do not override it just because another modality looks ordinary.

Additional audit only when the requested noun is sentiment and the transcript contains an evaluation of a product, work, experience, performance, or outcome:

- Identify who owns the evaluation and what it evaluates. The speaker's own judgment is relevant; a plot event, quotation, or report of other people's preferences is not automatically the speaker's sentiment.
- Inspect ratings together with their stated reasons. A rating explicitly reduced because of an objection supports unfavorable evaluation even when delivered calmly. Do not assume a rating scale or polarity from an isolated number.
- Inspect comparative restrictions and reservations. Praise limited to a single feature can accompany criticism of the rest. Determine whether words such as only, despite, or but actually express a contrast in the available utterance; their presence alone does not establish negative sentiment.
- Separate a factual description from a qualified judgment. When the utterance plainly conveys disapproval, a complaint, a poor assessment, or disappointment with the evaluated target, compare the negative candidate with neutral rather than treating declarative grammar as neutrality.
- Likewise, preserve clearly attributable favorable assessments. Do not turn balanced or reserved praise into negative sentiment unless the observed reservation dominates the evaluation.
- Quiet or flat delivery is not a contradiction of a clear evaluation. A smile may be conversational rather than approval of the evaluated target. Use facial and acoustic observations to interpret ambiguity, not to erase an unambiguous judgment.
- If the utterance is merely descriptive, incomplete, or ambiguous about the speaker's stance, do not manufacture an evaluation. Keep the supported reading and calibrate confidence.
- This audit does not apply merely because the topic is unpleasant, political, religious, sexual, or emotionally charged. It requires an attributable evaluation in the observations.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when informative heard and seen cues favor the same candidate, `false` when they favor different candidates.
- `hypotheses`: objects with `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored after this second reading. Confidence must be between 0 and 1.

Rules:

- Keep every label exactly as written in the candidate list.
- Cite only distinct observed ids that appear in `items`. Never cite t1 or other task-context items.
- An empty transcript supplies no spoken meaning. An empty prosody list supplies no acoustic conclusion. If speech is not established, do not assume acoustic phrases describe the foreground person's voice rather than other sound.
- Support must identify observations that favor the reading, not generic appearance or the question wording. Contradiction must identify genuinely opposing observations, not merely missing expressivity.
- Include multiple distinct supporting observations when they exist, but do not duplicate or pad evidence.
- Missing information is not evidence for neutral. Low confidence is appropriate when observations are sparse.
- When informative modalities disagree, keep the supported alternatives and calibrate confidence; do not hide conflict by dropping a hypothesis.
- When one modality is uninformative, do not invent a competing label for it; agreement can remain true.
- Output JSON only: no prose, code fence, or comments.

DETERMINISTIC CHECK (context, not citable evidence):
{verification_json}
Use this check to locate possible disagreements. Reconsider readings against the observed cues, never against the check as if it were evidence.
