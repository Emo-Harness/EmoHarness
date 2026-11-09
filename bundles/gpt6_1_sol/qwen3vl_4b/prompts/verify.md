<!-- Rendered with str.format_map. -->
<!-- Placeholders: {noun}, {label_set}, {audio_text}, {evidence_json}, {hypotheses_json}, {verification_json}. -->

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

1. Decide what the heard cues on their own support, ignoring the earlier ranking.
2. Decide what the seen cues on their own support.
3. Compare the readings, then re-order and re-score the hypotheses against observed evidence.

TARGETED AUDIT: CALM FAVOURABLE SENTIMENT

Apply this audit only when the requested noun is sentiment and the earlier first-ranked label is neutral. Otherwise use the ordinary verification procedure above.

- Sentiment is the speaker's conveyed favourable, unfavourable, or uncommitted attitude; it does not require an overt named emotion. Calm delivery can accompany a favourable attitude.
- Look for an overlooked favourable evaluation: endorsement, appreciation, approval of an idea, a stated benefit, amusement, or satisfaction with an outcome. Evaluate meaning in the transcript's language, including negation, contrast, and who holds the opinion.
- A positively evaluated idea can support positive sentiment even when the speaker is explaining it matter-of-factly. Do not require laughter, excitement, the word happy, or an explicit statement of feeling.
- Check visible mouth corners, cheek movement, and gestures for warmth or pleased engagement if those observations exist. Do not invent a smile from direct gaze, clothing, or the setting.
- An empty prosody list means no diagnostic prosody was supplied, not that a neutral voice was heard. Fast speech alone establishes neither positive nor negative valence.
- Preserve neutral for ordinary factual narration, descriptions of past activities, isolated nouns, and fragments without a supported evaluative attitude. A pleasant topic alone is not approval. Do not transfer another person's reported feelings to the speaker.
- If a favourable reading has genuine observed support that the earlier ranking overlooked, include the corresponding candidate-list label even if it was previously absent or unsupported. Shared ambiguous observations may support both neutral and positive; reserve the higher confidence for the reading with the better diagnostic cue.
- Change the earlier first choice only when identifiable observed approval or pleasant affect favours the alternative. Missing evidence is not a reason to force positive, and mixed valence must be assessed across the whole available clip.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: `true` when the heard and seen cues point to the same entry, `false` when they point to different entries. Missing evidence on one side is not an opposing reading.
- `hypotheses`: objects with `label`, `support`, `contradiction`, and `confidence`, ordered from most to least likely after verification.

Rules:

- Copy every label exactly from the candidate list. You may add a previously omitted candidate when observed evidence warrants it.
- Cite only distinct ids present in `items`. Never cite t1 or any task-context item as support or contradiction.
- Each supporting id must actually favour its label; merely existing in the record is insufficient.
- Contradiction means evidence against a reading, not merely evidence compatible with another reading.
- When genuine readings disagree, reflect that in `modalities_agree` and confidence; do not hide the conflict by dropping a supported hypothesis.
- Confidence is between 0 and 1. Do not manufacture certainty when only sparse evidence is available.
- Output JSON only: no prose, code fence, or comments.

DETERMINISTIC CHECK (context, not citable evidence):
{verification_json}
Use this check to locate disagreements, not as proof that an interpretation is correct. Cite observed cues, never the check itself or t1.
