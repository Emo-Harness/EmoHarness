<!-- Rendered with str.format_map. -->

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

1. Decide what the heard cues on their own point to, ignoring the earlier ranking. Separate transcript meaning, the speaker's conversational stance, and measured prosody.
2. Decide what the seen cues on their own point to. Attribute cues to the same person when possible; do not blend the listener's smile with the speaker's delivery.
3. Compare the two readings, then re-order and re-score the ranking. Check the full candidate list for a plausible supported reading that was omitted.

Focused comparison ONLY for emotion requests whose candidates distinguish excitement from happiness:

- First establish whether the displayed state is positive. Then distinguish positive activation from quiet contentment; positive valence alone does not settle the label.
- Consider excitement when the speaker displays eager anticipation, enthusiastic affirmation, celebratory engagement, or heightened positive energy. Supporting cues may include emphatic repetition, animated gestures, a broad changing smile, or fast and variable delivery in a positive context.
- A brief enthusiastic response or vividly anticipated desirable outcome can support excitement even without the literal word excited. Cite the observed speech or movement that makes this interpretation plausible.
- Consider happiness when the display is primarily satisfaction, warmth, amusement, or settled contentment without stronger eager activation.
- Explicit excitement or celebration should not be downgraded merely because happiness shares the smile evidence. Conversely, a reflective statement of personal contentment should not become excitement merely because it is favorable.
- A smile alone, routine conversational gestures, a simple acknowledgment, or isolated fast speech does not establish excitement. Prosody rules are coarse measurements: missing prosody is unknown, not proof of low arousal.
- Do not assign the speaker an emotion merely because the transcript names someone else's happiness or surprise.
- When the earlier winner is happiness, actively compare it with the available excitement candidate. Add that candidate if genuinely supported, but do not promote it without a discriminating cue.

Return a single JSON object and nothing else, with these keys:

- `modalities_agree`: true when informative heard and seen cues favor the same entry, false when they favor different entries. An absent or uninformative side does not establish a conflict.
- `hypotheses`: two to four distinct candidate labels with `label`, `support`, `contradiction`, and `confidence`, re-ordered and re-scored. Fewer are allowed when the candidate list is smaller.

Rules:

- Keep every label exactly as written in the candidate list. A missing supported candidate may be added.
- Use only distinct observed ids present in the record. Never cite text-modality items or t1 as evidence.
- Support must favor the reading, not merely describe the scene. A contradiction must be an opposing observation, not missing evidence or unrelated appearance.
- Preserve strong existing decisions unless a more discriminating observed cue warrants changing them.
- When informative modalities disagree, retain plausible alternatives and express the conflict through `modalities_agree`. Lowering every confidence equally does not resolve the ranking; compare cue relevance.
- Output JSON only: no surrounding prose, code fence, or comments.

DETERMINISTIC CHECK (context, not citable evidence):
{verification_json}
Use this check to locate possible disagreements. It is not evidence or a verdict; reconsider readings only against observed cues.
