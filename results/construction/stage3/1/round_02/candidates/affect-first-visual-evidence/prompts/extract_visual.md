<!-- Rendered with str.format_map. Placeholders: {noun}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {label_set}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

You are given a short video clip. Describe only what can be seen in it.

Return a single JSON object and nothing else, with these keys:

- `face`: a list of short strings, each one visible cue from facial expression, eyes, eyebrows, gaze, or mouth shape.
- `body`: a list of short strings, each one visible cue from posture, hands, gesture, movement, or bodily tension.
- `scene`: a list of short strings, each one relevant fact about the setting, the other people, or what is happening.
- `temporal_change`: one short string describing how the visible cues change between the start and the end of the clip, or `null` when they do not change.

Rules:

- Report observations, not conclusions. Write what a cue looks like, not what it means.
- Prioritize direct affective observations: smiling or frowning, raised or knitted brows, widened or narrowed eyes, gaze avoidance, visible tension, relaxedness, laughter-like facial change, crying-like facial change, expressive gestures, and changes in these cues.
- Describe the expression of the person who is speaking or otherwise central to the clip. If multiple people are visibly expressive, identify each without inferring their private feelings.
- Do not state the {noun} displayed in the video, and do not answer any question about the clip.
- Do not use ordinary sitting, clothing, hair, equipment, room layout, or generic mouth movement as evidence of an emotion. Include such details in `scene` only when they are relevant to what is happening.
- Do not call a person neutral merely because a single frame lacks a smile. Report the visible expression and whether it persists or changes across the clip.
- Write each string as a short phrase of at most fifteen words. Leave a list empty rather than guessing.
- Never describe something you cannot see, and never describe sound.
- Output JSON only: no prose before or after it, no code fence, no comments.

The strings you return become numbered evidence items in the order `face`, `body`, `scene`, `temporal_change`, so that later steps can cite them as v1, v2, v3 and so on. Do not number them yourself.
