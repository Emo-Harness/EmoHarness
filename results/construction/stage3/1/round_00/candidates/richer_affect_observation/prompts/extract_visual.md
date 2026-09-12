<!-- Rendered with str.format_map. Placeholders: {noun}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {label_set}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

You are given a short video clip. Describe only what can be seen in it.

Return a single JSON object and nothing else, with these keys:

- `face`: a list of short strings, each one visible cue from the face or gaze.
- `body`: a list of short strings, each one visible cue from posture, hands, or movement.
- `scene`: a list of short strings, each one fact about the setting, the other people, or what is happening.
- `temporal_change`: one short string describing how the visible cues change between the start and the end of the clip, or `null` when they do not change.

Rules:

- Report observations, not conclusions. Write what a cue looks like, not what it means.
- Do not state the {noun} displayed in the video, and do not answer any question about the clip.
- Inspect the face across the clip, not just one frame. When visible, record affect-relevant physical details such as a smile or frown, raised or furrowed brows, widened or narrowed eyes, gaze direction, mouth opening, lip tension, head position, and whether the expression changes.
- Also record visible affect-relevant body details such as leaning, withdrawal, open or clenched hands, repetitive movement, stillness, expansive gestures, or a change in posture. Describe the movement itself rather than calling it happy, sad, angry, or surprised.
- A small or partial cue is worth recording when it is genuinely visible, but do not turn ordinary appearance, clothing, lighting, or a plain background into emotional evidence.
- Write each string as a short phrase of at most fifteen words. Leave a list empty rather than guessing.
- Never describe something you cannot see, and never describe sound.
- Output JSON only: no prose before or after it, no code fence, no comments.

The strings you return become numbered evidence items in the order `face`, `body`, `scene`, `temporal_change`, so that later steps can cite them as v1, v2, v3 and so on. Do not number them yourself.

