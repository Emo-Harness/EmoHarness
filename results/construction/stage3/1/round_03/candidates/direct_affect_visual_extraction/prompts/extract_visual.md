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
- Prioritize direct affect-relevant observations before incidental detail. In `face`, record visible smiles or frowns, tightened or relaxed mouths, brow position, eye widening or narrowing, gaze aversion, grimaces, and clear expression changes when they are actually visible. In `body`, record tense or relaxed posture, leaning, withdrawal, agitation, open or emphatic gestures, hands covering the face, and changes in movement when they are actually visible.
- Describe the cue without naming an emotion: write `brows drawn together` rather than an emotion label. Do not infer that a person is frustrated, happy, afraid, or neutral.
- Ordinary sitting, clothing, room layout, identity, camera angle, and generic mouth movement are low-value observations. Keep them in `scene` only when they clarify the interaction, and do not let them replace direct face or body cues.
- Compare the beginning and end of the clip. Record a meaningful increase, decrease, onset, release, or change in expression or gesture in `temporal_change`; use `null` when no such change is visible.
- Never describe something you cannot see, and never describe sound.
- Write each string as a short phrase of at most fifteen words. Leave a list empty rather than guessing.
- Output JSON only: no prose before or after it, no code fence, no comments.

The strings you return become numbered evidence items in the order `face`, `body`, `scene`, `temporal_change`, so that later steps can cite them as v1, v2, v3 and so on. Do not number them yourself.
