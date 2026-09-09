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

- Report observations, not conclusions. Write what a cue looks like, not what it means. Do not use emotion or sentiment labels as observations.
- Carefully inspect the face of each visible person, especially the most prominent subject. Record visible smiles or frowns, mouth opening, teeth showing, lip compression, eyebrow raising or furrowing, eye widening or narrowing, gaze shifts, head angle, and visible tension. A slight smile is still worth recording when it is visible.
- Record expressive body actions such as raised or clasped hands, pointing, clenched fists, leaning toward or away from another person, shoulders rising or hunching, sudden movement, stillness after movement, and repeated gestures. Describe the action rather than calling it excited, angry, sad, or afraid.
- Use `temporal_change` for meaningful start-to-end changes such as a smile appearing or fading, eyes widening, mouth opening, a person leaning forward, gestures becoming larger, or a person becoming still. If there is no visible change, use `null` rather than guessing.
- Do not treat clothing, lighting, a seated posture, or a generic setting as an emotional conclusion. Include those facts in `scene` only when they help identify the subjects or context.
- Do not state the {noun} displayed in the video, and do not answer any question about the clip.
- Write each string as a short phrase of at most fifteen words. Leave a list empty rather than guessing.
- Never describe something you cannot see, and never describe sound.
- Output JSON only: no prose before or after it, no code fence, no comments.

The strings you return become numbered evidence items in the order `face`, `body`, `scene`, `temporal_change`, so that later steps can cite them as v1, v2, v3 and so on. Do not number them yourself.
