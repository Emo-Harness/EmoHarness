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
- Spend most of the available observations on discriminative affective cues: smile shape, raised or furrowed brows, eye widening or narrowing, gaze direction, lip compression, mouth opening, visible tension, laughter-like facial movement, tears, head position, and clear changes in expression.
- When a person smiles, record the visible smile precisely rather than replacing it with generic mouth movement. When an expression is not smiling, record the visible alternative only when it is actually present.
- For clips with more than one person, identify which person shows each facial or bodily cue when possible. Do not average different people into one emotion.
- Use `temporal_change` for a real start-to-end change such as neutral-to-smiling, smiling-to-neutral, widening eyes, lowering gaze, or increasing gestures. Say that cues persist only when they are visible across the clip.
- Do not fill `face` or `body` with clothing, equipment, room layout, ordinary sitting, or generic speaking when a more diagnostic visible cue is available. Put incidental facts in `scene` only if they help identify what is happening.
- Never describe something you cannot see, and never describe sound.
- Write each string as a short phrase of at most fifteen words. Leave a list empty rather than guessing.
- Output JSON only: no prose before or after it, no code fence, no comments.

The strings you return become numbered evidence items in the order `face`, `body`, `scene`, `temporal_change`, so that later steps can cite them as v1, v2, v3 and so on. Do not number them yourself.
