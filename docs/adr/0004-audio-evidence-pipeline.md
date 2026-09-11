# ADR-0004: Audio evidence pipeline

Date: 2026-09-06. Amended 2026-09-07 after the strict review; amended 2026-09-11 (describer removed). Status: accepted.

## Context

Qwen3-VL-8B accepts no audio. The benchmark ships `audio_clue/` files produced by Qwen2-Audio for the official judge. Those clues already contain emotion inferences ("indicating anger"). Feeding them to the Target would leak judge-side evidence and make Reasoning scores meaningless.

The review found two problems with the first version of this ADR. First, forbidding all emotion words in the audio description throws away information that audio-native competitors use; the leaderboard's Audio-Reasoner reaches 38.1 Recognition largely from audio. Second, the pipeline said nothing about clips without speech, clips with several speakers, or language detection, all of which occur in the benchmark.

## Decision

1. **The official `audio_clue/` files are never read** by any harness, Target, or Builder code path. A unit test asserts that no module imports or opens them, and the evaluation wrapper is the only code that passes them to the official judge.
2. **Two audio channels**, both produced locally and cached per clip:
   - *Observable-cue channel.* Whisper large-v3 (faster-whisper) transcript with word timestamps and detected language; openSMILE eGeMAPS functionals turned into short descriptors (pitch level and variability, loudness, speech rate, pause ratio) by fixed rules relative to per-corpus reference statistics; and a free-text description from an audio-language model that is **not** Qwen2-Audio, with every sentence that contains a candidate-label word or a word from a committed emotion lexicon removed. The default describer is Qwen3-Omni-30B-A3B-Captioner on the single 80 GB GPU; Qwen2.5-Omni-7B is the fallback.
   - *Audio-model opinion channel.* The same audio model, asked separately for coarse valence, arousal, and up to two candidate emotion words. This enters the evidence record tagged `source: audio_model_opinion` and is treated by the harness as one voter, never as evidence. Keeping it separate lets the "no audio-model opinion" ablation measure how much score comes from an audio model's judgment rather than from described cues.
3. **Clips without speech** get an empty transcript, no prosody descriptors, and a description restricted to non-speech sounds. The prompts instruct the Target that an empty transcript means no intelligible speech and that it must not invent words.
4. **Clips with several speakers** are diarized (pyannote). The dominant speaker by speech duration is treated as the subject; the Builder may replace this heuristic through `routing.yaml` at Stage 2.
5. **Language** detected by Whisper is stored in the manifest and used for the per-language analysis (ADR-0013), never for routing that names a task.
6. **Same-family check (P1).** One extra row runs Qwen2-Audio-7B in place of the default describer to quantify how much Reasoning score comes from aligning with the judge's audio agent rather than from evidence.

## Amendment of 2026-09-11: no audio-language model

Decision 2's free-text description and the whole audio-model opinion channel
are withdrawn, and decision 6 (the Qwen2-Audio same-family row) with them. The
reason is a design rule, not a measurement: every perception tool in the
harness must be deterministic code (frame sampling, whisper transcription,
openSMILE functionals turned into phrases by fixed rules). Asking a second
model, Qwen2.5-Omni-7B in the v1 run, to describe the clip and to vote on its
valence and arousal put another model's judgment into the evidence record, so
part of any gain would have been that model's, not the Builder's. The
observable-cue channel is now exactly the transcript (with language) and the
prosody phrases; the evidence ids are `v*`, `a*` (transcript sentences, then
prosody), and `t1`. The emotion lexicon filter, which only existed to scrub
the describer's text, is removed with it. The v1 results that used the
describer are kept under `results/archive/v1-omni-describer/`.

## Consequences

- Audio artifacts are cached per clip and shared by every method, including the Gemini matched-protocol run and every ladder rung from L1 upward (ADR-0009).
- The emotion lexicon and the describer prompts are frozen at pre-registration (ADR-0012).
- The audio-model opinion channel is a legitimate tool output, not a gold-label leak: it comes from a model with no access to labels.
