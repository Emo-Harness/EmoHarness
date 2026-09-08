# ADR-0015: Teacher switched to GPT 5.6 Luna; Qwen3-only model set

Date: 2026-09-07. Status: accepted. Supersedes the model choices in ADR-0004, ADR-0007, ADR-0008, ADR-0009, and ADR-0011 where they name Gemini as the Builder or upper bound, or any Qwen2.x model.

## Context

A previous construction run with Gemini 3.1 Pro as Builder and Qwen3-VL-8B as Target improved the Target by roughly 20% over five to six rounds. The next run replaces the teacher with GPT 5.6 Luna to test whether the Builder gain depends on the Builder, and standardizes every auxiliary model on the Qwen3 family.

## Decision

1. **Builder:** GPT 5.6 Luna through the OpenAI chat-completions API (`configs/builder/`, `provider: openai`). The Builder client is provider-agnostic; Gemini remains an alternative provider, not a default. The exact API model id and pricing are placeholders until confirmed.
2. **Upper bound (ladder rung L6):** the same teacher answering directly. The OpenAI API accepts images, not video, so the teacher row always runs the matched protocol: the shared cache's 2 fps frames plus the audio-evidence text appended to the official prompt. The native-video variant and the protocol-gap row of ADR-0008 apply only if a Gemini provider is used.
3. **Student:** Qwen3-VL-8B-Instruct, unchanged (ADR-0011).
4. **Qwen3 everywhere else:** cross-Target point is Qwen3-VL-4B-Instruct; the audio describer fallback is Qwen3-Omni-30B-A3B-Instruct; the Qwen2-Audio same-family ablation is dropped. Qwen2.5-VL-7B is no longer run.
5. **Proxy judge:** Gemini 3 Flash by default, now a different family from the Builder; an OpenAI scorer is selectable.

## Consequences

- Pipeline calibration against the published Qwen2.5-VL-7B leaderboard numbers (28.4 / 64.8 / 46.6) is no longer part of the default plan. The official judge test-retest (ADR-0007) becomes the only noise-floor measurement; if a leaderboard-anchored calibration is wanted later, a one-off Qwen2.5-VL-7B Direct row on the subset can be added as an optional P2 item.
- `OPENAI_API_KEY` is needed for the Builder, the teacher row, and the official judge; `GEMINI_API_KEY` only for the proxy judge.
- E8 (Builder strength) gains a natural second point: the earlier Gemini run versus this GPT run, provided both use the same frozen Stage 1 scaffold and construction data.
