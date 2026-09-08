# ADR-0011: Target variant, input protocol, and per-item call budget

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Decision

1. **Target:** Qwen3-VL-8B-Instruct, served by vLLM in bf16 on the single 80 GB GPU. Greedy decoding, temperature 0, fixed seed. The Thinking variant is a P2 Direct row on the subset.
2. **Video input:** native video through the Qwen3-VL processor at 2 fps, between 4 and 32 frames, using the processor's default per-frame pixel budget. The exact processor configuration is written into every run manifest. Direct rows use this path with the official prompt and one call. The M1 calibration with Qwen2.5-VL-7B checks that these settings reproduce the published Direct numbers within the judge noise floor.
3. **Harness-side visual additions:** the Stage 2 sampling config may enable up to 4 face-crop images of salient frames as extra image inputs. The human code supports the toggle; the Builder decides whether to use it.
4. **Per-item Target-call budget:** expected 4, hard cap 6, retries included. Each stage gets at most one retry with a stricter format reminder when its output fails schema validation. When the cap is reached, the controller completes the answer from whatever evidence and hypotheses exist; no extra call is made.
5. **Structured outputs:** every Target call returns JSON validated against the evidence or hypothesis schema; validation failures are logged per stage so that parse-failure rates appear in the results.
6. **Budget-matched controls:** Best-of-4 and sequential refinement use four calls each with the L2 inputs (ADR-0009).
7. **Second family Target (P2):** InternVL3.5-8B, same serving and input protocol, Direct and + L5 on the subset.
8. **Gemini model ids:** Gemini 3.1 Pro for Builder and Direct, Gemini 3 Flash for the proxy judge, all at temperature 0 with response ids logged.

## Consequences

- All methods share one preprocessing cache and one content-addressed Target-call cache (ADR-0006).
- Every Target, Builder, judge, and proxy call is logged with model id, parameters, token counts, latency, and cost.
