# ADR-0008: Gemini Direct upper-bound protocol

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

H3 compares Qwen3-VL-8B with the harness against Gemini 3.1 Pro answering directly. Gemini accepts video with its audio track natively, while the Target receives frames plus text derived from audio. The review re-costed the official judge at $150–250 per full run and found that a full Gemini run (inference plus judge) is the most expensive single item in the plan while not being part of the paper's contribution. It also found that the original H3 criterion, "retain 80% of Gemini's performance", is already met by the plain Qwen2.5-VL-7B baseline on the CoT score (46.6 / 56.0 = 83%), so the criterion had to change (ADR-0012).

## Decision

1. **Primary (P0).** Gemini 3.1 Pro Direct runs with native video (audio track included), the official prompt verbatim, temperature 0, on the **pre-registered 1,000-item subset**, scored by the official judge. Every ratio involving Gemini (Gap Closed, Gemini Performance Retained) is computed on these 1,000 items, paired with the Qwen rows on the same items.
2. **Protocol-gap row (P1).** Gemini Direct also runs on the same subset with exactly the Target's input protocol: the same sampled frames and the same audio-derived text from the shared cache. The difference between the two rows is the protocol gap.
3. **Full run (P2).** Gemini Direct on all 6,500 items runs only if the cost ledger shows at least $350 unspent after every P0 and P1 row is complete. Subset numbers are reported with their confidence intervals and are never extrapolated to the full set.
4. Gemini responses are logged with response ids and model version strings because the API is not fully deterministic.

## Consequences

- The shared preprocessing cache must render frames and audio text into a Gemini request without any harness logic.
- The H3 confidence interval is wider than a full-run interval would be; ADR-0013 states the detectable effect size.
