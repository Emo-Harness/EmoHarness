# ADR-0002: Compute placement and API budget

Date: 2026-09-06. Amended 2026-09-07. Status: accepted.

## Context

The harness calls the Target several times per item and the experiment plan needs many full-benchmark runs (baselines, harness, ablations, cross-target). The API budget for Builder and official judge is about $1,000 and must be spent carefully.

## Decision

1. The Target (Qwen3-VL-8B) and all audio tools run on local or cluster GPUs behind a vLLM server with deterministic decoding (temperature 0, fixed seed).
2. The official judge (GPT-4.1 + GPT-4o) is reserved for the final full-benchmark runs of methods that appear in the results tables.
3. Construction-time scoring uses local exact-match Recognition plus a cheap proxy for Reasoning (objective in ADR-0005, proxy in ADR-0007).
4. Every Builder, Target, judge, and proxy call is logged with token counts and cost so spend is visible per experiment.
5. Rows that do not carry a main claim run on a fixed, pre-registered 1,000-item subset; the full/subset allocation, the cost model, and the cut line are in ADR-0009. The official judge is re-costed there at $150–250 per full run.

## Consequences

- vLLM serving, frame sampling, and audio preprocessing must be cached so that repeated methods reuse the same inputs.
- A cost ledger is a first-class artifact under `results/`.
